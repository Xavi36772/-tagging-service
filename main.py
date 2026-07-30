"""
Fase 3 + 5: Microservicio de Predicción de Tags y Recomendación por Clustering.

Endpoints:
  POST /predict-tags       → Predice top-5 tags de una sinopsis
  POST /predict-tags-batch  → Predice tags en lote
  GET  /works/{id}/similar  → Obras similares por Jaccard
  POST /reindex-clusters    → Recalcula clustering sobre todas las obras
  GET  /health              → Health check
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from contextlib import asynccontextmanager

# ── Config ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))
TAXONOMY_PATH = Path(os.environ.get("TAXONOMY_PATH", "taxonomy.json"))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MAX_LEN = 256
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Modelo (igual que en train.py) ────────────────────────────────────
class BETOMultiLabel(nn.Module):
    """BETO with a deeper classifier head.
    CLS → Dense(768→512) → GELU → Dropout → Dense(512→num_labels)
    """
    def __init__(self, model_name: str, num_labels: int, hidden_dim: int = 512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels),
        )
        # Multi-sample dropout (only active during training in train.py)
        self._dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier_out = nn.Linear(bert_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.use_multisample = False

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_output)
        return logits


# ── State global ──────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.model: Optional[BETOMultiLabel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.taxonomy: list[str] = []
        self.tag2idx: dict[str, int] = {}
        self.thresholds: np.ndarray | None = None
        self.supabase: Client | None = None


state = AppState()


# ── Startup / Shutdown ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar taxonomía
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        state.taxonomy = json.load(f)
    state.tag2idx = {t.lower(): i for i, t in enumerate(state.taxonomy)}
    print(f"Taxonomía cargada: {len(state.taxonomy)} etiquetas")

    # Cargar modelo
    model_weights = MODEL_DIR / "pytorch_model.bin"
    if model_weights.exists():
        base_model = "dccuchile/bert-base-spanish-wwm-cased"
        state.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR) if (MODEL_DIR / "tokenizer_config.json").exists() else base_model)
        state.model = BETOMultiLabel(base_model, len(state.taxonomy)).to(device)
        state.model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
        state.model.eval()
        print(f"Modelo cargado desde {model_weights}")

        # Cargar thresholds
        th_path = MODEL_DIR / "thresholds.npy"
        if th_path.exists():
            state.thresholds = np.load(th_path)
            print(f"Thresholds cargados ({len(state.thresholds)} etiquetas)")
        else:
            state.thresholds = np.full(len(state.taxonomy), 0.5)
            print("Usando threshold por defecto = 0.5")
    else:
        print("ADVERTENCIA: No se encontró modelo fine-tuned. Solo funcionarán endpoints de clustering.")

    # Conectar Supabase
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        state.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Conectado a Supabase")
    else:
        print("ADVERTENCIA: Sin credenciales Supabase. Endpoints que requieren DB fallarán.")

    yield


app = FastAPI(title="Kotoba Tagging Service", lifespan=lifespan)


# ── Schemas ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    synopsis: str


class PredictResponse(BaseModel):
    tags: list[str] = []
    probabilities: list[float] = []
    all_probabilities: list[float] = []
    error: str | None = None


class PredictBatchRequest(BaseModel):
    synopses: list[str]


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


class SimilarResponse(BaseModel):
    work_id: str
    title: str
    jaccard_score: float
    tags: list[str]


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    top_tags: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "taxonomy_size": len(state.taxonomy),
        "device": str(device),
    }


@app.post("/predict-tags", response_model=PredictResponse)
def predict_tags(req: PredictRequest):
    if state.model is None:
        return PredictResponse(error="Modelo no cargado. Ejecuta train.py primero.")

    if not req.synopsis.strip():
        raise HTTPException(400, "La sinopsis no puede estar vacía")

    encoded = state.tokenizer(
        req.synopsis,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = state.model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    thresholds = state.thresholds if state.thresholds is not None else np.full(len(state.taxonomy), 0.5)

    # Obtener top-K por probabilidad (si pasa threshold mínimo)
    candidates = [(i, float(probabilities[i])) for i in range(len(state.taxonomy))
                  if probabilities[i] >= thresholds[i] * 0.8]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:TOP_K]

    # Si no hay suficientes, tomar las de mayor probabilidad aunque no pasen threshold
    if len(top) < 3:
        all_sorted = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
        seen = set(i for i, _ in top)
        for i, p in all_sorted:
            if i not in seen:
                top.append((i, float(p)))
                seen.add(i)
            if len(top) >= TOP_K:
                break

    tags = [state.taxonomy[i] for i, _ in top]
    probs = [p for _, p in top]

    return PredictResponse(
        tags=tags,
        probabilities=probs,
        all_probabilities=[float(p) for p in probabilities],
    )


@app.post("/predict-tags-batch", response_model=PredictBatchResponse)
def predict_tags_batch(req: PredictBatchRequest):
    results = [predict_tags(PredictRequest(synopsis=s)) for s in req.synopses]
    return PredictBatchResponse(results=results)


def _jaccard_similarity(tags_a: set[str], tags_b: set[str]) -> float:
    intersection = len(tags_a & tags_b)
    union = len(tags_a | tags_b)
    return intersection / union if union > 0 else 0.0


@app.get("/works/{work_id}/similar")
def similar_works(work_id: str, limit: int = 10):
    if state.supabase is None:
        raise HTTPException(503, "Supabase no configurado")

    # Obtener tags de la obra actual
    resp = state.supabase.table("works").select("id, title, tags").eq("id", work_id).neq("status", "draft").maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "Obra no encontrada")

    current = resp.data
    current_tags = set(t.lower() for t in (current.get("tags") or []))

    if not current_tags:
        raise HTTPException(400, "La obra no tiene etiquetas asignadas")

    # Obtener todas las demás obras con tags
    resp_all = state.supabase.table("works").select("id, title, tags").neq("status", "draft").neq("id", work_id).execute()
    if not resp_all.data:
        return {"works": [], "method": "jaccard"}

    scored = []
    for w in resp_all.data:
        w_tags = set(t.lower() for t in (w.get("tags") or []))
        if not w_tags:
            continue
        sim = _jaccard_similarity(current_tags, w_tags)
        if sim > 0:
            scored.append({
                "work_id": w["id"],
                "title": w["title"],
                "jaccard_score": round(sim, 4),
                "tags": w.get("tags", []),
            })

    scored.sort(key=lambda x: x["jaccard_score"], reverse=True)

    return {
        "query_work_id": work_id,
        "query_tags": current.get("tags", []),
        "method": "jaccard",
        "works": scored[:limit],
    }


@app.post("/reindex-clusters")
def reindex_clusters(n_clusters: int = 8):
    """Ejecuta K-Means sobre los vectores de tags de todas las obras y guarda cluster_id en Supabase."""
    if state.supabase is None:
        raise HTTPException(503, "Supabase no configurado")

    # Obtener todas las obras con tags
    resp = state.supabase.table("works").select("id, tags").neq("status", "draft").execute()
    if not resp.data:
        raise HTTPException(404, "No hay obras publicadas")

    works = resp.data
    tag_vector_map = {}
    matrix_rows = []

    for w in works:
        tags = set(t.lower() for t in (w.get("tags") or []))
        if not tags:
            continue
        # One-hot vector sobre la taxonomía
        vec = np.zeros(len(state.taxonomy), dtype=np.float32)
        for i, tag in enumerate(state.taxonomy):
            if tag.lower() in tags:
                vec[i] = 1.0
        tag_vector_map[w["id"]] = vec
        matrix_rows.append(vec)

    if len(matrix_rows) < n_clusters:
        n_clusters = max(2, len(matrix_rows))
        print(f"Cluster count ajustado a {n_clusters} (pocas obras con tags)")

    X = np.array(matrix_rows)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Guardar cluster_id en Supabase (columna opcional, podemos crear tabla o metadata)
    work_ids = list(tag_vector_map.keys())
    for work_id, cluster_id in zip(work_ids, labels):
        state.supabase.table("work_clusters").upsert(
            {"work_id": work_id, "cluster_id": int(cluster_id)},
            on_conflict="work_id",
        ).execute()

    # Estadísticas de los clústeres
    clusters_info = []
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_work_ids = [work_ids[i] for i in range(len(work_ids)) if mask[i]]
        # Tags más frecuentes en este clúster
        freq = np.sum(X[mask], axis=0)
        top_tag_indices = np.argsort(freq)[-5:][::-1]
        top_tags = [state.taxonomy[i] for i in top_tag_indices if freq[i] > 0]
        clusters_info.append({
            "cluster_id": int(cid),
            "size": int(mask.sum()),
            "top_tags": top_tags,
        })

    return {
        "n_clusters": n_clusters,
        "n_works": len(work_ids),
        "silhouette_score": None,  # Se calcula offline en cluster.py
        "clusters": clusters_info,
    }

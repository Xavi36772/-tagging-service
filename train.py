"""
Fase 2: Fine-Tuning de BETO para Clasificación Multietiqueta.

Uso: python train.py [--data-dir dataset] [--taxonomy taxonomy.json] [--epochs 10] [--batch-size 16] [--lr 2e-5]
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score


class TagDataset(Dataset):
    def __init__(self, data, tokenizer, tag2idx, max_len=256):
        self.texts = [item["synopsis"] for item in data]
        self.labels = torch.zeros((len(data), len(tag2idx)), dtype=torch.float32)
        for i, item in enumerate(data):
            for tag in item["tags"]:
                idx = tag2idx.get(tag.lower())
                if idx is not None:
                    self.labels[i, idx] = 1.0
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


class BETOMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag_names: list[str]):
    """Calcula métricas de clasificación multietiqueta."""
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    # F1 por etiqueta individual
    per_tag = {}
    for i, tag in enumerate(tag_names):
        per_tag[tag] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    metrics["per_tag_f1"] = per_tag
    return metrics


def find_optimal_thresholds(model, val_loader, device, tag_names):
    """Busca el threshold óptimo por etiqueta maximizando F1 en validación."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    thresholds = []
    for i in range(len(tag_names)):
        best_f1 = 0.0
        best_th = 0.5
        for th in np.arange(0.1, 0.95, 0.05):
            pred = (all_logits[:, i] >= th).astype(int)
            f1 = f1_score(all_labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        thresholds.append(best_th)

    return np.array(thresholds)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning BETO multi-label classification")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directorio del dataset")
    parser.add_argument("--taxonomy", type=str, default="taxonomy.json", help="Archivo de taxonomía")
    parser.add_argument("--model-name", type=str, default="dccuchile/bert-base-spanish-wwm-cased",
                        help="Modelo BETO base")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Longitud máxima de tokens")
    parser.add_argument("--output-dir", type=str, default="model", help="Directorio de salida del modelo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Cargar taxonomía y dataset
    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    tag2idx = {t.lower(): i for i, t in enumerate(taxonomy)}
    print(f"Taxonomía: {len(taxonomy)} etiquetas")

    with open(Path(args.data_dir) / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(Path(args.data_dir) / "val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Tokenizer + modelo
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BETOMultiLabel(args.model_name, len(taxonomy)).to(device)

    train_dataset = TagDataset(train_data, tokenizer, tag2idx, args.max_len)
    val_dataset = TagDataset(val_data, tokenizer, tag2idx, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Evaluación en validación
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) >= 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(batch["labels"].cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        metrics = compute_metrics(y_true, y_pred, taxonomy)
        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f} | F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Precision Macro: {metrics['precision_macro']:.4f} | Recall Macro: {metrics['recall_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            # Guardar mejor modelo
            torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(output_dir)
            print(f"  -> Mejor modelo guardado (F1 Macro: {best_f1:.4f})")

        # Top-5 peores etiquetas por F1
        sorted_tags = sorted(metrics["per_tag_f1"].items(), key=lambda x: x[1])
        worst = sorted_tags[:5]
        print(f"  Peores 5 etiquetas: {[(t, f'{s:.3f}') for t, s in worst]}")

    # Encontrar thresholds óptimos
    print("\nCalculando thresholds óptimos por etiqueta...")
    model.load_state_dict(torch.load(output_dir / "pytorch_model.bin"))
    thresholds = find_optimal_thresholds(model, val_loader, device, taxonomy)
    np.save(output_dir / "thresholds.npy", thresholds)

    # Evaluación final con thresholds óptimos
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    y_logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred_opt = (y_logits >= thresholds.reshape(1, -1)).astype(int)

    metrics_opt = compute_metrics(y_true, y_pred_opt, taxonomy)
    print("\n=== MÉTRICAS FINALES (thresholds óptimos) ===")
    print(f"F1 Macro: {metrics_opt['f1_macro']:.4f}")
    print(f"F1 Micro: {metrics_opt['f1_micro']:.4f}")
    print(f"Hamming Loss: {metrics_opt['hamming_loss']:.4f}")

    # Guardar métricas como JSON
    metrics_clean = {k: v for k, v in metrics_opt.items() if k != "per_tag_f1"}
    metrics_clean["per_tag_f1"] = {t: float(v) for t, v in metrics_opt["per_tag_f1"].items()}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, ensure_ascii=False, indent=2)

    print(f"\nModelo y métricas guardados en {output_dir}/")


if __name__ == "__main__":
    main()

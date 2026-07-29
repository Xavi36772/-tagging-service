"""
Fase 1: Generación de Dataset Sintético usando Grok (xAI API).

Genera pares (sinopsis, [tags]) a partir de un catálogo cerrado de etiquetas.
Uso: python generate_dataset.py [--samples N] [--output-dir dataset]
"""

import json
import os
import time
import argparse
import random
from pathlib import Path
from httpx import Client, HTTPError

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = "grok-beta"

random.seed(42)


def load_taxonomy(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(taxonomy: list[str], primary_tags: list[str]) -> str:
    tax_str = ", ".join(taxonomy)
    focus = ", ".join(primary_tags)
    return f"""Eres un escritor creativo. Genera una sinopsis de historia original en español de aproximadamente 3 a 5 párrafos.

La historia debe estar FUERTEMENTE INSPIRADA en estas etiquetas principales: {focus}.
Puede incluir elementos de otras etiquetas del catálogo si enriquece la historia.

Catálogo completo de etiquetas disponibles: {tax_str}

Devuelve EXCLUSIVAMENTE un objeto JSON válido con esta estructura exacta (sin markdown, sin explicaciones):
{{"synopsis": "texto de la sinopsis aquí", "tags": ["etiqueta1", "etiqueta2", ...]}}

La sinopsis debe ser original, detallada y coherente. Las etiquetas deben ser EXACTAMENTE del catálogo proporcionado, entre 3 y 5 etiquetas, e INCLUIR AL MENOS 2 de las etiquetas principales: {focus}."""


def make_balanced_batch_prompts(taxonomy: list[str], samples: int) -> list[str]:
    """Genera prompts balanceados: cada etiqueta aparece como 'principal' aprox. samples/len(taxonomy) veces."""
    n_tags = len(taxonomy)
    per_tag = max(1, samples // n_tags)
    prompts = []

    for tag in taxonomy:
        # Combinar con 1-2 etiquetas adicionales aleatorias para variedad
        for _ in range(per_tag):
            others = random.sample([t for t in taxonomy if t != tag], min(2, n_tags - 1))
            primary = [tag] + others
            prompts.append(make_prompt(taxonomy, primary))

    # Completar hasta samples si faltan
    while len(prompts) < samples:
        primary = random.sample(taxonomy, min(3, n_tags))
        prompts.append(make_prompt(taxonomy, primary))

    random.shuffle(prompts)
    return prompts[:samples]


def call_grok(api_key: str, prompt: str) -> dict | None:
    """Llama a la API de Grok y parsea la respuesta JSON."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": XAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 600,
    }

    with Client(timeout=60) as client:
        try:
            resp = client.post(XAI_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Intentar extraer JSON de la respuesta (puede venir con markdown)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("\n", 1)[0] if content.endswith("```") else content
                if content.endswith("```"):
                    content = content[:-3].strip()

            return json.loads(content)
        except (HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"  Error: {e}")
            return None


def validate_entry(entry: dict, taxonomy: list[str]) -> bool:
    """Valida que la entrada tenga sinopsis y tags válidos."""
    if not isinstance(entry, dict):
        return False
    if "synopsis" not in entry or "tags" not in entry:
        return False
    if not isinstance(entry["synopsis"], str) or len(entry["synopsis"].strip()) < 50:
        return False
    if not isinstance(entry["tags"], list) or len(entry["tags"]) < 3:
        return False
    # Normalizar tags: capitalizar primera letra
    entry["tags"] = [t.strip().title() for t in entry["tags"] if t.strip()]
    # Filtrar tags que no están en la taxonomía
    valid = set(t.lower() for t in taxonomy)
    entry["tags"] = [t for t in entry["tags"] if t.lower() in valid]
    if len(entry["tags"]) < 3:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético con Grok")
    parser.add_argument("--samples", type=int, default=1500, help="Número de ejemplos a generar")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Directorio de salida")
    parser.add_argument("--api-key", type=str, default=None, help="API key de xAI (o variable GROK_API_KEY)")
    parser.add_argument("--taxonomy", type=str, default="taxonomy.json", help="Archivo de taxonomía")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROK_API_KEY")
    if not api_key:
        print("ERROR: Se necesita GROK_API_KEY (variable de entorno o --api-key)")
        return

    taxonomy = load_taxonomy(args.taxonomy)
    print(f"Taxonomía cargada: {len(taxonomy)} etiquetas")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = make_balanced_batch_prompts(taxonomy, args.samples)
    print(f"Se generarán {len(prompts)} prompts balanceados")

    entries = []
    errors = 0

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generando...", end=" ")
        result = call_grok(api_key, prompt)

        if result and validate_entry(result, taxonomy):
            entries.append(result)
            print(f"OK -> {result['tags']}")
        else:
            errors += 1
            print(f"FALLÓ (válido: {result if result else 'None'})")

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\nTotal generados: {len(entries)} | Errores: {errors}")

    # Split train/val (80/20)
    random.shuffle(entries)
    split = int(len(entries) * 0.8)
    train = entries[:split]
    val = entries[split:]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)}")
    print("Dataset generado exitosamente.")


if __name__ == "__main__":
    main()

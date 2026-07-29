"""
Generate diverse natural-language synopses using Groq API (free).
Combines with template-based dataset for improved model accuracy.
Target: F1 > 0.65 by adding varied, non-template data.

Usage: python generate_groq_dataset.py
"""

import json
import os
import time
import random
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

TAXONOMY = [
    "Acción", "Aventura", "Romance", "Drama", "Comedia", "Terror", "Suspenso",
    "Misterio", "Ciencia Ficción", "Fantasía", "Distopía", "Cyberpunk",
    "Realismo Mágico", "Histórico", "Mitología", "Apocalíptico", "Thriller Psicológico",
    "Crimen", "Western", "Bélico", "Superhéroes", "Steampunk", "Space Opera",
    "Slice of Life", "Coming of Age", "LGBTQ+", "Feminismo", "Filosófico",
    "Religioso", "Humor Negro", "Parodia", "Infantil", "Juvenil", "New Adult",
    "Poesía", "Epistolar", "Antología", "Leyendas Urbanas", "Survival", "Artes Marciales"
]

PROMPT_TEMPLATE = """Genera 5 sinopsis de historias literarias en español que sean DIVERSAS y NATURALES, como si fueran escritas por diferentes autores humanos.

REQUISITOS IMPORTANTES:
- Cada sinopsis debe ser ORIGINAL y NO usar plantillas
- Debe sonar a texto escrito por un humano real, variado en estilo y estructura
- Debe tener entre 30 y 80 palabras cada una
- NO uses frases hechas como "descubre que posee un poder ancestral" o "se embarca en un viaje"
- Varía la estructura: a veces empieza con personaje, a veces con evento, a veces con descripción del mundo

Los tags principales que deben cubrir estas sinopsis son variados (incluye distintos géneros).

Para cada sinopsis, asigna 2-5 tags de esta lista: {tags}

RESPONDE ÚNICAMENTE EN JSON CON ESTE FORMATO, SIN OTRO TEXTO:
{{"entries": [
    {{"synopsis": "sinopsis 1 aquí", "tags": ["Tag1", "Tag2"]}},
    {{"synopsis": "sinopsis 2 aquí", "tags": ["Tag3", "Tag4", "Tag5"]}}
]}}

Genera 5 entradas ahora, cada una con tags variados y diferentes entre sí."""


def call_groq(prompt: str, retries: int = 3) -> dict | None:
    import httpx
    for attempt in range(retries):
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9,
                    "max_tokens": 2000,
                },
                timeout=30
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Extract JSON from response
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
    return None


def main():
    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data to avoid duplication
    existing = set()
    for split in ["train.json", "val.json"]:
        fpath = output_dir / split
        if fpath.exists():
            with open(fpath) as f:
                for entry in json.load(f):
                    existing.add(entry["synopsis"][:50])

    print(f"Existing unique synopses (prefix): {len(existing)}")

    tags_csv = ", ".join(TAXONOMY)
    all_entries = []
    api_calls = 0
    target_batches = 40  # 40 * 5 = 200 new entries

    # Focus tags that need more data (low-performing ones)
    focus_tags = ["Romance", "Terror", "Suspenso", "Misterio", "Crimen",
                  "Thriller Psicológico", "Histórico", "Mitología", "Western",
                  "Slice of Life", "Coming of Age", "LGBTQ+", "Ciencia Ficción",
                  "Distopía", "Cyberpunk", "Space Opera", "Realismo Mágico",
                  "Apocalíptico", "Filosófico", "Survival", "Artes Marciales",
                  "Leyendas Urbanas", "Infantil", "Poesía", "Epistolar"]

    print(f"Generating {target_batches} batches via Groq API...")
    print(f"Focus tags: {focus_tags[:5]}... (total {len(focus_tags)})")

    for batch_idx in range(target_batches):
        # Rotate focus: each batch emphasizes different tags
        start_idx = (batch_idx * 3) % len(focus_tags)
        batch_focus = focus_tags[start_idx:start_idx + 3]

        prompt = PROMPT_TEMPLATE.format(tags=tags_csv)
        prompt = f"""Genera 5 sinopsis DIVERSAS de historias literarias en español.

REQUISITOS:
- Sinopsis ORIGINALES que suenen a texto humano real
- 30-80 palabras cada una
- Variedad de géneros, estructuras y estilos
- Asegúrate de que al menos 2 de las 5 sinopsis usen estos tags como PRIMARIOS: {batch_focus}

Tags disponibles: {tags_csv}

RESPONDE EN JSON:
{{"entries": [
    {{"synopsis": "...", "tags": ["Tag1", "Tag2"]}},
    ...
]}}"""

        print(f"  Batch {batch_idx+1}/{target_batches} [focus: {batch_focus[0]}]...", end=" ")
        result = call_groq(prompt)
        if result and "entries" in result:
            new_count = 0
            for entry in result["entries"]:
                prefix = entry["synopsis"][:50]
                if prefix not in existing:
                    existing.add(prefix)
                    all_entries.append(entry)
                    new_count += 1
            print(f"Got {len(result['entries'])} entries, {new_count} new")
        else:
            print(f"Failed or no entries")

        api_calls += 1
        # Rate limit: 30 req/min for llama-3.3-70b, use 2s delay
        time.sleep(2.5)

    print(f"\nTotal new entries from Groq: {len(all_entries)}")

    # Save Groq-only dataset
    with open(output_dir / "groq_generated.json", "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Merge with template dataset
    template_entries = []
    for split in ["train.json", "val.json"]:
        fpath = output_dir / split
        if fpath.exists():
            with open(fpath) as f:
                template_entries.extend(json.load(f))

    print(f"Template entries: {len(template_entries)}")
    print(f"Groq entries: {len(all_entries)}")

    # Combine: all template + all groq
    combined = template_entries + all_entries
    random.shuffle(combined)
    print(f"Combined total: {len(combined)}")

    # Tag distribution stats
    tag_counts = {}
    for entry in combined:
        for tag in entry["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\nTag distribution:")
    for tag in sorted(tag_counts, key=tag_counts.get, reverse=True):
        print(f"  {tag}: {tag_counts[tag]}")

    # Split 80/20
    split = int(len(combined) * 0.8)
    train = combined[:split]
    val = combined[split:]

    with open(output_dir / "train_combined.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val_combined.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"\nTrain: {len(train)} | Val: {len(val)}")
    print("Combined dataset saved to train_combined.json / val_combined.json")


if __name__ == "__main__":
    main()

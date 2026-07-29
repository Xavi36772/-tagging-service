"""
Generate diverse synopses using Groq API.
"""
import json, os, time, random, re, httpx
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable required")

TAXONOMY = [
    "Acci\u00f3n", "Aventura", "Romance", "Drama", "Comedia", "Terror", "Suspenso",
    "Misterio", "Ciencia Ficci\u00f3n", "Fantas\u00eda", "Distop\u00eda", "Cyberpunk",
    "Realismo M\u00e1gico", "Hist\u00f3rico", "Mitolog\u00eda", "Apocal\u00edptico",
    "Thriller Psicol\u00f3gico", "Crimen", "Western", "B\u00e9lico", "Superh\u00e9roes",
    "Steampunk", "Space Opera", "Slice of Life", "Coming of Age", "LGBTQ+",
    "Feminismo", "Filos\u00f3fico", "Religioso", "Humor Negro", "Parodia",
    "Infantil", "Juvenil", "New Adult", "Poes\u00eda", "Epistolar", "Antolog\u00eda",
    "Leyendas Urbanas", "Survival", "Artes Marciales"
]

def normalize_tag(tag):
    """Match free-form tag to taxonomy."""
    t = tag.lower().strip()
    for tax in TAXONOMY:
        if tax.lower() == t or tax.lower().startswith(t) or t.startswith(tax.lower()[:5]):
            return tax
    # Fuzzy matching
    for tax in TAXONOMY:
        if len(set(t.split()) & set(tax.lower().split())) >= 2:
            return tax
    return tag

def call_groq(prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "SOLO respondes con JSON v\u00e1lido, nada m\u00e1s."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.9,
                    "max_tokens": 2500,
                },
                timeout=30
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 10
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Extract JSON
            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            first = content.find("{")
            last = content.rfind("}")
            if first >= 0 and last > first:
                content = content[first:last+1]
            content = re.sub(r",\s*}", "}", content)
            content = re.sub(r",\s*]", "]", content)
            return json.loads(content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
    return None

def main():
    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing prefixes
    existing = set()
    for f in ["train.json", "val.json"]:
        p = output_dir / f
        if p.exists():
            for e in json.load(open(p, "r", encoding="utf-8")):
                existing.add(e["synopsis"][:40])

    print(f"Existing: {len(existing)} unique prefixes", flush=True)

    # Target tags that need help
    needs_help = [
        "Steampunk", "Space Opera", "LGBTQ+", "Feminismo",
        "Infantil", "Religioso", "Slice of Life", "Poes\u00eda",
        "Epistolar", "Antolog\u00eda", "Leyendas Urbanas", "Survival",
        "Artes Marciales", "Filos\u00f3fico", "Humor Negro",
        "Parodia", "Western", "Realismo M\u00e1gico"
    ]

    call_count = 0
    all_entries = []
    tags_csv = ", ".join(TAXONOMY)

    for i in range(100):
        focus = needs_help[i % len(needs_help)]
        prompt = (
            f"Genera 5 sinopsis en espa\u00f1ol originales. "
            f"Al menos 2 con g\u00e9nero {focus}. "
            f"Tags disponibles: {tags_csv[:250]}. "
            "JSON: {\"entries\": [{\"synopsis\": \"...\", \"tags\": [\"Tag\"]}]}"
        )

        print(f"  {i+1}/200 [focus: {focus}]...", end=" ", flush=True)
        result = call_groq(prompt)
        if result and "entries" in result:
            new_count = 0
            for e in result["entries"]:
                synopsis = e.get("synopsis", "")
                tags = [normalize_tag(t) for t in e.get("tags", []) if t in TAXONOMY]
                if synopsis and tags and synopsis[:40] not in existing:
                    existing.add(synopsis[:40])
                    all_entries.append({"synopsis": synopsis, "tags": tags})
                    new_count += 1
            print(f"{new_count} new", flush=True)
        else:
            print("failed", flush=True)

        call_count += 1
        time.sleep(1.0)

        # Save after every batch (in case of timeout)
        with open(output_dir / "groq_natural.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_entries)} Groq entries", flush=True)

    # Merge with existing
    template = []
    for f in ["train.json", "val.json"]:
        p = output_dir / f
        if p.exists():
            template.extend(json.load(open(p, "r", encoding="utf-8")))

    print(f"Template: {len(template)}, Groq: {len(all_entries)}", flush=True)

    combined = template + all_entries
    random.shuffle(combined)

    split = int(len(combined) * 0.8)
    train = combined[:split]
    val = combined[split:]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    with open(output_dir / "groq_natural.json", "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)}", flush=True)

    # Tag counts
    from collections import Counter
    c = Counter()
    for e in combined:
        for t in e["tags"]:
            c[t] += 1
    print("Distribution:")
    for t, n in c.most_common(15):
        print(f"  {t}: {n}")

if __name__ == "__main__":
    main()

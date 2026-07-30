"""
Generate targeted Groq synopses focused on WEAK tags.
Generates 100+ synopses per weak tag to fix the long-tail distribution problem.

Usage:
  GROQ_API_KEY=xxx python generate_groq_targeted.py [--calls 300] [--batch-size 5]

Requires: httpx
"""
import json, os, time, random, re, httpx, argparse
from pathlib import Path
from collections import Counter

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable required")

TAXONOMY = [
    "Acción", "Aventura", "Romance", "Drama", "Comedia", "Terror", "Suspenso",
    "Misterio", "Ciencia Ficción", "Fantasía", "Distopía", "Cyberpunk",
    "Realismo Mágico", "Histórico", "Mitología", "Apocalíptico",
    "Thriller Psicológico", "Crimen", "Western", "Bélico", "Superhéroes",
    "Steampunk", "Space Opera", "Slice of Life", "Coming of Age", "LGBTQ+",
    "Feminismo", "Filosófico", "Religioso", "Humor Negro", "Parodia",
    "Infantil", "Juvenil", "New Adult", "Poesía", "Epistolar", "Antología",
    "Leyendas Urbanas", "Survival", "Artes Marciales"
]

# Tags with F1 < 0.35 from evaluation results — these need the most help
WEAK_TAGS = [
    "Epistolar", "Infantil", "Antología", "LGBTQ+", "Poesía",
    "Leyendas Urbanas", "Parodia", "Artes Marciales", "Humor Negro",
    "Feminismo", "New Adult", "Religioso", "Survival", "Juvenil",
    "Space Opera", "Steampunk", "Aventura", "Comedia",
    "Thriller Psicológico", "Slice of Life", "Apocalíptico",
]

# Genre-specific prompting hints to guide Groq toward diverse, accurate synopses
TAG_HINTS = {
    "Epistolar": "narrada a través de cartas, correos, diarios, mensajes o documentos",
    "Infantil": "protagonistas niños, lenguaje sencillo, moralejas, temas de amistad y aprendizaje",
    "Antología": "colección de relatos cortos, cuentos reunidos, múltiples historias independientes",
    "LGBTQ+": "relaciones LGBT, identidad de género, diversidad sexual, orgullo",
    "Poesía": "versos, estrofas, prosa poética, rimas, metáforas líricas",
    "Leyendas Urbanas": "mitos modernos, historias que se cuentan de noche, creepypastas, folklore contemporáneo",
    "Parodia": "sátira, humor que se burla de géneros, exageración cómica de tropos",
    "Artes Marciales": "kung fu, karate, dojos, torneos de lucha, maestros marciales",
    "Humor Negro": "comedia sobre temas oscuros, muerte tratada con ironía, sarcasmo macabro",
    "Feminismo": "empoderamiento femenino, lucha por igualdad, patriarcado, sororidad",
    "New Adult": "adultos jóvenes 18-25, universidad, primer trabajo, independencia",
    "Religioso": "fe, iglesia, monasterios, dilemas espirituales, textos sagrados",
    "Survival": "sobrevivir en condiciones extremas, naturaleza hostil, recurso limitados",
    "Juvenil": "adolescentes como protagonistas, instituto, primer amor adolescente, rebeldía",
    "Space Opera": "batallas espaciales, imperios galácticos, naves interestelares, planetas lejanos",
    "Steampunk": "tecnología de vapor, era victoriana alternativa, engranajes, dirigibles",
    "Aventura": "viajes, exploración, búsquedas épicas, mundos por descubrir",
    "Comedia": "situaciones hilarantes, malentendidos cómicos, personajes excéntricos",
    "Thriller Psicológico": "manipulación mental, paranoia, giros inesperados, juegos psicológicos",
    "Slice of Life": "vida cotidiana, momentos pequeños, rutinas, relaciones familiares simples",
    "Apocalíptico": "fin del mundo, civilización colapsada, catástrofe global",
}

TAXONOMY_SET = set(t.lower() for t in TAXONOMY)
TAXONOMY_MAP = {t.lower(): t for t in TAXONOMY}


def normalize_tag(tag: str) -> str | None:
    """Match free-form tag to taxonomy. Returns None if no match."""
    t = tag.strip().lower()
    # Exact match
    if t in TAXONOMY_MAP:
        return TAXONOMY_MAP[t]
    # Prefix match
    for tax_lower, tax_orig in TAXONOMY_MAP.items():
        if tax_lower.startswith(t) or t.startswith(tax_lower):
            return tax_orig
    # Word overlap
    t_words = set(t.split())
    for tax_lower, tax_orig in TAXONOMY_MAP.items():
        tax_words = set(tax_lower.split())
        if len(t_words & tax_words) >= 1 and len(tax_words) > 1:
            return tax_orig
    return None


def call_groq(prompt: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": (
                            "Eres un escritor experto en literatura en español. "
                            "SOLO respondes con JSON válido, nada más. "
                            "Cada sinopsis debe tener 3-6 oraciones variadas y originales."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.95,
                    "max_tokens": 3000,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 10
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Clean up JSON
            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            first = content.find("[")
            last = content.rfind("]")
            if first >= 0 and last > first:
                content = content[first:last + 1]
            else:
                first = content.find("{")
                last = content.rfind("}")
                if first >= 0 and last > first:
                    content = content[first:last + 1]
            content = re.sub(r",\s*}", "}", content)
            content = re.sub(r",\s*]", "]", content)
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return {"entries": parsed}
            return parsed
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
            else:
                print(f"    Error: {e}", flush=True)
    return None


def build_prompt(focus_tag: str, secondary_tags: list[str], batch_size: int = 5) -> str:
    """Build a prompt that generates diverse synopses for a specific tag."""
    hint = TAG_HINTS.get(focus_tag, "")
    secondary = ", ".join(secondary_tags[:4])

    return (
        f"Genera {batch_size} sinopsis de libros en español. "
        f"TODAS deben ser del género '{focus_tag}' ({hint}). "
        f"Combina '{focus_tag}' con otros géneros como: {secondary}. "
        f"Cada sinopsis: 3-6 oraciones, original, sin repetir tramas. "
        f"Varía ambientación (rural, urbano, futurista, histórico, fantástico). "
        f"Varía protagonistas (edad, género, profesión). "
        f"Los tags válidos son EXACTAMENTE: {', '.join(TAXONOMY)}. "
        f"Responde SOLO con JSON: "
        f'[{{"synopsis": "...", "tags": ["{focus_tag}", "OtroTag"]}}]'
    )


def main():
    parser = argparse.ArgumentParser(description="Generate targeted Groq synopses for weak tags")
    parser.add_argument("--calls", type=int, default=300,
                        help="Total API calls to make (default: 300, ~1500 synopses)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Synopses per API call (default: 5)")
    parser.add_argument("--output", type=str, default="dataset/groq_targeted.json",
                        help="Output file (default: dataset/groq_targeted.json)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls (default: 1.5)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing synopses to avoid duplicates
    existing_prefixes = set()
    for f in ["dataset/train.json", "dataset/val.json", "dataset/groq_natural.json"]:
        p = Path(f)
        if p.exists():
            for entry in json.load(open(p, "r", encoding="utf-8")):
                existing_prefixes.add(entry["synopsis"][:50])

    # Resume from partial output if exists
    all_entries = []
    if output_path.exists():
        all_entries = json.load(open(output_path, "r", encoding="utf-8"))
        for e in all_entries:
            existing_prefixes.add(e["synopsis"][:50])
        print(f"Resuming: {len(all_entries)} existing targeted entries")

    print(f"Existing unique prefixes: {len(existing_prefixes)}")
    print(f"Weak tags to generate: {len(WEAK_TAGS)}")
    print(f"Target: {args.calls} API calls × {args.batch_size} synopses = ~{args.calls * args.batch_size} samples")
    print()

    # Distribute calls across weak tags (more calls for worse tags)
    # First half of WEAK_TAGS (worst) get 2x calls
    calls_per_tag = {}
    priority_tags = WEAK_TAGS[:len(WEAK_TAGS) // 2]
    normal_tags = WEAK_TAGS[len(WEAK_TAGS) // 2:]

    total_weight = len(priority_tags) * 2 + len(normal_tags)
    base_calls = args.calls / total_weight

    for tag in priority_tags:
        calls_per_tag[tag] = int(base_calls * 2)
    for tag in normal_tags:
        calls_per_tag[tag] = int(base_calls)

    # Distribute remainder
    remaining = args.calls - sum(calls_per_tag.values())
    for i, tag in enumerate(WEAK_TAGS):
        if remaining <= 0:
            break
        calls_per_tag[tag] += 1
        remaining -= 1

    tag_counter = Counter()
    total_new = 0

    for tag, n_calls in calls_per_tag.items():
        print(f"\n{'='*60}")
        print(f"Tag: {tag} — {n_calls} calls planned")
        print(f"{'='*60}")

        for call_i in range(n_calls):
            # Pick random secondary tags to combine with
            other_tags = [t for t in TAXONOMY if t != tag]
            secondary = random.sample(other_tags, min(6, len(other_tags)))

            prompt = build_prompt(tag, secondary, args.batch_size)

            print(f"  [{call_i + 1}/{n_calls}] ", end="", flush=True)
            result = call_groq(prompt)

            if result and "entries" in result:
                new_count = 0
                for entry in result["entries"]:
                    synopsis = entry.get("synopsis", "").strip()
                    raw_tags = entry.get("tags", [])

                    # Normalize tags strictly
                    valid_tags = []
                    for t in raw_tags:
                        normalized = normalize_tag(t)
                        if normalized and normalized not in valid_tags:
                            valid_tags.append(normalized)

                    # Ensure focus tag is present
                    if tag not in valid_tags:
                        valid_tags.insert(0, tag)

                    if synopsis and len(synopsis) > 50 and valid_tags and synopsis[:50] not in existing_prefixes:
                        existing_prefixes.add(synopsis[:50])
                        all_entries.append({"synopsis": synopsis, "tags": valid_tags})
                        new_count += 1
                        for vt in valid_tags:
                            tag_counter[vt] += 1

                total_new += new_count
                print(f"+{new_count} (total: {len(all_entries)})", flush=True)
            else:
                print("failed", flush=True)

            time.sleep(args.delay)

            # Save periodically (every 10 calls)
            if (call_i + 1) % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Generated {total_new} new synopses ({len(all_entries)} total)")
    print(f"Saved to: {output_path}")
    print(f"\nTag distribution in new data:")
    for tag, count in tag_counter.most_common():
        print(f"  {tag}: {count}")

    print(f"\nNext steps:")
    print(f"  1. Run in Colab: !python generate_groq_targeted.py --calls 300")
    print(f"  2. Merge with existing data (the notebook will handle this)")
    print(f"  3. Retrain with: !python train.py --epochs 40 --batch-size 32 --lr 2e-5 --focal-loss --dynamic-weights 2.0")


if __name__ == "__main__":
    main()
""", "Description": "Script to generate targeted Groq synopses for weak tags with genre-specific prompting hints"

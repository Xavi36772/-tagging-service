"""Quick test of Groq API for synopsis generation."""
import httpx, json, sys, time, os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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

tags_csv = ", ".join(TAXONOMY)

prompt = (
    "Genera 3 sinopsis DIVERSAS de historias literarias en espa\u00f1ol.\n"
    "Cada sinopsis debe ser ORIGINAL, entre 30-80 palabras, como escrita por un humano real.\n"
    "Aseg\u00farate de incluir variedad de g\u00e9neros.\n"
    f"Tags disponibles: {tags_csv}\n"
    "RESPONDE EN JSON SOLAMENTE:\n"
    '{"entries": [\n'
    '    {"synopsis": "...", "tags": ["Tag1", "Tag2"]},\n'
    "    ...\n"
    "]}\n"
)

print("Calling Groq API...", flush=True)
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
    print(f"Status: {resp.status_code}", flush=True)
    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
        print(f"Response:\n{content[:500]}", flush=True)
        # Try to parse JSON
        try:
            data = json.loads(content)
            print(f"\nParsed {len(data.get('entries', []))} entries", flush=True)
        except:
            # Try to extract from markdown
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                print(f"\nExtracted {len(data.get('entries', []))} entries", flush=True)
    else:
        print(f"Body: {resp.text[:500]}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

print("Done", flush=True)

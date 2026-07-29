import httpx, json, os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

resp = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Eres un asistente que SOLO responde con JSON v\u00e1lido, nada m\u00e1s."},
            {"role": "user", "content": 'Genera 2 sinopsis. Responde SOLO JSON: {"entries": [{"synopsis": "texto", "tags": ["Tag1"]}]}'}
        ],
        "temperature": 0.5,
        "max_tokens": 1000,
    },
    timeout=30
)
print("Status:", resp.status_code)
content = resp.json()["choices"][0]["message"]["content"]
print("Raw response:")
print(repr(content[:500]))
print()
# Try to find JSON
start = content.find('{"entries"')
if start >= 0:
    print(f"Found 'entries' at position {start}")
    print(f"From there: {content[start:start+200]}")
else:
    print("No 'entries' found")
    # Try other patterns
    for p in ['"entries"', "'entries'", "entries"]:
        idx = content.find(p)
        if idx >= 0:
            print(f"Found '{p}' at {idx}: {content[max(0,idx-20):idx+200]}")

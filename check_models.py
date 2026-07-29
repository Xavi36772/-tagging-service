import httpx, os
key = os.environ.get("GROQ_API_KEY", "")
resp = httpx.get("https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=10)
print("Status:", resp.status_code)
data = resp.json()
for m in data.get("data", []):
    print(f"  {m['id']} - active: {m.get('active', '?')}")

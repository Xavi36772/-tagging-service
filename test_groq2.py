import sys, json
sys.path.insert(0, ".")
from generate_groq_natural import call_groq, extract_entries

# Test extraction
test_text = '{"entries": [{"synopsis": "La historia de un joven...", "tags": ["Drama"]}]}'
print("Test 1 - clean JSON:", extract_entries(test_text))

test_text2 = 'Texto antes\n{"entries": [{"synopsis": "Historia 1", "tags": ["Tag1"]}]}\nTexto despues'
print("Test 2 - wrapped:", extract_entries(test_text2))

test_text3 = "```json\n{\"entries\": [{\"synopsis\": \"Historia 2\", \"tags\": [\"Tag2\"]}]}\n```"
print("Test 3 - markdown:", extract_entries(test_text3))

test_text4 = '{"entries": [{"synopsis": "Historia 3", "tags": ["Tag3"]},]}'
print("Test 4 - trailing comma:", extract_entries(test_text4))

# Test API call
print("\n--- Testing Groq API ---")
result = call_groq("Genera 2 sinopsis en espanol. SOLO JSON: {\"entries\": [{\"synopsis\": \"...\", \"tags\": [\"Tag\"]}]}")
if result and "entries" in result:
    print(f"SUCCESS: {len(result['entries'])} entries")
    for e in result["entries"]:
        print(f'  - {e["synopsis"][:60]}... tags: {e["tags"]}')
else:
    print("FAILED:", result)

import json

with open("data/2026_q_values.json") as f:
    q = json.load(f)

states = ["30_170_-8_0", "30_140_-9_0"]

keys = list(q.keys())
print("First 5 keys:", keys[:5])
print()

# Search for the problem states
for k in keys:
    if "30_170_-8" in str(k) or "30_140_-9" in str(k):
        print(f"{k}: {q[k]}")

print("30_170_-8_0:", q["30_170_-8_0"])
print("30_140_-9_0:", q["30_140_-9_0"])

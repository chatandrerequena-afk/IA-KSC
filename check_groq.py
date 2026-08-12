
import os
from groq import Groq

MODEL = "qwen/qwen3.6-27b"

key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("Falta GROQ_API_KEY en las variables de entorno.")

client = Groq(api_key=key, timeout=30, max_retries=2)
models = client.models.list()
ids = {m.id for m in models.data}

print("Groq conectado.")
print("Modelo esperado:", MODEL)
print("Disponible:", MODEL in ids)

if MODEL not in ids:
    print("\nModelos visibles en tu cuenta:")
    for mid in sorted(ids):
        print("-", mid)

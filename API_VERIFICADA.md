# API verificada — 10 de agosto de 2026

## Groq

Documentación oficial consultada:
- https://console.groq.com/docs/vision
- https://console.groq.com/docs/model/qwen/qwen3.6-27b
- https://console.groq.com/docs/rate-limits
- https://github.com/groq/groq-python

Configuración utilizada:

```python
from groq import Groq

client = Groq(api_key=...)
client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,..."
                }
            }
        ]
    }],
    response_format={"type": "json_object"},
    reasoning_effort="none",
)
```

Groq documenta actualmente:
- `qwen/qwen3.6-27b` como modelo de texto + imagen.
- Chat Completions para visión.
- Base64 para imágenes locales.
- JSON Object Mode con imágenes.
- Free Plan sujeto a rate limits.
- El SDK oficial de Python tiene versión 1.6.0 publicada en julio de 2026.

## USDA FoodData Central

- https://fdc.nal.usda.gov/api-guide
- Acceso con API key.
- `DEMO_KEY` puede usarse para exploración con límites bajos.
- Datos de FoodData Central en dominio público/CC0.

## Open Food Facts

- https://openfoodfacts.github.io/openfoodfacts-server/api/
- API v3.6 es la versión actual recomendada para integraciones nuevas.
- Las operaciones READ no requieren autenticación, pero se debe usar User-Agent identificable.

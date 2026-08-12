# NutriVision MGP V3

Proyecto web en Python para EUREKA 2026.

## Qué hace

- Reconoce alimentos desde una foto usando Groq Vision.
- Usa el modelo `qwen/qwen3.6-27b`.
- La imagen se envía en Base64 siguiendo el formato oficial de Groq.
- Pide respuesta JSON con `response_format={"type":"json_object"}`.
- Busca información nutricional en USDA FoodData Central.
- Si usas `DEMO_KEY`, no necesitas una segunda clave para probar.
- Busca productos envasados por código de barras con Open Food Facts v3.6.
- Permite ingresar manualmente una etiqueta y mostrar advertencias peruanas.
- Guarda pruebas para medir precisión en Eureka.

## Importante sobre costos

El código no requiere un servicio de hosting de pago:
- Streamlit Community Cloud tiene despliegue gratuito.
- Groq tiene Free Plan sujeto a sus límites actuales.
- USDA FoodData Central es gratuito; `DEMO_KEY` tiene límites bajos.
- Open Food Facts permite lecturas de productos sin autenticación, con límites de uso.

Los proveedores pueden cambiar límites o planes en el futuro.

## 1. Requisitos

Usa Python 3.12 si puedes.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Colocar la clave de Groq

NO pegues la clave dentro de `app.py`.

Copia:

`.streamlit/secrets.toml.example`

como:

`.streamlit/secrets.toml`

y edita:

```toml
GROQ_API_KEY = "gsk_TU_CLAVE_REAL"
CONTACT_EMAIL = "tu_correo@example.com"
```

Ese archivo está excluido por `.gitignore`.

## 3. Ejecutar

```powershell
streamlit run app.py
```

## 4. Publicar gratis

1. Sube el proyecto a GitHub, sin `secrets.toml`.
2. Abre Streamlit Community Cloud.
3. Crea una app desde el repositorio.
4. En Advanced settings > Secrets coloca:

```toml
GROQ_API_KEY = "gsk_TU_CLAVE_REAL"
CONTACT_EMAIL = "tu_correo@example.com"
```

5. Despliega.

## USDA

Por defecto:

```python
USDA_API_KEY = "DEMO_KEY"
```

La clave DEMO oficial sirve para explorar y tiene límites menores.
Si después quieres una clave gratuita propia de data.gov, agrega:

```toml
USDA_API_KEY = "TU_CLAVE"
```

## Pruebas Eureka

Cada vez que analices una imagen:
1. confirma qué alimento había realmente;
2. marca si la identificación fue correcta;
3. registra el caso.

La app crea `validacion_eureka.csv`.

No dependas únicamente de ese archivo en Streamlit Cloud porque el almacenamiento
local de una app desplegada puede reiniciarse. Descarga el CSV periódicamente.

## Seguridad

- Nunca publiques `GROQ_API_KEY`.
- Nunca compartas la clave en screenshots o GitHub.
- Si una clave se filtra, revócala y crea otra.

## Alcance

La IA puede equivocarse.
Una fotografía no mide con exactitud gramos, aceite, ingredientes ni receta.
NutriVision es una herramienta educativa, no un sistema médico ni una dieta personalizada.

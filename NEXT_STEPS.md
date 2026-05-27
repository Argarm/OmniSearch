# DocuStream — Próximos Pasos

Guía para retomar el desarrollo desde donde lo dejamos. Sigue los pasos en orden.

---

## Prerequisitos

- [ ] Docker Desktop instalado y corriendo
- [ ] Python 3.11+ instalado
- [ ] Cuenta en GitHub (para el remote y los Secrets)
- [ ] API key de OpenAI (o URL de Ollama local)

---

## PASO 1 — Conectar el repositorio a GitHub

```bash
# Crear el repo en GitHub (sin inicializar — ya tenemos commits locales)
# En github.com: New repository → nombre: "docustream" → Create (sin README, sin .gitignore)

# Luego en la terminal:
git remote add origin https://github.com/<tu-usuario>/docustream.git
git branch -M main
git push -u origin main
```

---

## PASO 2 — Configurar las variables de entorno locales

```bash
cp .env.example .env
```

Editar `.env` con los valores reales:

| Variable | Dónde conseguirla |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API keys |
| `NOTION_TOKEN` | notion.so → Settings → Integrations → New integration |
| `NOTION_ROOT_PAGE_ID` | URL de la página raíz de Notion (el ID al final) |
| `CONFLUENCE_URL` | `https://<tu-empresa>.atlassian.net` |
| `CONFLUENCE_USER_EMAIL` | Tu email de Atlassian |
| `CONFLUENCE_TOKEN` | id.atlassian.com → Security → API tokens |
| `CONFLUENCE_SPACE_KEY` | Clave del espacio en Confluence (ej. `DOCS`) |

> Si solo quieres probar con PDFs locales, solo necesitas `OPENAI_API_KEY`. El resto es opcional.

---

## PASO 3 — Configurar GitHub Secrets (para el CI/CD automático)

En GitHub → tu repo → Settings → Secrets and variables → Actions → New repository secret.

Añadir un secret por cada variable de `.env` que tenga valor real:

```
OPENAI_API_KEY
NOTION_TOKEN
NOTION_ROOT_PAGE_ID
CONFLUENCE_URL
CONFLUENCE_USER_EMAIL
CONFLUENCE_TOKEN
CONFLUENCE_SPACE_KEY
```

> Sin esto el workflow `indexer.yml` no podrá conectarse a las fuentes externas.

---

## PASO 4 — Primera prueba local (sin Docker)

Ideal para verificar que todo funciona antes de dockerizar.

```bash
# Instalar dependencias
pip install -e ".[ingestion,backend,frontend,dev]"

# Levantar Qdrant local (solo este servicio via Docker)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest

# Poner un PDF de prueba
# → Copia cualquier PDF a: data/sources/

# Ejecutar el pipeline de ingesta
python -m ingestion.pipeline --source pdf

# Levantar el backend (en una terminal)
python backend/main.py

# Levantar el frontend (en otra terminal)
chainlit run frontend/app.py --port 8501

# Abrir en el navegador: http://localhost:8501
# Hacer una pregunta sobre el PDF que indexaste
```

---

## PASO 5 — Despliegue completo con Docker Compose

Una vez verificado que todo funciona en local:

```bash
# Stack completo (Qdrant + Backend + Frontend)
docker compose up --build

# Frontend disponible en: http://localhost:8501
```

Para desarrollo con hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
# Backend también expuesto en: http://localhost:8000/docs (Swagger UI)
# Qdrant dashboard en: http://localhost:6333/dashboard
```

Para indexar con Docker:

```bash
# Poner PDFs en data/sources/ y ejecutar:
docker compose --profile ingestion run ingestion

# Para indexar solo una fuente específica:
docker compose --profile ingestion run ingestion python -m ingestion.pipeline --source notion
```

---

## PASO 6 — Ejecutar los tests

```bash
# Tests unitarios (sin dependencias externas)
pytest tests/unit/ -v

# Tests de integración (requiere Qdrant corriendo)
QDRANT_URL=http://localhost:6333 pytest tests/integration/ -v

# Tests de estrés (requiere Qdrant + modelos + PDFs de fixture)
# 1. Copiar PDFs complejos a tests/stress/fixtures/
# 2. Asegurarse de que uno de ellos mencione un hecho conocido (ej. "$4.2M")
# 3. Actualizar el test_grounding_check con el nombre del PDF y el hecho
RUN_STRESS_TESTS=1 pytest tests/stress/ -v
```

---

## PASO 7 — Activar y verificar el CI/CD automático

1. Hacer un push a `main` → verifica que el workflow `ci.yml` pase en GitHub Actions
2. Ir a Actions → `DocuStream Indexer` → `Run workflow` → ejecutar manualmente con `source: pdf`
3. Confirmar que el workflow termina sin errores y los vectores están en Qdrant

Para que la indexación nightly funcione en producción, el workflow necesita apuntar a tu instancia de Qdrant persistente:

```yaml
# En .github/workflows/indexer.yml, reemplazar:
QDRANT_URL: http://localhost:6333   # ← servicio efímero de CI

# Por el secret que apunta a tu Qdrant en producción:
QDRANT_URL: ${{ secrets.QDRANT_URL }}
```

Y añadir el secret `QDRANT_URL` en GitHub apuntando a tu servidor.

---

## PASO 8 — Ajustes de calidad RAG (cuando tengas datos reales)

Una vez indexados documentos reales, ajustar estos parámetros en `config/settings.yaml`:

```yaml
chunking:
  chunk_size: 800       # Subir si los documentos tienen párrafos muy largos
  chunk_overlap: 150    # Subir si las respuestas pierden contexto entre chunks

retrieval:
  top_k: 6             # Subir a 8-10 si las respuestas son incompletas
  score_threshold: 0.35 # Bajar si hay pocas respuestas; subir si hay ruido

embedding:
  device: "cuda"        # Cambiar si tienes GPU disponible (10x más rápido)
```

Para cambiar el modelo de embeddings a uno más ligero (menor uso de RAM):
```yaml
embedding:
  model_name: "BAAI/bge-base-en-v1.5"   # 768-dim, mitad de RAM que bge-large
  vector_size: 768                        # También en qdrant.vector_size
```

> Si cambias el modelo de embeddings, hay que recrear la colección:
> `python scripts/reset_collection.py` y re-indexar todo.

---

## PASO 9 — Usar Ollama en lugar de OpenAI (opcional, sin costos)

Para usar un LLM local (Llama 3, Mistral, etc.):

```bash
# Instalar Ollama: ollama.com
ollama pull llama3.2

# En .env:
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=ollama   # Valor ficticio, Ollama no valida la key
```

No hay que cambiar nada en el código — el backend usa `OPENAI_BASE_URL` directamente.

---

## PASO 10 — Producción (despliegue en servidor)

Cuando el sistema esté validado localmente:

1. **Servidor**: una VM Linux con Docker instalado (mínimo 8GB RAM para el modelo de embeddings)
2. **Qdrant persistente**: el volumen `qdrant_data` del compose ya persiste los datos — montar en disco duradero
3. **Reverse proxy**: poner Nginx o Traefik delante del puerto 8501 con SSL
4. **Autenticación Chainlit**: configurar `CHAINLIT_AUTH_SECRET` en `.env` para proteger el acceso

```bash
# En el servidor, clonar el repo y arrancar:
git clone https://github.com/<tu-usuario>/docustream.git
cd docustream
cp .env.example .env && nano .env   # Configurar credenciales
docker compose up -d --build
```

---

## Referencia rápida de comandos

| Acción | Comando |
|---|---|
| Levantar stack completo | `docker compose up --build` |
| Solo Qdrant local | `docker run -d -p 6333:6333 qdrant/qdrant` |
| Indexar PDFs | `python -m ingestion.pipeline --source pdf` |
| Indexar todo | `python -m ingestion.pipeline --source all` |
| Resetear colección | `python scripts/reset_collection.py` |
| Tests unitarios | `pytest tests/unit/ -v` |
| Tests integración | `QDRANT_URL=http://localhost:6333 pytest tests/integration/ -v` |
| Ver logs backend | `docker compose logs -f backend` |
| Swagger UI | `http://localhost:8000/docs` (solo en dev) |
| Qdrant dashboard | `http://localhost:6333/dashboard` (solo en dev) |

# RAGForge

RAGForge is a backend service for turning PDF documents into something an LLM can actually search over. You upload a PDF, it gets stored, its text is extracted, split into chunks, and each chunk is embedded into a vector — all so that a later question like *"what's the vacation policy?"* can be matched against the chunks that actually talk about vacation, by meaning rather than by keyword. It's the retrieval half of a Retrieval-Augmented Generation (RAG) pipeline.

## What it does

```
PDF upload
   │
   ▼
Saved to disk + metadata row in Postgres
   │
   ▼
Text extracted page-by-page (pypdf)
   │
   ▼
Text split into overlapping chunks (custom recursive chunker)
   │
   ▼
Each chunk embedded into a 384-dim vector (sentence-transformers, local)
   │
   ▼
Chunk + vector stored in Postgres (pgvector)
   │
   ▼
POST /search embeds a query the same way and ranks chunks by cosine similarity
```

Nothing here calls out to an LLM yet — this project stops at "here are the most relevant chunks for your question." Generating an actual answer from those chunks (the "G" in RAG) is the natural next step, not something built yet.

## Tools & frameworks

| Tool | Role |
|---|---|
| **FastAPI** | The web framework — defines all HTTP routes (`/documents`, `/search`, `/jobs`) and handles request validation via Pydantic. |
| **Uvicorn** | ASGI server that actually runs the FastAPI app. |
| **PostgreSQL** | Primary datastore — document metadata, extracted text, and chunk rows all live here. |
| **pgvector** | Postgres extension that adds a native `vector` column type and similarity operators (`<=>` for cosine distance). Compiled from source for this project since there's no prebuilt Windows binary. |
| **SQLAlchemy 2.0** | ORM — all models (`Document`, `Document_Chunks`, `Jobs`) and queries, including the pgvector-aware `vector(384)` column, go through it. |
| **Alembic** | Database migrations — every schema change (new columns, new tables) is a versioned migration under `migrations/versions/`. |
| **Pydantic / pydantic-settings** | Request/response schema validation, and typed loading of config (`.env`) via `app/config.py`. |
| **pypdf** | Extracts text from uploaded PDFs, page by page. |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Turns text into 384-dimensional embedding vectors, entirely locally — no external API calls or per-request cost. |
| **pgvector-python** | The Python-side counterpart to the Postgres extension — gives SQLAlchemy a `Vector` column type and `.cosine_distance()` query operator. |
| **tiktoken** *(optional)* | Used for accurate token counts during chunking if installed; the chunker falls back to a character-based estimate if it isn't. |
| **pytest** | Test suite — 39 tests across document upload, ingestion/chunking, retrieval, and the search endpoint. |
| **psycopg** | The actual Postgres driver SQLAlchemy talks through. |

## Project structure

```
app/
├── main.py                  # FastAPI app, router registration
├── config.py                 # Settings loaded from .env
├── database.py                # SQLAlchemy engine/session setup
├── models/                    # ORM models: Document, Document_Chunks, Jobs
├── schemas/                   # Pydantic request/response models
├── routes/                    # HTTP endpoints (documents, jobs, search)
├── services/
│   ├── ingestion_service.py   # Orchestrates extract → chunk → embed → store
│   └── retrieval_service.py   # Embeds a query and runs the similarity search
├── loaders/
│   └── pdf_loader.py          # PDF → per-page text (pypdf)
├── chunkers/
│   └── recursive_chunker.py   # Text → overlapping token-bounded chunks
└── embeddings/
    └── embedding_service.py   # Text → vector (sentence-transformers)

migrations/                  # Alembic migration history
tests/                       # pytest suite
uploads/                     # Saved PDF files (gitignored)
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a PDF — saves the file, extracts text, chunks it, embeds each chunk, and stores everything. Returns the document with status `READY` or `FAILED`. |
| `GET` | `/documents` | List all documents. |
| `GET` | `/documents/{id}` | Get one document's metadata. |
| `DELETE` | `/documents/{id}` | Delete a document (cascades to its chunks). |
| `GET` | `/documents/{id}/text` | Debug endpoint — view the raw extracted page text. |
| `GET` | `/documents/{id}/chunks` | View the chunks generated for a document. |
| `POST` | `/search` | Semantic search — `{"query": "...", "top_k": 5, "document_id": "..."}` (`document_id` optional, scopes search to one document). Returns chunks ranked by similarity. |
| `GET`/`POST`/`DELETE` | `/jobs`, `/documents/{id}/jobs` | Basic job/status tracking records tied to a document. |

## Setup

Prerequisites: Python 3.12+, PostgreSQL running locally with a `ragforge` database, and the pgvector extension built for your Postgres version (see `migrations/` — the extension must be enabled before running migrations).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
POSTGRES_PASSWORD=your_postgres_password
```

Enable pgvector and run migrations:

```bash
psql -U postgres -d ragforge -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
```

Run the server:

```bash
uvicorn app.main:app --reload
```

## Testing

```bash
pytest
```

The suite is fully self-contained — document/job tests run against an in-memory SQLite database, and the ingestion/retrieval/search tests either exercise the real embedding model directly or mock the database layer, so `pytest` needs no live Postgres connection to pass.

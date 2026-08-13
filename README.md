# RAGForge

RAGForge is a full Retrieval-Augmented Generation (RAG) backend for PDF documents. You upload a PDF, it gets stored, its text is extracted, split into chunks, and each chunk is embedded into a vector — so that a later question like *"what's the vacation policy?"* can be matched against the chunks that actually talk about vacation, by meaning rather than by keyword. Those matched chunks are then fed to an LLM, which answers the question and cites exactly which chunk each part of its answer came from.

**This requires an LLM to actually be running and reachable** — RAGForge doesn't call out to a hosted API (OpenAI, Anthropic, etc.) for generation. It talks to [Ollama](https://ollama.com) over plain HTTP, either running on the same machine or on another PC on your network. Without an Ollama server reachable at the configured `OLLAMA_URL`, document upload/search still work, but `POST /generate` will fail. See Setup below.

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
   │
   ▼
POST /generate does the same retrieval, then sends the top chunks + question to
an LLM (Ollama, local or LAN) and returns an answer with citations back to the
exact chunks it used
```

## Tools & frameworks

| Tool | Role |
|---|---|
| **FastAPI** | The web framework — defines all HTTP routes (`/documents`, `/search`, `/generate`, `/jobs`) and handles request validation via Pydantic. |
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
| **[Ollama](https://ollama.com)** | Runs the LLM that actually generates answers in `POST /generate`. **Not a Python package** — a separate application you (or someone on your network) must have installed and running, with at least one model pulled. RAGForge talks to it over plain HTTP (`OLLAMA_URL`), not an SDK. |
| **httpx** | The HTTP client used to call Ollama's REST API from `generation_service.py`. |
| **pytest** | Test suite — 50 tests across document upload, ingestion/chunking, retrieval, search, and generation. |
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
│   ├── ingestion_service.py    # Orchestrates extract → chunk → embed → store
│   ├── retrieval_service.py    # Embeds a query and runs the similarity search
│   └── generation_service.py   # Retrieves chunks, prompts Ollama, returns a cited answer
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
| `POST` | `/generate` | Full RAG — same request shape as `/search`, but sends the retrieved chunks to Ollama and returns `{"answer": "...", "citations": [...]}`. **Requires Ollama to be reachable** — see Setup. |
| `GET`/`POST`/`DELETE` | `/jobs`, `/documents/{id}/jobs` | Basic job/status tracking records tied to a document. |

## Setup

Prerequisites: Python 3.12+, PostgreSQL running locally with a `ragforge` database, the pgvector extension built for your Postgres version (see `migrations/` — the extension must be enabled before running migrations), and **an Ollama server reachable somewhere on your network** (see below — required for `/generate`, not for upload/search).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
POSTGRES_PASSWORD=your_postgres_password
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.1
```

Enable pgvector and run migrations:

```bash
psql -U postgres -d ragforge -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
```

### Setting up Ollama (required for `/generate`)

RAGForge does not bundle or host an LLM — it makes plain HTTP calls to an Ollama server's `/api/generate` endpoint. You need one running somewhere reachable, with at least one model pulled:

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1
```

**If Ollama is on this same machine**, the default `.env` values above already work — no changes needed.

**If Ollama is on a different PC on your network**, two things need to happen:

1. On the Ollama machine, make it listen on the network instead of just `127.0.0.1` (set `OLLAMA_HOST=0.0.0.0` before running `ollama serve`), and allow inbound connections to port `11434` through its firewall.
2. In this project's `.env`, point `OLLAMA_URL` at that machine's LAN IP instead of `localhost`:
   ```
   OLLAMA_URL=http://<other-pc-ip>:11434/api/generate
   ```

Set `OLLAMA_MODEL` to whatever model is actually pulled on that Ollama instance — mismatched model names fail at request time, not at startup. You can check what's available with:

```bash
curl http://<ollama-host>:11434/api/tags
```

Document upload, extraction, chunking, and `/search` all work with no Ollama connection at all — only `POST /generate` needs it.

Run the server:

```bash
uvicorn app.main:app --reload
```

## Testing

```bash
pytest
```

The suite is fully self-contained — document/job tests run against an in-memory SQLite database, and the ingestion/retrieval/search/generation tests either exercise the real embedding model directly or mock the database layer and the Ollama HTTP call, so `pytest` needs no live Postgres connection and no running Ollama server to pass.

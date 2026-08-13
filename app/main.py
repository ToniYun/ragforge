import uuid

from fastapi import FastAPI

from app.routes.document import router as documents_router
from app.routes.job import router as jobs_router
from app.routes.search import router as search_router
from app.routes.generation import router as generation_router

app = FastAPI(title="RAGForge API")

@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(documents_router)
app.include_router(jobs_router)
app.include_router(search_router)
app.include_router(generation_router)
import uuid

from fastapi import FastAPI

from app.routes.document import router as documents_router

app = FastAPI(title="RAGForge API")

@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(documents_router) 
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from ai_document_intelligence.api.routes.documents import router as documents_router
from ai_document_intelligence.logging_config import setup_logging

import logging

from ai_document_intelligence.exceptions import DocumentNotFoundError

from contextlib import asynccontextmanager

from ai_document_intelligence.core.resources import create_vector_store

logger = logging.getLogger(__name__)

setup_logging()  # Initialize logging configuration
logger.info("AI Document Intelligence API initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated")
    app.state.application_name = "AI Document Intelligence"

    try:
        app.state.vector_store = create_vector_store()
    except Exception:
        logger.exception("Failed to initialize vector store")
        raise

    logger.info("Vector store initialized successfully")

    try:
        yield
    finally:
        logger.info("Application shutdown completed")

app = FastAPI(title="AI Document Intelligence API",
              lifespan=lifespan)

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_exception_handler(request: Request, exc: DocumentNotFoundError):
    logger.warning("Document not found: document_id=%s", exc.document_id)
    return JSONResponse(
        status_code=404,
        content={
                    "detail": str(exc)
                },
    )

app.include_router(documents_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/app-info")
def get_app_info(request: Request):
    return {
        "application_name": request.app.state.application_name
    }

@app.get("/ready")
def readiness_check(request: Request):
    vector_store = getattr(request.app.state, "vector_store", None)

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not initialized"
        )

    try:
        vector_store.get(limit=1)
    except Exception:
        logger.exception("Vector store readiness check failed")
        raise HTTPException(
            status_code=503,
            detail="Vector store is unavailable"
        )

    return {"status": "ready"}
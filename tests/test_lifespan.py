from unittest.mock import patch,Mock

import pytest
from fastapi.testclient import TestClient

from ai_document_intelligence.main import app


def test_startup_fails_when_vector_store_initialization_fails():
    with patch(
        "ai_document_intelligence.main.create_vector_store",
        side_effect=RuntimeError("ChromaDB initialization failed"),
    ):
        with pytest.raises(
            RuntimeError,
            match="ChromaDB initialization failed",
        ):
            with TestClient(app):
                pass

def test_application_startup_and_shutdown(caplog):
    mock_vector_store = Mock()

    with patch(
        "ai_document_intelligence.main.create_vector_store",
        return_value=mock_vector_store,
    ) as mock_create:

        with TestClient(app) as client:
            
            assert client.app.state.vector_store is mock_vector_store

            response = client.get("/health")

            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

        mock_create.assert_called_once()
        assert "Application shutdown completed" in caplog.text

def test_readiness_when_vector_store_initialized():
    mock_vector_store = Mock()

    with patch(
        "ai_document_intelligence.main.create_vector_store",
        return_value=mock_vector_store,
    ):
        with TestClient(app) as client:
            response = client.get("/ready")

            assert response.status_code == 200
            assert response.json() == {"status": "ready"}

def test_readiness_when_vector_store_is_not_initialized():
    with patch(
        "ai_document_intelligence.main.create_vector_store",
        return_value=None,
    ):
        with TestClient(app) as client:
            response = client.get("/ready")

            assert response.status_code == 503
            assert response.json() == {
                "detail": "Vector store is not initialized"
            }

def test_readiness_when_chromadb_fails():
    mock_vector_store = Mock()

    mock_vector_store.get.side_effect = RuntimeError(
        "ChromaDB connection failed"
    )

    with patch(
        "ai_document_intelligence.main.create_vector_store",
        return_value=mock_vector_store,
    ):
        with TestClient(app) as client:
            response = client.get("/ready")

            assert response.status_code == 503
            assert response.json() == {
                "detail": "Vector store is unavailable"
            }

    mock_vector_store.get.assert_called_once_with(limit=1)
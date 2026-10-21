from fastapi.testclient import TestClient
from ai_document_intelligence.main import app

from io import BytesIO
from pypdf import PdfWriter

from unittest.mock import patch, Mock

from pathlib import Path

import logging
import pytest

from ai_document_intelligence.services.document_service import ingest_document

mock_vector_store = Mock()
app.state.vector_store = mock_vector_store

client = TestClient(app)


def test_upload_invalid_file_type():
    response = client.post(
        "/documents",
        files={
            "file": (
                "sample.txt",
                b"This is not a PDF.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or corrupted PDF file."

def test_upload_file_too_large():
    large_file = b"%PDF-1.4\n" + b"A" * (11 * 1024 * 1024)

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.pdf",
                large_file,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File size exceeds the 10 MiB limit."


def test_upload_corrupted_pdf():
    response = client.post(
        "/documents",
        files={
            "file": (
                "corrupted.pdf",
                b"This is not a real PDF document.",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or corrupted PDF file."

def test_upload_empty_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    response = client.post(
        "/documents",
        files={
            "file": (
                "empty.pdf",
                pdf_buffer.getvalue(),
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded PDF contains no text."

def test_upload_valid_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    with patch(
        "ai_document_intelligence.api.routes.documents.ingest_document",
        return_value="abc-123"
    ) as mock_ingest:

        response = client.post(
            "/documents",
            files={
                "file": (
                    "sample.pdf",
                    pdf_buffer.getvalue(),
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == "abc-123"
    mock_ingest.assert_called_once()
    args, kwargs = mock_ingest.call_args

    assert len(args) == 2
    assert isinstance(args[0], Path)
    assert args[0].suffix == ".pdf"
    assert args[1] is mock_vector_store
    assert kwargs == {}

def test_ask_question_document_not_found():
    document_id = "non-existent-123"

    with patch(
        "ai_document_intelligence.api.routes.documents.document_exists",
        return_value=False
    ) as mock_exists:

        response = client.post(
            f"/documents/{document_id}/ask",
            json={
                "question": "What is Docker?"
            }
        )

    assert response.status_code == 404
    assert response.json()["detail"] == f"Document not found: {document_id}"

    mock_exists.assert_called_once_with(document_id,mock_vector_store)

def test_ask_question_no_results():
    document_id = "test-doc-123"

    with patch(
        "ai_document_intelligence.api.routes.documents.document_exists",
        return_value=True
    ), patch(
        "ai_document_intelligence.api.routes.documents.search_documents",
        return_value=[]
    ) as mock_search, patch(
        "ai_document_intelligence.api.routes.documents.generate_answer"
    ) as mock_llm:

        response = client.post(
            f"/documents/{document_id}/ask",
            json={"question": "What is Kubernetes?"}
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't know."
    assert response.json()["retrieved_chunks"] == 0
    assert response.json()["sources"] == []

    mock_search.assert_called_once_with(
        document_id=document_id,
        query="What is Kubernetes?",
        top_k=3,
        vector_store=mock_vector_store
    )
    mock_llm.assert_not_called()


def test_ask_question_success():
    document_id = "test-doc-456"
    question = "What is Docker?"
    mock_results = [
        {
            "content": "Docker packages applications into containers.",
            "metadata": {"page_label": "1"},
            "score": 0.25
        }
    ]

    with patch(
        "ai_document_intelligence.api.routes.documents.document_exists",
        return_value=True
    ), patch(
        "ai_document_intelligence.api.routes.documents.search_documents",
        return_value=mock_results
    ) as mock_search, patch(
        "ai_document_intelligence.api.routes.documents.generate_answer",
        return_value="Docker is a platform for containerizing applications."
    ) as mock_llm:

        response = client.post(
            f"/documents/{document_id}/ask",
            json={"question": question}
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Docker is a platform for containerizing applications."
    assert response.json()["retrieved_chunks"] == 1
    mock_search.assert_called_once_with(
        document_id=document_id,
        query=question,
        top_k=3,
        vector_store=mock_vector_store
    )
    mock_llm.assert_called_once()
    mock_llm.assert_called_once_with(
        question=question,
        search_results=mock_results,
    )
    assert len(response.json()["sources"]) == 1

    source = response.json()["sources"][0]

    assert source["page"] == "1"
    assert source["text"] == "Docker packages applications into containers."
    assert source["score"] == 0.25

def test_ask_question_missing_question():
    document_id = "test-doc-789"

    with patch(
        "ai_document_intelligence.api.routes.documents.document_exists",
        return_value=True
    ):

        response = client.post(
            f"/documents/{document_id}/ask",
            json={}
        )

    assert response.status_code == 422
    # assert "question" in response.json()["detail"][0]["loc"]

def test_document_not_found_exception_handler():
    from ai_document_intelligence.exceptions import DocumentNotFoundError
    from ai_document_intelligence.main import app
    from fastapi.testclient import TestClient

    @app.get("/test-document-not-found")
    def raise_document_not_found():
        raise DocumentNotFoundError("test-123")

    client = TestClient(app)
    response = client.get("/test-document-not-found")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found: test-123"



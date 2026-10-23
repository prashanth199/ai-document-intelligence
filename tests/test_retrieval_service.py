from unittest.mock import MagicMock

from ai_document_intelligence.services.retrieval_service import search_documents
from langchain_core.documents import Document
import pytest


def test_search_documents_passes_correct_parameters():
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = []

    results = search_documents(
        document_id="doc-123",
        query="What is Spring Boot?",
        vector_store=mock_vector_store,
        top_k=3
    )

    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        query="What is Spring Boot?",
        k=3,
        filter={"document_id": "doc-123"}
    )

    assert results == []




def test_search_documents_filters_irrelevant_chunks():
    mock_vector_store = MagicMock()

    relevant_doc = Document(
        page_content="Spring Boot supports auto-configuration.",
        metadata={"document_id": "doc-123", "page_label": "1"}
    )

    irrelevant_doc = Document(
        page_content="Unrelated information.",
        metadata={"document_id": "doc-123", "page_label": "2"}
    )

    mock_vector_store.similarity_search_with_score.return_value = [
        (relevant_doc, 0.5),
        (irrelevant_doc, 1.8)
    ]

    results = search_documents(
        document_id="doc-123",
        query="What is Spring Boot?",
        vector_store=mock_vector_store
    )

    assert len(results) == 1
    assert results[0]["content"] == "Spring Boot supports auto-configuration."
    assert results[0]["score"] == 0.5

def test_search_documents_threshold_boundary():
    mock_vector_store = MagicMock()

    scores = [1.49, 1.50, 1.51]

    mock_vector_store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content=f"Chunk with score {score}",
                metadata={"document_id": "doc-123"}
            ),
            score
        )
        for score in scores
    ]

    results = search_documents(
        document_id="doc-123",
        query="Spring Boot",
        vector_store=mock_vector_store
    )

    assert len(results) == 2
    assert [result["score"] for result in results] == [1.49, 1.50]





def test_search_documents_propagates_database_error():
    mock_vector_store = MagicMock()

    mock_vector_store.similarity_search_with_score.side_effect = RuntimeError(
        "Vector database unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Vector database unavailable"
    ):
        search_documents(
            document_id="doc-123",
            query="What is Spring Boot?",
            vector_store=mock_vector_store
        )
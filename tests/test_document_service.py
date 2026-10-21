from unittest.mock import Mock
import pytest
import logging 

from ai_document_intelligence.services.document_service import ingest_document

def test_ingest_document_chromadb_failure(caplog):
    pdf_path = "sample_documents/sample.pdf"

    mock_vector_store = Mock()

    mock_vector_store.add_documents.side_effect = RuntimeError(
        "ChromaDB connection failed"
    )

    with pytest.raises(RuntimeError, match="ChromaDB connection failed"):
        ingest_document(pdf_path, mock_vector_store)

    mock_vector_store.add_documents.assert_called_once()

    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]

    assert len(error_records) == 1
    assert error_records[0].exc_info is not None
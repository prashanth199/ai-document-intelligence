from pathlib import Path

from ai_document_intelligence.services.document_service import ingest_document

document_path = Path("sample_documents/spring_boot.pdf")
document_id = ingest_document(str(document_path))

print("Document ID:", document_id)
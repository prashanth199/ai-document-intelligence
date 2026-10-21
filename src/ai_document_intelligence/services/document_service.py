from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_chroma import Chroma

import logging

logger = logging.getLogger(__name__)

def ingest_document(pdf_path: str | Path, vector_store: Chroma) -> str:
	"""Load and process a PDF and return its unique document ID."""
	logger.info("Document ingestion started")
	path = Path(pdf_path)
	document_id = str(uuid4())
	reader = PdfReader(str(path))

	pages = [
		Document(
			page_content=page.extract_text() or "",
			metadata={
				"document_id": document_id,
				"source": str(path),
				"page": page_number,
				"page_label": str(page_number + 1),
				"total_pages": len(reader.pages),
			},
		)
		for page_number, page in enumerate(reader.pages)
	]

	splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
	split_chunks = splitter.split_documents(pages)

	if not split_chunks:
		logger.error(
        	"Document ingestion failed: No text found in PDF"
    	)
		raise ValueError("No text found in the PDF.")

	try:
		vector_store.add_documents(split_chunks)
	except Exception:
		logger.exception("Document ingestion failed: document_id=%s", document_id)
		raise
	logger.info(
		"Document ingestion completed: document_id=%s, chunks=%d",
		document_id,
		len(split_chunks)
	)

	return document_id

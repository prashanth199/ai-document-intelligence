from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

pdf_path = Path("sample_documents/spring_boot.pdf")

pdf_reader = PdfReader(pdf_path)

documents = []

document_id = str(uuid4())

for page_number,page in enumerate(pdf_reader.pages):
    
    text = page.extract_text() or ""
    document = Document(
        page_content=text,
        metadata={
            "source": str(pdf_path),
            "page": page_number,
            "page_label": str(page_number + 1),
            "total_pages": len(pdf_reader.pages),
            "document_id": document_id,
        }
    )

    documents.append(document)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)

split_chunks = splitter.split_documents(documents)

for index, chunk in enumerate(split_chunks):
    print(f"\n--- CHUNK {index + 1} ---")
    print("Page:", chunk.metadata["page_label"])
    print("Length:", len(chunk.page_content))
    print("Content:", chunk.page_content)
    print("Document ID:", chunk.metadata["document_id"])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="multi_document_test",    
    embedding_function=embeddings,
    persist_directory="./chroma_multi_db",
)

vector_store.add_documents(split_chunks)

question = "What is dependency injection in Spring Boot?"

results = vector_store.similarity_search(
    query=question,
    k=3,
    filter={
        "document_id": document_id
    }
)

for index, document in enumerate(results, start=1):
    print(f"\n--- RESULT {index} ---")
    print("Content:", document.page_content)
    print("Page:", document.metadata.get("page_label"))
    print("Document ID:", document.metadata.get("document_id"))

print("\nDocument ID:", document_id)
print("Chunks stored:", len(split_chunks))
print("Total chunks in Chroma:", vector_store._collection.count())
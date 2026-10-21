from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma


load_dotenv()

pdf_path = Path("sample_documents/sample.pdf")

reader = PdfReader(pdf_path)

documents = []

for page_number, page in enumerate(reader.pages):
    text = page.extract_text() or ""

    document = Document(
        page_content=text,
        metadata={
            "source": str(pdf_path),
            "page": page_number,
            "page_label": str(page_number + 1),
            "total_pages": len(reader.pages),
        }
    )

    documents.append(document)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(documents)

print("Documents:", len(documents))
print("Chunks:", len(splits))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="document_intelligence",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

print("Collection:", vector_store._collection.name)
print("Stored chunks:", vector_store._collection.count())

ids = vector_store.add_documents(documents=splits)
print("Inserted IDs:", ids)
print("Stored chunks:", vector_store._collection.count())
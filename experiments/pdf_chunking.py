from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = Path("sample_documents/sample.pdf")

reader = PdfReader(pdf_path)

documents = []

for page_number, page in enumerate(reader.pages):
    text = page.extract_text()

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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Original documents:", len(documents))
print("Number of chunks:", len(chunks))

print("\nTYPE:")
print(type(chunks[0]))

print("\nFIRST CHUNK:")
print(chunks[0].page_content)

print("\nMETADATA:")
print(chunks[0].metadata)

for index, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {index + 1} ---")
    print("Page:", chunk.metadata["page_label"])
    print("Length:", len(chunk.page_content))
    print(chunk.page_content)
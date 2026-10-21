from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


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


print(type(documents))
print("Number of documents:", len(documents))

print("\nFIRST DOCUMENT:")
print(documents[0])

print("\nCONTENT:")
print(documents[0].page_content)

print("\nMETADATA:")
print(documents[0].metadata)
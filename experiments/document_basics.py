from langchain_core.documents import Document

document = Document(
    page_content="""
    Docker is a containerization platform that allows developers
    to package applications together with their dependencies.
    """,
    metadata={
        "source": "docker_guide.pdf",
        "page": 1,
        "author": "Prashanth"
    }
)

print(document)
print(type(document))

print("\nCONTENT:")
print(document.page_content)

print("\nMETADATA:")
print(document.metadata)
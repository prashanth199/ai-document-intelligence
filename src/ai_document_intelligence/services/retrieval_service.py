

from langchain_chroma import Chroma

def search_documents(document_id: str, query: str, vector_store: Chroma, top_k: int = 3) -> list[dict]:
    """Search for documents in the vector store based on a query."""


    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter={
            "document_id": document_id
        }
    )
    return [
        {
            "content": result.page_content,
            "metadata": result.metadata,
            "score": score,
        }
        for result,score in results if score<=1.5
    ]

def document_exists(document_id: str,vector_store: Chroma) -> bool:
    """Check if a document exists in the vector store."""
    results = vector_store.get(
        where={"document_id": document_id},
        limit=1
    )
    return bool(results["ids"])
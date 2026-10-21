from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ai_document_intelligence.config import get_settings


def create_vector_store() -> Chroma:
    settings = get_settings()

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model
    )

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_directory,
    )
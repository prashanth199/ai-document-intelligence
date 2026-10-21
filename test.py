from ai_document_intelligence.config import Settings

settings = Settings()

print("Embedding model:", settings.embedding_model)
print("Chroma collection:", settings.chroma_collection)
print("Chroma directory:", settings.chroma_persist_directory)
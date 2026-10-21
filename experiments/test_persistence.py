from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="document_intelligence",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

print("Stored chunks:", vector_store._collection.count())

question = "What are the benefits of Docker?"

results = vector_store.similarity_search_with_score(
    query=question,
    k=3
)

for index, (document, score) in enumerate(results, start=1):
    print(f"\n--- RESULT {index} ---")
    print("Content:", document.page_content)
    print("Page:", document.metadata.get("page_label"))
    print("Source:", document.metadata.get("source"))
    print("Score:", score)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


document_id = "ea63ac49-30fd-4692-977d-08c234ee73c1"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="multi_document_test",
    embedding_function=embeddings,
    persist_directory="./chroma_multi_db",
)

question = "What are the benefits of Docker?"

results = vector_store.similarity_search_with_score(
    query=question,
    k=3,
    filter={
        "document_id": document_id
    }
)

for index, (document, score) in enumerate(results, start=1):
    print(f"\n--- RESULT {index} ---")
    print("Content:", document.page_content)
    print("Page:", document.metadata.get("page_label"))
    print("Document ID:", document.metadata.get("document_id"))
    print("Score:", score)

print("\nTotal chunks in Chroma:", vector_store._collection.count())
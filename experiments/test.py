from ai_document_intelligence.core.resources import create_vector_store

vector_store = create_vector_store()

# results = vector_store.get(limit=5)

# for metadata in results["metadatas"]:
#     print(metadata.get("document_id"))

# document_id = "974bbefb-d5a3-4d78-8eab-39972e213e6b"

# results = vector_store.get(
#     where={"document_id": document_id},
#     limit=5
# )

# for chunk_id, text in zip(results["ids"], results["documents"]):
#     print(f"{chunk_id}: {text}")

results = vector_store._collection.query(
    query_texts=["How does Spring Boot simplify application development?"],
    n_results=3,
    where={
        "document_id": "974bbefb-d5a3-4d78-8eab-39972e213e6b"
    }
)

print(results["ids"])
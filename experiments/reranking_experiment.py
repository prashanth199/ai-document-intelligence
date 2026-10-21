from sentence_transformers import CrossEncoder

from ai_document_intelligence.core.resources import create_vector_store


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")

    top_k_retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    number_of_relevant_items_in_top_k = len(top_k_retrieved_ids & relevant_ids)
    total_number_of_relevant_items = len(relevant_ids)

    return number_of_relevant_items_in_top_k / total_number_of_relevant_items

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",device="cpu")

print("Cross-encoder model loaded successfully!")

question = "What are the main advantages of using Spring Boot?"

chunks = [
    "Spring Boot simplifies application development through auto-configuration and reduced boilerplate code.",
    "Spring Boot applications use SpringApplication.run() and an embedded Tomcat server.",
    "Spring Boot accelerates application setup and integrates with the Spring ecosystem."
]

pairs = []

for chunk in chunks:
    pairs.append([question, chunk])

scores = model.predict(pairs)

for chunk, score in zip(chunks, scores):
    print(f"Score: {float(score):.4f}")
    print(f"Chunk: {chunk}")
    print()

ranked_chunks = sorted(
    zip(chunks, scores),
    key=lambda item: item[1],
    reverse=True
)

for chunk, score in ranked_chunks:
    print("___Sorted and Reranked Chunks___")
    print(f"Score: {float(score):.4f}")
    print(f"Chunk: {chunk}")
    print()

vector_store = create_vector_store()

question = "What are the main advantages of using Spring Boot?"

question_embedding = vector_store.embeddings.embed_query(question)

query_results = vector_store._collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    where={
        "document_id": "974bbefb-d5a3-4d78-8eab-39972e213e6b"
    }
)

retrieved_ids = query_results["ids"][0]
retrieved_documents = query_results["documents"][0]

pairs = []

for document in retrieved_documents:
    pairs.append([question, document])

scores = model.predict(pairs)

ranked_results = zip(retrieved_ids, retrieved_documents, scores)

highly_ranked_chunks = sorted(ranked_results,key=lambda item: item[2],reverse=True)

reranked_ids = [item[0] for item in highly_ranked_chunks]

relevant_ids = {
    "adf1f8ee-090e-402a-b6dd-017e173f9c4b",
    "8be434a2-f766-4948-8bd1-6c0d2e630836"
}

for k in [1, 2, 3]:
    recall = recall_at_k(reranked_ids, relevant_ids, k)
    print(f"Reranked Recall@{k}: {recall}")

for chunk_id,document,score in highly_ranked_chunks:
    print(f"Chunk ID: {chunk_id}")
    print(f"Score: {score}")
    print(f"Document Content: {document}")

# for chunk_id, content in zip(query_results["ids"][0], query_results["documents"][0]):
#     print(f"Chunk ID: {chunk_id}")
#     print(f"Content: {content}\n")


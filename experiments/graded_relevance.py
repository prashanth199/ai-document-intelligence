from math import log2

from sentence_transformers import CrossEncoder
from ai_document_intelligence.core.resources import create_vector_store

relevance_grades = {
    "8c409c30-1267-4453-bca4-cd35e08dfd26": 1,
    "8be434a2-f766-4948-8bd1-6c0d2e630836": 2,
    "adf1f8ee-090e-402a-b6dd-017e173f9c4b": 2
}

for chunk_id, grade in relevance_grades.items():
    print(f"Chunk ID: {chunk_id}, Grade: {grade}")

ranked_grades = [1,2,2]

# DCG@k = sum((2^rel_i - 1) / log2(i + 1)) for i = 1..k
# rel_i corresponds to the graded relevance at rank i.
dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ranked_grades, start=1))
print(f"DCG: {dcg}")

ideal_grades = sorted(ranked_grades, reverse=True)
idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
ndcg = dcg / idcg if idcg != 0 else 0
print(f"NDCG: {ndcg:.4f}")
print(f"Reranker NDCG: {ndcg:.4f}")

chroma_grades = [2, 1, 2]
chroma_dcg = sum(
    (2**grade - 1) / log2(rank + 1)
    for rank, grade in enumerate(chroma_grades, start=1)
)
chroma_ideal_grades = [2, 2, 1]
chroma_idcg = sum(
    (2**grade - 1) / log2(rank + 1)
    for rank, grade in enumerate(chroma_ideal_grades, start=1)
)
chroma_ndcg = chroma_dcg / chroma_idcg if chroma_idcg != 0 else 0
print(f"ChromaDB NDCG: {chroma_ndcg:.4f}")

def ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    if k <= 0 or not ranked_grades:
        return 0.0

    top_k = ranked_grades[:k]
    dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(top_k, start=1))

    ideal_grades = sorted(ranked_grades, reverse=True)[:k]
    idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))

    return dcg / idcg if idcg != 0 else 0.0



print(f"ChromaDB NDCG@3: {ndcg_at_k([2, 1, 2], 3):.4f}")
print(f"Reranker NDCG@3: {ndcg_at_k([1, 2, 2], 3):.4f}")

evaluation_grades = {
    "How does Spring Boot simplify application development?": [2, 1, 0],
    "What are the main advantages of using Spring Boot?": [2, 1, 2]
}

ndcg_scores = []
for question, grades in evaluation_grades.items():
    score = ndcg_at_k(grades, k=3)
    ndcg_scores.append(score)
    print(f"Question: {question}, NDCG@3: {score:.4f}")

mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
print(f"Mean NDCG@3: {mean_ndcg:.4f}")

question_relevance_grades={
    "What are the main advantages of using Spring Boot?": {
        "8c409c30-1267-4453-bca4-cd35e08dfd26": 1,
        "8be434a2-f766-4948-8bd1-6c0d2e630836": 2,
        "adf1f8ee-090e-402a-b6dd-017e173f9c4b": 2
    }
}

for question, chunk_grades in question_relevance_grades.items():
    print(f"Question: {question}")
    for chunk_id, grade in chunk_grades.items():
        print(f"  Chunk ID: {chunk_id}, Grade: {grade}")

retrieved_ids = [
    "8c409c30-1267-4453-bca4-cd35e08dfd26",
    "8be434a2-f766-4948-8bd1-6c0d2e630836",
    "adf1f8ee-090e-402a-b6dd-017e173f9c4b",
]
spring_boot_grades = question_relevance_grades[
    "What are the main advantages of using Spring Boot?"
]

# chroma_retrieved_ids = [...]
# reranked_retrieved_ids = [...]

retrieved_grades = []
for chunk_id in retrieved_ids:
    retrieved_grades.append(spring_boot_grades.get(chunk_id, 0))

print(retrieved_grades)

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

chroma_retrieved_ids = query_results["ids"][0]
chroma_grades = [spring_boot_grades.get(chunk_id, 0) for chunk_id in chroma_retrieved_ids]

print("ChromaDB retrieved IDs:", chroma_retrieved_ids)
print("ChromaDB grades:", chroma_grades)


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2",device="cpu")

retrieved_documents = query_results["documents"][0]

pairs = []

for document in retrieved_documents:
    pairs.append([question, document])

scores = model.predict(pairs)

ranked_results = zip(
    chroma_retrieved_ids,
    retrieved_documents,
    scores
)

highly_ranked_chunks = sorted(
    ranked_results,
    key=lambda item: item[2],
    reverse=True
)

reranked_ids = [item[0] for item in highly_ranked_chunks]

print("Reranked IDs:", reranked_ids)

reranked_grades = [
    spring_boot_grades.get(chunk_id, 0)
    for chunk_id in reranked_ids
]

print("Reranker grades:", reranked_grades)

chroma_ndcg = ndcg_at_k(chroma_grades, 3)
reranker_ndcg = ndcg_at_k(reranked_grades, 3)

print(f"Chroma NDCG@3:{chroma_ndcg}")
print(f"Reranker NDCG@3:{reranker_ndcg}")


def rerank_documents(question, retrieved_ids, retrieved_documents, model):

    for document in retrieved_documents:
        pairs.append([question,document])

    scores = model.predict(pairs)

    ranked_results = zip(retrieved_ids,retrieved_documents,scores)

    sorted_ranked_results = sorted(ranked_results,key=lambda item: item[2],reverse=True)

    reranked_ids = [item[0] for item in sorted_ranked_results]

    return reranked_ids

result = rerank_documents(
    question,
    chroma_retrieved_ids,
    retrieved_documents,
    model
)

print("Function reranked IDs:", result)
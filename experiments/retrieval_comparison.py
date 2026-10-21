from math import log2

from ai_document_intelligence.core.resources import create_vector_store
from sentence_transformers import CrossEncoder

def rerank_documents(question, retrieved_ids, retrieved_documents, model):
    pairs = []
    if not retrieved_documents:
        return []
    for document in retrieved_documents:
        pairs.append([question,document])

    scores = model.predict(pairs)

    ranked_results = zip(retrieved_ids,retrieved_documents,scores)

    sorted_ranked_results = sorted(ranked_results,key=lambda item: item[2],reverse=True)

    reranked_ids = [item[0] for item in sorted_ranked_results]

    return reranked_ids

def ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    if k <= 0 or not ranked_grades:
        return 0.0

    top_k = ranked_grades[:k]
    dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(top_k, start=1))

    ideal_grades = sorted(ranked_grades, reverse=True)[:k]
    idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))

    return dcg / idcg if idcg != 0 else 0.0

evaluation_dataset = {
    "What are the main advantages of using Spring Boot?": {
        "document_id": "974bbefb-d5a3-4d78-8eab-39972e213e6b",
        "relevance_grades": {
            "8c409c30-1267-4453-bca4-cd35e08dfd26": 1,
            "8be434a2-f766-4948-8bd1-6c0d2e630836": 2,
            "adf1f8ee-090e-402a-b6dd-017e173f9c4b": 2
        }
    }
}

vector_store = create_vector_store()

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu")

for question, evaluation in evaluation_dataset.items():
    document_id = evaluation["document_id"]
    relevance_grades = evaluation["relevance_grades"]

    question_embedding = vector_store.embeddings.embed_query(question)

    query_results = vector_store._collection.query(
        query_embeddings= [question_embedding],
        n_results=3,
        where={
            "document_id": document_id
        }
    )
    retrieved_ids = query_results["ids"][0]
    retrieved_documents = query_results["documents"][0]

    reranked_ids = rerank_documents(
        question,
        retrieved_ids,
        retrieved_documents,
        model
    )

    chroma_grades = [
        relevance_grades.get(chunk_id, 0) for chunk_id in retrieved_ids
    ]

    reranked_grades = [
        relevance_grades.get(chunk_id,0) for chunk_id in reranked_ids
    ]

    chromadb_ndcg = ndcg_at_k(chroma_grades,3)
    reranked_ndcg = ndcg_at_k(reranked_grades,3)


    print(f"Question: {question}")
    print(f"Retrieved ID's: {retrieved_ids}")
    print(f"Re-Ranked ID's: {reranked_ids}")
    print(f"Chroma NDCG@3: {chromadb_ndcg:.4f}")
    print(f"Re-Ranked NDCG@3: {reranked_ndcg:.4f}")

from ai_document_intelligence.core.resources import create_vector_store

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    # Write your logic here
    if k < 0:
        raise ValueError("k must be non-negative")

    top_k_retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 0.0

    number_of_relevant_items_in_top_k = len(top_k_retrieved_ids & relevant_ids)
    total_number_of_relevant_items = len(relevant_ids)

    return number_of_relevant_items_in_top_k / total_number_of_relevant_items


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    
    if k < 0:
        raise ValueError("k must be non-negative")

    if k == 0:
        return 0.0

    top_k_retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)

    number_of_relevant_items_in_top_k = len(top_k_retrieved_ids & relevant_ids)
    return number_of_relevant_items_in_top_k / k


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")

    if k == 0:
        return 0.0

    top_k_retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    return 1.0 if top_k_retrieved_ids & relevant_ids else 0.0

evaluation_dataset = [
    {
        "question": "What is Docker Compose?",
        "relevant_ids": {"chunk_2"}
    },
    {
        "question": "What are Docker images?",
        "relevant_ids": {"chunk_3"}
    }
]

retrieval_results = {
    "What is Docker Compose?": ["chunk_1", "chunk_2", "chunk_3"],
    "What are Docker images?": ["chunk_1", "chunk_2"]
}

for item in evaluation_dataset:
    print(item["question"], item["relevant_ids"])

question1 = evaluation_dataset[0]
recall1 = recall_at_k(retrieved_ids=["chunk_1", "chunk_2", "chunk_3"], 
                      relevant_ids=question1["relevant_ids"], k=3)
precision1 = precision_at_k(retrieved_ids=["chunk_1", "chunk_2", "chunk_3"], 
                           relevant_ids=question1["relevant_ids"], k=3)

print(f"Recall@3 for question '{question1['question']}': {recall1}")
print(f"Precision@3 for question '{question1['question']}': {precision1}")


recall_scores = []
for item in evaluation_dataset:
    question = item["question"]
    retrieved_ids = retrieval_results[question]
    recall = recall_at_k(retrieved_ids, item["relevant_ids"], k=2)
    recall_scores.append(recall)
    print(f"Recall@2 for question '{question}': {recall}")

mean_recall_at_2 = sum(recall_scores) / len(recall_scores)
print(f"Mean Recall@2: {mean_recall_at_2}")


vector_store = create_vector_store()

spring_boot_evaluation_dataset= [
    {
        "question": "How does Spring Boot simplify application development?",
        "relevant_ids": {"adf1f8ee-090e-402a-b6dd-017e173f9c4b"}
    },
    {
        "question": "Which Spring MVC annotations are used to create REST endpoints?",
        "relevant_ids": {"446670a5-e1f5-46e3-a2f6-b147af65bf62"}
    },
    {
        "question": "How does Spring Boot start an application and run it without an external application server?",
        "relevant_ids": {"8c409c30-1267-4453-bca4-cd35e08dfd26"}
    },
    {
        "question": "What are the main advantages of using Spring Boot?",
        "relevant_ids": {
            "adf1f8ee-090e-402a-b6dd-017e173f9c4b",
            "8be434a2-f766-4948-8bd1-6c0d2e630836"
        }
    }
]

springboot_recall_scores = []

mean_recall_at_1 = []
mean_recall_at_2 = []
for item in spring_boot_evaluation_dataset:
    question = item["question"]
    relevant_ids = item["relevant_ids"]
    question_embedding = vector_store.embeddings.embed_query(question)
    query_results = vector_store._collection.query(
        # query_texts=[question],
        query_embeddings=[question_embedding],
        n_results=3,
        where={
            "document_id": "974bbefb-d5a3-4d78-8eab-39972e213e6b"
        }
    )
    retrieved_ids = query_results["ids"][0]

    for k in [1,2,3]:
        recall = recall_at_k(retrieved_ids, relevant_ids, k=k)
        if k==1:
            mean_recall_at_1.append(recall)
        if k==2:
            mean_recall_at_2.append(recall)
        print(f"Recall@{k}: {recall_at_k(retrieved_ids, relevant_ids, k)}")

    recall = recall_at_k(retrieved_ids, relevant_ids, k=3)
    precision = precision_at_k(retrieved_ids, relevant_ids, k=3)
    hit = hit_at_k(retrieved_ids, relevant_ids, k=3)
    springboot_recall_scores.append(recall)
    print(f"Question: {question}")
    print(f"Retrieved IDs: {retrieved_ids}")
    print(f"Relevant IDs: {relevant_ids}")
    print(f"Recall@3: {recall}")
    print(f"Precision@3: {precision}")
    print(f"Hit@3: {hit}")
print("Mean Recall:",sum(springboot_recall_scores)/len(springboot_recall_scores))
print(f"Mean Recall@1: {sum(mean_recall_at_1)/len(mean_recall_at_1)}")
print(f"Mean Recall@2: {sum(mean_recall_at_2)/len(mean_recall_at_2)}")
# spring_boot_question = spring_boot_evaluation_dataset[0]

# query_results = vector_store._collection.query(
#     query_texts=[spring_boot_question["question"]],
#     n_results=3,
#     where={
#         "document_id": "974bbefb-d5a3-4d78-8eab-39972e213e6b"
#     }
# )
# #spring_boot_retrieved_ids = ['adf1f8ee-090e-402a-b6dd-017e173f9c4b', '8c409c30-1267-4453-bca4-cd35e08dfd26', '8be434a2-f766-4948-8bd1-6c0d2e630836']
# spring_boot_retrieved_ids = query_results["ids"][0]
# spring_boot_recall = recall_at_k(
#     spring_boot_retrieved_ids, spring_boot_question["relevant_ids"], k=3
# )
# spring_boot_precision = precision_at_k(
#     spring_boot_retrieved_ids, spring_boot_question["relevant_ids"], k=3
# )
# spring_boot_hit = hit_at_k(
#     spring_boot_retrieved_ids, spring_boot_question["relevant_ids"], k=3
# )

# print(f"Recall@3: {spring_boot_recall}")
# print(f"Precision@3: {spring_boot_precision}")
# print(f"Hit@3: {spring_boot_hit}")


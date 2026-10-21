from ai_document_intelligence.services.retrieval_service import search_documents
from ai_document_intelligence.services.llm_service import generate_answer

document_id = "60879ceb-5ced-43ad-a597-5f49110f3939"
question = "What is dependency injection?"

search_results = search_documents(
    document_id=document_id,
    query=question,
    top_k=3
)

generated_answer = generate_answer(
    question=question,
    search_results=search_results
)

print("Search Results:")
for result in search_results:
    print(f"Page: {result['metadata'].get('page_label')}")
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print()

print("Generated Answer:")
print(generated_answer)
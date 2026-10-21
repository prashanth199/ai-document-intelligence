from ai_document_intelligence.services.retrieval_service import search_documents

search_results = search_documents(
    document_id="60879ceb-5ced-43ad-a597-5f49110f3939",
    query="What is dependency injection?",
    top_k=3
)   

print("Search Results:")
for index, result in enumerate(search_results, start=1):
    print(f"\n--- RESULT {index} ---")
    print("Content:", result["content"])
    print("Metadata:", result["metadata"])
    print("Score:", result["score"])
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

from ai_document_intelligence.core.resources import create_vector_store
from experiments.ollama_evaluator import OllamaEvaluator
from experiments.groq_evaluator import GroqEvaluator

from ai_document_intelligence.services.retrieval_service import search_documents
from ai_document_intelligence.services.llm_service import generate_answer

document_id = "974bbefb-d5a3-4d78-8eab-39972e213e6b"
# question = "What are the main advantages of using Spring Boot?"

# question = "Who is the current Prime Minister of Australia?"

question = "What is the average annual salary of a Spring Boot developer according to this document?"

vector_store = create_vector_store()

search_results = search_documents(
    document_id=document_id,
    query=question,
    vector_store=vector_store
)

print("Retrieved chunks:", len(search_results))

for index, result in enumerate(search_results, start=1):
    print(f"\nChunk {index}:")
    print(result["content"])

actual_answer = generate_answer(
    question=question,
    search_results=search_results
)

print("\nGenerated answer:")
print(actual_answer)

test_case = LLMTestCase(
    input=question,
    actual_output=actual_answer,
    retrieval_context=[
        result["content"]
        for result in search_results
    ]
)

metric = FaithfulnessMetric(
    threshold=0.7,
    model=GroqEvaluator(),
    include_reason=True,
    async_mode=False
)

#metric.measure(test_case)

# print("\nReal RAG Faithfulness Evaluation")
# print("Score:", metric.score)
# print("Reason:", metric.reason)
# print("Passed:", metric.is_successful())
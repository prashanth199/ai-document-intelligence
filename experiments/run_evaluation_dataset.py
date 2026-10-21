from experiments.evaluation_dataset import evaluation_dataset

from ai_document_intelligence.core.resources import create_vector_store
from ai_document_intelligence.services.retrieval_service import search_documents
from ai_document_intelligence.services.llm_service import generate_answer

from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from experiments.groq_evaluator import GroqEvaluator

document_id = "974bbefb-d5a3-4d78-8eab-39972e213e6b"

vector_store = create_vector_store()

evaluator = GroqEvaluator()

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=evaluator,
    include_reason=True,
    async_mode=False
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=evaluator,
    include_reason=True,
    async_mode=False
)

for index, test in enumerate(evaluation_dataset, start=1):
    question = test["question"]
    expected_answer = test["expected_answer"]

    # TODO 1: Retrieve chunks for this question
    search_results = search_documents(
        document_id=document_id,
        query=question,
        vector_store=vector_store
    )

    # TODO 2: Generate an answer using the retrieved chunks
    actual_answer = generate_answer(
        question=question,
        search_results=search_results
    )

    is_unknown_question = expected_answer == "I don't know."

    if is_unknown_question:
        passed = actual_answer.strip().lower() == expected_answer.strip().lower()
        print(f"Test {index} - Unknown-answer test passed: {passed}")

    # TODO 3: Print:
    print(f"""
    {'=' * 60}
    Test Number: {index}
    Question: {question}
    Retrieved Chunk Count: {len(search_results)}
    Expected Answer: {expected_answer}
    Generated Answer: {actual_answer}
    {'=' * 60}
    """)
    # Test number
    # Question
    # Retrieved chunk count
    # Expected answer
    # Generated answer

    if expected_answer != "I don't know.":
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer
        )

        relevancy_metric.measure(test_case)

        print(f"Answer Relevancy Score: {relevancy_metric.score}")
        print(f"Reason: {relevancy_metric.reason}")
        print(f"Passed: {relevancy_metric.is_successful()}")

        faithfulness_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            retrieval_context=[
                result["content"] for result in search_results
            ]
        )

        faithfulness_metric.measure(faithfulness_case)

        print(f"Faithfulness Score: {faithfulness_metric.score}")
        print(f"Faithfulness Reason: {faithfulness_metric.reason}")
        print(f"Faithfulness Passed: {faithfulness_metric.is_successful()}")
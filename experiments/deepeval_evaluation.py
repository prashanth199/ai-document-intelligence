from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from experiments.ollama_evaluator import OllamaEvaluator
from deepeval.metrics import AnswerRelevancyMetric ,ContextualPrecisionMetric, ContextualRecallMetric


test_case = LLMTestCase(
    input="What is Spring Boot?",
    actual_output="Spring Boot simplifies Java application development through auto-configuration and embedded servers.",
    retrieval_context=[
        "Spring Boot is a Java framework that simplifies application development.",
        "Spring Boot provides auto-configuration and embedded servers."
    ],
    expected_output="Spring Boot is a framework for building Java applications with minimal configuration."
)

print(test_case)

metric = FaithfulnessMetric(
    threshold=0.7,
    model=OllamaEvaluator(),
    include_reason=True,
    async_mode=False
)

metric.measure(test_case)

print("Faithfulness score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())

relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=OllamaEvaluator(),
    include_reason=True,
    async_mode=False
)

relevancy_metric.measure(test_case)

print("Answer Relevancy:", relevancy_metric.score)
print("Reason:", relevancy_metric.reason)
print("Passed:", relevancy_metric.is_successful())

context_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=OllamaEvaluator(),
    include_reason=True,
    async_mode=False
)

context_precision_metric.measure(test_case)

print("Contextual Precision:", context_precision_metric.score)
print("Reason:", context_precision_metric.reason)
print("Passed:", context_precision_metric.is_successful())

context_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=OllamaEvaluator(),
    include_reason=True,
    async_mode=False
)

context_recall_metric.measure(test_case)

print("Contextual Recall:", context_recall_metric.score)
print("Reason:", context_recall_metric.reason)
print("Passed:", context_recall_metric.is_successful())
from unittest.mock import MagicMock

from ai_document_intelligence.services.llm_service import generate_answer
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def test_generate_answer_with_empty_context():
    mock_llm = MagicMock()

    answer = generate_answer(
        question="Who is the Prime Minister of Australia?",
        search_results=[],
        llm=mock_llm
    )

    assert answer == "I don't know."
    mock_llm.invoke.assert_not_called()

def test_generate_answer_with_context():
    fake_model = FakeListChatModel(
        responses=["Spring Boot simplifies Java application development."]
    )

    search_results = [
        {
            "content": "Spring Boot simplifies Java application development.",
            "metadata": {"page_label": "1"},
            "score": 0.2
        }
    ]

    answer = generate_answer(
        question="What is Spring Boot?",
        search_results=search_results,
        llm=fake_model
    )
    assert answer == "Spring Boot simplifies Java application development."

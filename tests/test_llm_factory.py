from ai_document_intelligence.core.llm_factory import create_llm
from ai_document_intelligence.config import Settings
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import pytest


def test_create_llm_with_ollama(monkeypatch):
    settings = Settings(
        llm_provider="ollama",
        ollama_model="qwen3"
    )

    monkeypatch.setattr(
        "ai_document_intelligence.core.llm_factory.get_settings",
        lambda: settings
    )

    llm = create_llm()

    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3"


def test_create_llm_with_groq(monkeypatch):
    settings = Settings(
        llm_provider="groq",
        groq_model="openai/gpt-oss-20b",
        groq_api_key="test-api-key"
    )

    monkeypatch.setattr(
        "ai_document_intelligence.core.llm_factory.get_settings",
        lambda: settings
    )

    llm = create_llm()

    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "openai/gpt-oss-20b"


def test_create_llm_with_missing_groq_api_key(monkeypatch):
    settings = Settings(
        llm_provider="groq",
        groq_api_key=None
    )

    monkeypatch.setattr(
        "ai_document_intelligence.core.llm_factory.get_settings",
        lambda: settings
    )

    with pytest.raises(
        ValueError,
        match="GROQ_API_KEY is required"
    ):
        create_llm()
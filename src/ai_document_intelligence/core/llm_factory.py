from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from ai_document_intelligence.config import get_settings


def create_llm() -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=2048,
            num_predict=256,
            reasoning=False,
        )

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq"
            )

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )
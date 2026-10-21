
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from langchain_ollama import ChatOllama



load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3.8-27B",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0,
# )

# chat_model = ChatHuggingFace(llm=llm)

chat_model = ChatOllama(
    model="qwen3",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    reasoning=False
)


def generate_answer(question: str, search_results: list[dict]) -> str:

    if not search_results:
        return "I don't know."

    """Generate an answer to a question based on search results."""
    context = "\n\n".join(
        f"""
        Page: {result['metadata'].get('page_label')}
        Content:
        {result['content']}
        """
        for result in search_results
    )

    prompt = ChatPromptTemplate.from_messages([
        (  
            
            "system",
            """
            You are an AI assistant that provides helpful answers to questions based on the provided context.
            If the answer cannot be found in the context, respond with "I don't know."
            """,
        ),
        (
            "human",
            """
            Context:
            {context}

            Question: {question}
            """,
        ),
    ])

    chain = prompt | chat_model

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.content

    
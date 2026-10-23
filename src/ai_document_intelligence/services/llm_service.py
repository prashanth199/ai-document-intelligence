
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
# from dotenv import load_dotenv

# from langchain_ollama import ChatOllama



# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3.8-27B",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0,
# )

# chat_model = ChatHuggingFace(llm=llm)



def generate_answer(question: str, search_results: list[dict],llm: BaseChatModel) -> str:

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

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.content

    
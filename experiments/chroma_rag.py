from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="document_intelligence",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

print("Stored chunks:", vector_store._collection.count())

question = "Who founded Microsoft?"

results = vector_store.similarity_search_with_score(
    query=question,
    k=3
)

context = "\n\n".join(
    f"""
    Source: {document.metadata.get("source")}
    Page: {document.metadata.get("page_label")}
    Content:
    {document.page_content}
    """
    for document,_ in results
)

print("\nRETRIEVED CONTEXT:")
print(context)

for index, (document, score) in enumerate(results, start=1):
    print(f"\n--- RESULT {index} ---")
    print("Content:", document.page_content)
    print("Page:", document.metadata.get("page_label"))
    print("Source:", document.metadata.get("source"))
    print("Score:", score)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    task="text-generation",
    max_new_tokens=256,
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document question-answering assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that the provided document does not contain enough information.

When possible, mention the page number that supports your answer.
"""
    ),
    (
        "human",
        """
Context:

{context}

Question:

{question}
"""
    )
])

chain = prompt | model

response = chain.invoke({
    "context": context, 
    "question": question
})
print("\nANSWER:")
print(response.content)
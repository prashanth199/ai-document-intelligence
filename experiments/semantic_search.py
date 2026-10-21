from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path("sample_documents/sample.pdf")

reader = PdfReader(pdf_path)

documents = []

for page_number, page in enumerate(reader.pages):
    text = page.extract_text() or ""

    document = Document(
        page_content=text,
        metadata={
            "source": str(pdf_path),
            "page": page_number,
            "page_label": str(page_number + 1),
            "total_pages": len(reader.pages),
        }
    )

    documents.append(document)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

chunks = splitter.split_documents(documents)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]
chunk_vectors = embeddings.embed_documents(chunk_texts)

print("Number of chunk vectors:", len(chunk_vectors))
print("Vector dimensions:", len(chunk_vectors[0]))

question = "Who is the CEO of Microsoft?"

question_vector = embeddings.embed_query(question)

print("Question vector dimensions:", len(question_vector))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

results = []

for index, chunk_vector in enumerate(chunk_vectors):
    similarity = cosine_similarity(question_vector, chunk_vector)
    print(f"Chunk {index + 1} similarity:", similarity)

    results.append((similarity, index))

results.sort(reverse=True)

top_results = results[:3]

print("\nTOP 3 RESULTS")

for similarity, index in top_results:
    chunk = chunks[index]

    print(f"\n--- CHUNK {index + 1} ---")
    print("Similarity:", similarity)
    print("Page:", chunk.metadata["page_label"])
    print("Content:")
    print(chunk.page_content)

context_parts = []

for similarity, index in top_results:
    chunk = chunks[index]

    context_parts.append(
        f"""
        Source: {chunk.metadata["source"]}
        Page: {chunk.metadata["page_label"]}
        Content:
        {chunk.page_content}
        """
    )

context = "\n".join(context_parts)

print("\nRAG CONTEXT:")
print(context)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    task="text-generation",
    max_new_tokens=256,
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a document question-answering assistant.

        Answer the user's question using only the provided context.

        If the context does not contain enough information to answer the question,
        say that the provided document does not contain enough information.

        When possible, mention the page number that supports the answer.
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

print("\nRAG ANSWER:")
print(response.content)
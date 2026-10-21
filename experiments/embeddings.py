from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Docker is a containerization platform."

vector = embeddings.embed_query(text)

print("TYPE:")
print(type(vector))

print("\nDIMENSIONS:")
print(len(vector))

print("\nFIRST 10 VALUES:")
print(vector[:10])

texts = [
    "Docker runs applications inside containers.",
    "Containers provide isolated application environments.",
    "The football team won the championship."
]

vectors = embeddings.embed_documents(texts)

print("\nNumber of vectors:", len(vectors))
print("Dimensions per vector:", len(vectors[0]))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

print("\nSIMILARITIES:")

print(
    "Docker vs Containers:",
    cosine_similarity(vectors[0], vectors[1])
)

print(
    "Docker vs Football:",
    cosine_similarity(vectors[0], vectors[2])
)
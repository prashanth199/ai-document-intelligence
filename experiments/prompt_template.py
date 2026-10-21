from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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
        "You are an experienced software architect."
    ),
    (
        "human",
        "Explain {topic} in a way that is appropriate for a {experience_level}."
    )
])

chain = prompt | model

response = chain.invoke({
    "topic": "Docker",
    "experience_level": "Senior Backend Engineer"
})

# messages = prompt.invoke({
#     "topic": "Docker"
# })

# print(messages)

# response = model.invoke(messages)
print(response.content)
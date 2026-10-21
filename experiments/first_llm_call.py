from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    task="text-generation",
    max_new_tokens=256,
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

# response = model.invoke(
#     "Explain what a REST API is in one paragraph."
# )

# print(response.content)

messages = [
    SystemMessage(
        content="You are an experienced software architect."
    ),
    HumanMessage(
        content="Explain microservices in simple terms."
    ),
]

response = model.invoke(messages)

print("TYPE:")
print(type(response))

print("\nFULL RESPONSE:")
print(response)

print("\nCONTENT:")
print(response.content)


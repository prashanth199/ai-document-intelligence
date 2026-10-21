from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class DocumentAnalysis(BaseModel):
    title: str = Field(
        description="A short title describing the document"
    )
    summary: str = Field(
        description="A concise summary of the document"
    )
    category: str = Field(
        description="The category the document belongs to"
    )
    key_points: list[str] = Field(
        description="The most important points from the document"
    )


parser = PydanticOutputParser(
    pydantic_object=DocumentAnalysis
)


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
        "You analyze documents and extract structured information."
    ),
    (
        "human",
        "Analyze the following document:\n\n{document} {format_instructions}"
    )
])

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

response = chain.invoke({
    "document": """
    Docker is a containerization platform that allows developers
    to package applications together with their dependencies.
    Applications are packaged into Docker images and executed as
    containers. Containers provide consistent environments across
    development, testing, and production.
    """
})

# messages = prompt.invoke({
#     "topic": "Docker"
# })

# print(messages)

# response = model.invoke(messages)
print(response)
print(response.title)
print(type(response))
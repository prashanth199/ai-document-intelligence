from pydantic import BaseModel, Field

class AskQuestionRequest(BaseModel):
    question: str = Field(min_length=1)

class SourceResponse(BaseModel):
    page: str
    text: str
    score: float

class AskQuestionResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: list[SourceResponse]
    retrieved_chunks: int
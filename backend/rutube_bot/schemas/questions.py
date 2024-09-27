from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str

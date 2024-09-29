from typing import Optional, List

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    class_1: Optional[str] = 'some class'
    class_2: Optional[str] = 'some class'
    docs: List[str]
    total_docs: int

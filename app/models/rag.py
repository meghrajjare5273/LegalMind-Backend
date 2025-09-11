from pydantic import BaseModel, Field
from typing import List

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique chat session")
    message: str = Field(..., description="User question")

class ChatResponse(BaseModel):
    answer: str
    context_chunks: List[str]
    sources: List[str] = []

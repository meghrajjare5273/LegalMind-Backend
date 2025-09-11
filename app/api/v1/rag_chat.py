from fastapi import APIRouter, HTTPException
from models.rag import ChatRequest, ChatResponse
from services.rag.chat import answer_question

router = APIRouter(prefix="/rag-chat", tags=["RAG Chat"])

@router.post("", response_model=ChatResponse)
async def rag_chat(req: ChatRequest):
    try:
        answer, ctx = answer_question(req.message)
        return ChatResponse(answer=answer, context_chunks=ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

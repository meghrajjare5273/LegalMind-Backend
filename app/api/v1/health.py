from fastapi import APIRouter
from core.config import settings
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)

@router.get("")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "version": "5.0.0",
            "services": {
                "contract_analysis": "available",
                "rag_chat": "available",
                "gemini_ai": "available" if settings.GEMINI_API_KEY else "unavailable",
                "pinecone": "available" if settings.PINECONE_API_KEY else "unavailable"
            },
            "environment": settings.ENV
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }

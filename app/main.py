from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import health, contract_analysis, rag_chat
from core.config import settings
import os

app = FastAPI(
    title="LegalMind API", 
    version="5.0.0",
    description="AI-Powered Legal Assistant with Contract Analysis and RAG Chat"
)

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "https://legal-mind-eight.vercel.app",
    os.getenv("FRONTEND_URL", "")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in ALLOWED_ORIGINS if origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(contract_analysis.router, prefix="/api/v1") 
app.include_router(rag_chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LegalMind Backend API - AI-Powered Legal Assistant",
        "version": "5.0.0",
        "endpoints": {
            "contract_analysis": "/api/v1/contract-analysis/analyze",
            "rag_chat": "/api/v1/rag-chat",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }

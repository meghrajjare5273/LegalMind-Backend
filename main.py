import os
import asyncio
import json
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from pdf_utils.pdf_functions import extract_text_from_pdf, is_valid_pdf
from contracts.analyzer import HybridContractAnalyzer
from utils.cache import CacheManager

from agents.orchestrator import OrchestratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
analyzer = None
orchestrator = None
cache_manager = CacheManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global analyzer, orchestrator
    try:
        logger.info("Initializing Hybrid Contract Analyzer and Orchestrator")
        analyzer = HybridContractAnalyzer()
        orchestrator = OrchestratorAgent()
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Application shutdown")

    

app = FastAPI(
    title="LegalMind Backend API",
    description="AI-Powered Legal Contract Analysis with Hybrid Intelligence",
    version="4.0.0",
    lifespan=lifespan
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

# Pydantic Models
class RiskAnalysis(BaseModel):
    sentence: str = Field(..., description="The problematic sentence")
    risk_category: str = Field(..., description="Category of risk")
    risk_level: str = Field(..., description="Risk severity level")
    risk_type: str = Field(..., description="Type of legal risk")
    description: str = Field(..., description="Detailed risk explanation")
    specific_concerns: List[str] = Field(default_factory=list)
    negotiation_strategies: List[str] = Field(default_factory=list)
    priority_score: int = Field(..., ge=1, le=10, description="Priority from 1-10")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    legal_concepts: List[str] = Field(default_factory=list)
    entities: List[Dict] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    alternative_language: str = Field(default="")
    cost_implications: str = Field(default="")

class ContractSection(BaseModel):
    title: str
    content: str
    risk_count: int
    section_type: str = Field(default="general")

class RiskSummary(BaseModel):
    total_risks: int
    critical_risk_count: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    overall_risk_level: str
    risk_distribution: Dict[str, int] = Field(default_factory=dict)

class AnalysisResponse(BaseModel):
    filename: str
    extracted_text: str
    analysis: List[RiskAnalysis]
    summary: RiskSummary
    sections: List[ContractSection]
    recommendations: List[str]
    overall_summary: str
    document_complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    party_power_balance: float = Field(default=0.5, ge=0.0, le=1.0)
    processing_time: Optional[float] = None

# API Endpoints
@app.post("/extract_and_analyze", response_model=AnalysisResponse)
async def extract_and_analyze(file: UploadFile = File(...)):
    """Extract text from PDF and perform comprehensive contract analysis"""
    
    # Validate file type
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Validate file size (max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Validate PDF format
    if not is_valid_pdf(file_content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Extract text
        logger.info(f"Processing file: {file.filename}")
        extracted_text = extract_text_from_pdf(file_content)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable text found in PDF"
            )
        
        # Perform analysis
        analysis_result = await analyzer.analyze_contract(extracted_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = AnalysisResponse(
            filename=file.filename,
            extracted_text=extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text,
            analysis=analysis_result["analyses"],
            summary=analysis_result["summary"],
            sections=analysis_result["sections"],
            recommendations=analysis_result["recommendations"],
            overall_summary=analysis_result["overall_summary"],
            document_complexity_score=analysis_result.get("complexity_score", 0.3),
            party_power_balance=analysis_result.get("power_balance", 0.5),
            processing_time=processing_time
        )
        
        logger.info(f"Analysis completed for {file.filename} in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        ai_status = "available" if analyzer and analyzer.gemini_client else "unavailable"
        return {
            "status": "healthy",
            "version": app.version,
            "analyzer": "hybrid-intelligent",
            "ai_enhancement": ai_status,
            "cache_size": cache_manager.get_cache_info(),
            "uptime": "running"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LegalMind Backend API - AI-Powered Contract Analysis",
        "version": app.version,
        "endpoints": {
            "analysis": "/extract_and_analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") != "production"
    )

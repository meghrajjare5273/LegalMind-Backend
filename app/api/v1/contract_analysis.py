from fastapi import APIRouter, HTTPException, UploadFile, File, status
from models.contract import AnalysisResponse
from services.pdf.processor import PDFProcessor
from services.contract.analyzer import EnhancedContractAnalyzer
from utils.cache import cache_result
import time
import logging
import hashlib

router = APIRouter(prefix="/contract-analysis", tags=["Contract Analysis"])
logger = logging.getLogger(__name__)

# Initialize analyzer as a singleton
# Initialize enhanced analyzer as a singleton
_enhanced_analyzer = None

def get_enhanced_analyzer():
    global _enhanced_analyzer
    if _enhanced_analyzer is None:
        logger.info("Initializing Enhanced Contract Analyzer...")
        _enhanced_analyzer = EnhancedContractAnalyzer()
        logger.info("Enhanced Contract Analyzer initialized successfully")
    return _enhanced_analyzer

@router.post("/analyze", response_model=AnalysisResponse)
async def extract_and_analyze_enhanced(file: UploadFile = File(...)):
    """Extract text from PDF and perform enhanced contract analysis"""
    
    # Keep all your existing validation logic
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    file_content = await file.read()
    
    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    try:
        start_time = time.time()
        
        if not PDFProcessor.is_valid_pdf(file_content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format"
            )
        
        logger.info(f"Processing file: {file.filename}")
        extracted_text = PDFProcessor.extract_text_from_pdf(file_content)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable text found in PDF"
            )
        
        # Use enhanced analyzer
        analyzer = get_enhanced_analyzer()
        analysis_result = await analyzer.analyze_contract(extracted_text)
        
        processing_time = time.time() - start_time
        
        # Build response (keep your existing structure)
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
        
        logger.info(f"Enhanced analysis completed for {file.filename} in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced analysis failed: {str(e)}"
        )
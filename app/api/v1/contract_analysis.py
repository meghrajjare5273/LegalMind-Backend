from fastapi import APIRouter, HTTPException, UploadFile, File, status
from models.contract import AnalysisResponse
from services.pdf.processor import PDFProcessor
from services.contract.analyzer import HybridContractAnalyzer
import time
import logging

router = APIRouter(prefix="/contract-analysis", tags=["Contract Analysis"])
logger = logging.getLogger(__name__)

# Initialize analyzer (you might want to make this a dependency)
analyzer = HybridContractAnalyzer()

@router.post("/analyze", response_model=AnalysisResponse)
async def extract_and_analyze(file: UploadFile = File(...)):
    """Extract text from PDF and perform comprehensive contract analysis"""
    
    # Validate file type
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Validate file size (max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    file_content = await file.read()
    
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Validate PDF format
    if not PDFProcessor.is_valid_pdf(file_content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format"
        )
    
    try:
        start_time = time.time()
        
        # Extract text
        logger.info(f"Processing file: {file.filename}")
        extracted_text = PDFProcessor.extract_text_from_pdf(file_content)
        
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

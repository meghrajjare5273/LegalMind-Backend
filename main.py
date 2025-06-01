import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_utils.pdf_functions import extract_text_from_pdf

app = FastAPI(
    title="LegalMind Backend API",
    description="API for LegalMind an AI Application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "https://legal-mind-eight.vercel.app/*", "https://legal-mind-eight.vercel.app/contract-review"],  # Add your NextJS ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Risk keywords for contract analysis
RISK_KEYWORDS = [
    "liability", "penalty", "termination", "breach", "default", "damages",
    "indemnification", "force majeure", "confidentiality", "non-compete",
    "limitation", "exclusion", "warranty", "guarantee", "arbitration",
    "jurisdiction", "governing law", "intellectual property", "patent",
    "copyright", "trademark", "liquidated damages", "consequential damages",
    "punitive damages", "injunctive relief", "specific performance"
]

class RiskAnalysis(BaseModel):
    sentence: str
    risk: str
    explanation: str
    negotiation_tip: str

@app.post("/extract_and_analyze")
async def extract_and_analyze(file: UploadFile = File(...)):
    """
    Combined endpoint to extract text from PDF and analyze for contract risks.
    
    Args:
        file (UploadFile): The uploaded PDF file.
    
    Returns:
        dict: Combined results of text extraction and risk analysis.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")
    
    content = await file.read()
    try:
        # Extract text
        text = extract_text_from_pdf(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Analyze contract
        sentences = re.split(r'(?<=[.!?])\s+', text)
        analysis = []
        
        # Analyze each sentence for risks
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            for keyword in RISK_KEYWORDS:
                if keyword.lower() in sentence.lower():
                    analysis.append(RiskAnalysis(
                        sentence=sentence,
                        risk=f"Potential risk related to {keyword}",
                        explanation=f"This sentence mentions '{keyword}', which can be a critical term in contracts.",
                        negotiation_tip=f"Consider negotiating the terms related to {keyword} to minimize potential liabilities."
                    ))
                    break
        
        # Return results regardless of whether risks are found
        return {
            "filename": file.filename,
            "extracted_text": text,
            "analysis": [risk.dict() for risk in analysis],
            "risks_found": len(analysis)
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
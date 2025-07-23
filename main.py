import re
import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_utils.pdf_functions import extract_text_from_pdf
from contracts.contract_analyzer import AdvancedContractAnalyzer
from typing import List, Dict, Optional
import logging
from cachetools import TTLCache
import json
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalMind Backend API",
    description="AI-Powered Legal Contract Analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:3001", 
        "https://legal-mind-eight.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models to match frontend expectations
class EnhancedRiskAnalysis(BaseModel):
    sentence: str
    risk_category: str
    risk_level: str  # "HIGH" | "MEDIUM" | "LOW"
    risk_type: str
    description: str
    specific_concerns: List[str]
    negotiation_strategies: List[str]
    priority_score: int  # 1-10 scale
    # Additional fields from advanced analyzer
    confidence_score: Optional[float] = 0.0
    legal_concepts: Optional[List[str]] = []
    entities: Optional[List[Dict]] = []
    mitigation_strategies: Optional[List[str]] = []

class ContractSection(BaseModel):
    title: str
    content: str
    risk_count: int

class RiskSummary(BaseModel):
    total_risks: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    overall_risk_level: str

class EnhancedExtractAndAnalyzeResponse(BaseModel):
    filename: str
    extracted_text: str
    analysis: List[EnhancedRiskAnalysis]
    summary: RiskSummary
    sections: List[ContractSection]
    recommendations: List[str]
    overall_summary: str
    # Additional enhanced features
    document_complexity_score: Optional[float] = 0.0
    party_power_balance: Optional[float] = 0.5
    compliance_coverage: Optional[Dict[str, float]] = {}

# Global analyzer instance
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the advanced contract analyzer on startup"""
    global analyzer
    try:
        logger.info("Initializing Advanced Contract Analyzer...")
        analyzer = AdvancedContractAnalyzer()
        logger.info("Advanced Contract Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        # Fallback to basic analyzer if advanced initialization fails
        from contracts.contract_analyzer import ContractAnalyzer
        analyzer = ContractAnalyzer()
        logger.warning("Using fallback basic analyzer")

class EnhancedContractAnalyzer:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.gemini_client = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini client if API key is available"""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("Gemini API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    async def analyze_contract_comprehensive(self, text: str) -> Dict:
        """Comprehensive contract analysis combining advanced NLP and Gemini AI"""
        try:
            # Use advanced NLP analyzer
            if hasattr(analyzer, 'analyze_contract_advanced'):
                analyses, doc_profile = await analyzer.analyze_contract_advanced(text)
                
                # Convert to frontend-compatible format
                enhanced_analyses = []
                for analysis in analyses:
                    # Map advanced analysis to frontend format
                    enhanced_risk = {
                        "sentence": analysis.sentence,
                        "risk_category": analysis.risk_category,
                        "risk_level": analysis.risk_level.value,
                        "risk_type": analysis.clause_type.value,
                        "description": analysis.risk_explanation,
                        "specific_concerns": [analysis.risk_explanation],
                        "negotiation_strategies": analysis.mitigation_strategies,
                        "priority_score": analysis.negotiation_priority,
                        "confidence_score": analysis.confidence_score,
                        "legal_concepts": analysis.legal_concepts,
                        "entities": [{"text": e.text, "label": e.label, "confidence": e.confidence} 
                                   for e in analysis.entities],
                        "mitigation_strategies": analysis.mitigation_strategies
                    }
                    
                    # Enhance with Gemini if available
                    if self.gemini_client:
                        enhanced_risk = await self._enhance_with_gemini(enhanced_risk)
                    
                    enhanced_analyses.append(enhanced_risk)
                
                # Extract sections using advanced analysis
                sections = self._extract_sections_from_advanced_analysis(analyses, text)
                
                return {
                    "analyses": enhanced_analyses,
                    "doc_profile": doc_profile,
                    "sections": sections
                }
            else:
                # Fallback to basic analysis
                return await self._fallback_analysis(text)
                
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            return await self._fallback_analysis(text)

    async def _enhance_with_gemini(self, risk_analysis: Dict) -> Dict:
        """Enhance risk analysis with Gemini API insights"""
        if not self.gemini_client:
            return risk_analysis
        
        try:
            prompt = f"""
            Analyze this contract risk and provide enhanced insights:
            
            Sentence: "{risk_analysis['sentence']}"
            Risk Category: {risk_analysis['risk_category']}
            Current Analysis: {risk_analysis['description']}
            
            Provide:
            1. Enhanced risk description
            2. Specific legal concerns
            3. Practical negotiation strategies
            4. Priority assessment (1-10)
            
            Respond in JSON format.
            """
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            try:
                gemini_insights = json.loads(response.text)
                
                # Merge Gemini insights with existing analysis
                risk_analysis.update({
                    "description": gemini_insights.get("enhanced_description", risk_analysis["description"]),
                    "specific_concerns": gemini_insights.get("specific_concerns", risk_analysis["specific_concerns"]),
                    "negotiation_strategies": gemini_insights.get("negotiation_strategies", risk_analysis["negotiation_strategies"]),
                    "priority_score": min(gemini_insights.get("priority_score", risk_analysis["priority_score"]), 10)
                })
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini response as JSON")
                
        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
        
        return risk_analysis

    def _extract_sections_from_advanced_analysis(self, analyses, text: str) -> List[Dict]:
        """Extract contract sections from advanced analysis"""
        sections = []
        
        # Group analyses by clause type
        section_map = {}
        for analysis in analyses:
            section_name = analysis.clause_type.value.replace('_', ' ').title()
            if section_name not in section_map:
                section_map[section_name] = []
            section_map[section_name].append(analysis)
        
        # Create sections
        for section_name, section_analyses in section_map.items():
            # Extract representative content
            content_sentences = [a.sentence for a in section_analyses[:3]]  # Limit to 3 sentences
            content = " ... ".join(content_sentences)
            
            sections.append({
                "title": section_name,
                "content": content,
                "risk_count": len(section_analyses)
            })
        
        return sections

    async def _fallback_analysis(self, text: str) -> Dict:
        """Fallback analysis using basic analyzer"""
        if hasattr(analyzer, 'analyze_contract_risks'):
            risks = await analyzer.analyze_contract_risks(text)
            
            # Convert basic risks to enhanced format
            enhanced_analyses = []
            for risk in risks:
                enhanced_analyses.append({
                    "sentence": risk.get("sentence", ""),
                    "risk_category": risk.get("risk_category", "General"),
                    "risk_level": risk.get("risk_level", "MEDIUM"),
                    "risk_type": risk.get("specific_risk", "General Risk"),
                    "description": risk.get("explanation", ""),
                    "specific_concerns": risk.get("potential_consequences", []),
                    "negotiation_strategies": risk.get("negotiation_strategies", []),
                    "priority_score": 5,  # Default priority
                    "confidence_score": 0.7,
                    "legal_concepts": [],
                    "entities": [],
                    "mitigation_strategies": []
                })
            
            return {
                "analyses": enhanced_analyses,
                "doc_profile": None,
                "sections": []
            }
        
        return {"analyses": [], "doc_profile": None, "sections": []}

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedContractAnalyzer()

@app.post("/extract_and_analyze", response_model=EnhancedExtractAndAnalyzeResponse)
async def extract_and_analyze(file: UploadFile = File(...)):
    """Enhanced contract analysis with advanced NLP capabilities"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")

    content = await file.read()
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Perform comprehensive analysis
        analysis_result = await enhanced_analyzer.analyze_contract_comprehensive(text)
        analyses = analysis_result["analyses"]
        doc_profile = analysis_result["doc_profile"]
        sections = analysis_result["sections"]

        # Calculate summary statistics
        high_count = sum(1 for risk in analyses if risk["risk_level"] == "HIGH")
        medium_count = sum(1 for risk in analyses if risk["risk_level"] == "MEDIUM")
        low_count = sum(1 for risk in analyses if risk["risk_level"] == "LOW")
        
        # Determine overall risk level
        if high_count > 0:
            overall_risk = "HIGH"
        elif medium_count > 2:
            overall_risk = "MEDIUM-HIGH"
        elif medium_count > 0:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        # Generate recommendations
        recommendations = []
        if high_count > 3:
            recommendations.append("🚨 Multiple high-risk clauses detected. Legal review strongly recommended.")
        if doc_profile and hasattr(doc_profile, 'party_power_balance') and doc_profile.party_power_balance < 0.3:
            recommendations.append("⚖️ Contract appears heavily weighted against one party. Review for balance.")
        if len(analyses) > 10:
            recommendations.append("📋 Complex contract with numerous risk factors. Consider comprehensive legal review.")
        
        if not recommendations:
            recommendations.append("✅ Contract shows manageable risk profile. Standard review recommended.")

        # Create response matching frontend interface
        response = EnhancedExtractAndAnalyzeResponse(
            filename=file.filename,
            extracted_text=text,
            analysis=[EnhancedRiskAnalysis(**analysis) for analysis in analyses],
            summary=RiskSummary(
                total_risks=len(analyses),
                high_risk_count=high_count,
                medium_risk_count=medium_count,
                low_risk_count=low_count,
                overall_risk_level=overall_risk
            ),
            sections=[ContractSection(**section) for section in sections],
            recommendations=recommendations,
            overall_summary=f"Contract analysis complete. Found {len(analyses)} risks with {high_count} high-priority issues requiring attention.",
            document_complexity_score=doc_profile.legal_complexity_score if doc_profile else 0.3,
            party_power_balance=doc_profile.party_power_balance if doc_profile else 0.5,
            compliance_coverage=doc_profile.compliance_coverage if doc_profile else {}
        )

        return response

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "analyzer_type": "advanced" if hasattr(analyzer, 'analyze_contract_advanced') else "basic",
        "gemini_available": enhanced_analyzer.gemini_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Enhanced main.py with improved risk analysis

import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_utils.pdf_functions import extract_text_from_pdf
from typing import List, Dict

app = FastAPI(
    title="LegalMind Backend API",
    description="API for LegalMind an AI Application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "https://legal-mind-eight.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced risk patterns with detailed analysis
RISK_PATTERNS = {
    "liability": {
        "keywords": ["liability", "liable", "responsible for", "shall be liable"],
        "risk_level": "HIGH",
        "category": "Financial Risk",
        "description": "Clauses that assign responsibility for damages or losses",
        "red_flags": ["unlimited liability", "joint and several liability", "strict liability"],
        "negotiation_tips": [
            "Cap liability to a specific dollar amount",
            "Exclude consequential and indirect damages",
            "Add mutual liability limitations"
        ]
    },
    "termination": {
        "keywords": ["terminate", "termination", "end this agreement", "cancel"],
        "risk_level": "MEDIUM",
        "category": "Contract Continuity",
        "description": "Conditions under which the contract can be ended",
        "red_flags": ["immediate termination", "terminate at will", "no notice required"],
        "negotiation_tips": [
            "Require reasonable notice period (30-90 days)",
            "Include cure periods for breaches",
            "Define specific termination triggers"
        ]
    },
    "indemnification": {
        "keywords": ["indemnify", "indemnification", "hold harmless", "defend"],
        "risk_level": "HIGH",
        "category": "Financial Risk",
        "description": "Obligations to protect another party from legal claims",
        "red_flags": ["broad indemnification", "indemnify against all claims", "no exceptions"],
        "negotiation_tips": [
            "Limit indemnification to specific scenarios",
            "Add carve-outs for gross negligence",
            "Make indemnification mutual where possible"
        ]
    },
    "intellectual_property": {
        "keywords": ["intellectual property", "patent", "copyright", "trademark", "trade secret"],
        "risk_level": "HIGH",
        "category": "IP Risk",
        "description": "Rights and obligations regarding intellectual property",
        "red_flags": ["broad IP assignment", "work for hire", "unlimited license"],
        "negotiation_tips": [
            "Retain ownership of pre-existing IP",
            "Limit scope of IP grants",
            "Include IP indemnification provisions"
        ]
    },
    "confidentiality": {
        "keywords": ["confidential", "non-disclosure", "proprietary", "trade secret"],
        "risk_level": "MEDIUM",
        "category": "Information Security",
        "description": "Obligations to protect sensitive information",
        "red_flags": ["perpetual confidentiality", "broad definition", "no standard exceptions"],
        "negotiation_tips": [
            "Add standard exceptions (publicly available, independently developed)",
            "Limit confidentiality period (3-5 years typical)",
            "Define what constitutes confidential information"
        ]
    },
    "warranty": {
        "keywords": ["warrant", "warranty", "guarantee", "represent"],
        "risk_level": "MEDIUM",
        "category": "Performance Risk",
        "description": "Promises about product/service quality or legal compliance",
        "red_flags": ["unlimited warranties", "fitness for particular purpose", "no warranty disclaimers"],
        "negotiation_tips": [
            "Limit warranties to specific, measurable criteria",
            "Include warranty disclaimers where appropriate",
            "Set reasonable warranty periods"
        ]
    },
    "force_majeure": {
        "keywords": ["force majeure", "act of god", "unforeseeable circumstances"],
        "risk_level": "LOW",
        "category": "Performance Risk",
        "description": "Clauses excusing performance due to extraordinary events",
        "red_flags": ["no force majeure clause", "narrow definition", "no notice requirements"],
        "negotiation_tips": [
            "Include comprehensive force majeure clause",
            "Add pandemic/epidemic events",
            "Specify notice and mitigation requirements"
        ]
    }
}

class EnhancedRiskAnalysis(BaseModel):
    sentence: str
    risk_category: str
    risk_level: str
    risk_type: str
    description: str
    specific_concerns: List[str]
    negotiation_strategies: List[str]
    priority_score: int  # 1-10 scale

class ContractSection(BaseModel):
    title: str
    content: str
    risk_count: int

@app.post("/extract_and_analyze")
async def extract_and_analyze(file: UploadFile = File(...)):
    """
    Enhanced endpoint to extract text from PDF and perform comprehensive risk analysis.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")

    content = await file.read()
    try:
        # Extract text
        text = extract_text_from_pdf(content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Enhanced analysis
        analysis_results = analyze_contract_comprehensively(text)
        
        return {
            "filename": file.filename,
            "extracted_text": text,
            "analysis": analysis_results["detailed_risks"],
            "summary": analysis_results["summary"],
            "sections": analysis_results["sections"],
            "recommendations": analysis_results["recommendations"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def analyze_contract_comprehensively(text: str) -> Dict:
    """
    Perform comprehensive contract analysis with context-aware risk assessment.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    detailed_risks = []
    risk_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    # Analyze each sentence with enhanced logic
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:  # Skip very short sentences
            continue

        # Check against each risk pattern
        for risk_type, pattern_info in RISK_PATTERNS.items():
            if any(keyword.lower() in sentence.lower() for keyword in pattern_info["keywords"]):
                
                # Calculate priority score based on red flags and context
                priority_score = calculate_priority_score(sentence, pattern_info)
                
                # Identify specific concerns in this sentence
                specific_concerns = identify_specific_concerns(sentence, pattern_info)
                
                risk_analysis = EnhancedRiskAnalysis(
                    sentence=sentence,
                    risk_category=pattern_info["category"],
                    risk_level=pattern_info["risk_level"],
                    risk_type=risk_type.replace("_", " ").title(),
                    description=pattern_info["description"],
                    specific_concerns=specific_concerns,
                    negotiation_strategies=pattern_info["negotiation_tips"],
                    priority_score=priority_score
                )
                
                detailed_risks.append(risk_analysis.dict())
                risk_summary[pattern_info["risk_level"]] += 1
                break  # Only match first pattern to avoid duplicates

    # Sort risks by priority score (highest first)
    detailed_risks.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Generate sections analysis
    sections = analyze_contract_sections(text)
    
    # Generate recommendations
    recommendations = generate_recommendations(detailed_risks, risk_summary)
    
    return {
        "detailed_risks": detailed_risks,
        "summary": {
            "total_risks": len(detailed_risks),
            "high_risk_count": risk_summary["HIGH"],
            "medium_risk_count": risk_summary["MEDIUM"],
            "low_risk_count": risk_summary["LOW"],
            "overall_risk_level": determine_overall_risk(risk_summary)
        },
        "sections": sections,
        "recommendations": recommendations
    }

def calculate_priority_score(sentence: str, pattern_info: Dict) -> int:
    """
    Calculate priority score (1-10) based on red flags and context.
    """
    base_score = {"HIGH": 7, "MEDIUM": 5, "LOW": 3}[pattern_info["risk_level"]]
    
    # Check for red flags
    red_flag_bonus = 0
    for red_flag in pattern_info["red_flags"]:
        if red_flag.lower() in sentence.lower():
            red_flag_bonus += 2
    
    # Check for protective language (reduces score)
    protective_terms = ["limited to", "except", "excluding", "maximum", "cap", "not exceed"]
    protective_penalty = 0
    for term in protective_terms:
        if term.lower() in sentence.lower():
            protective_penalty += 1
    
    final_score = min(10, max(1, base_score + red_flag_bonus - protective_penalty))
    return final_score

def identify_specific_concerns(sentence: str, pattern_info: Dict) -> List[str]:
    """
    Identify specific concerns within a sentence based on context.
    """
    concerns = []
    sentence_lower = sentence.lower()
    
    # Check for concerning modifiers
    concerning_modifiers = {
        "unlimited": "No caps or limits specified",
        "immediately": "No grace period or notice",
        "sole discretion": "One-sided decision making power",
        "at will": "Can be exercised without cause",
        "irrevocable": "Cannot be reversed or modified",
        "perpetual": "No time limitations",
        "broad": "Overly wide scope"
    }
    
    for modifier, concern in concerning_modifiers.items():
        if modifier in sentence_lower:
            concerns.append(concern)
    
    # Add pattern-specific red flags found
    for red_flag in pattern_info["red_flags"]:
        if red_flag.lower() in sentence_lower:
            concerns.append(f"Contains: {red_flag}")
    
    return concerns[:3]  # Limit to top 3 concerns

def analyze_contract_sections(text: str) -> List[Dict]:
    """
    Identify and analyze different contract sections.
    """
    # Simple section detection based on common contract headings
    section_patterns = [
        r'(?i)(payment|compensation|fees?)\s*(terms?|provisions?|clause)?',
        r'(?i)(termination|expiration)\s*(clause|provisions?)?',
        r'(?i)(liability|damages)\s*(limitation|clause|provisions?)?',
        r'(?i)(intellectual\s*property|ip)\s*(rights?|clause|provisions?)?',
        r'(?i)(confidentiality|non-disclosure)\s*(agreement|clause|provisions?)?',
        r'(?i)(warranty|warranties|guarantee)\s*(clause|provisions?|disclaimer)?'
    ]
    
    sections = []
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            section_name = matches[0].group(1).title()
            # Extract surrounding context (simplified)
            start_pos = max(0, matches[0].start() - 200)
            end_pos = min(len(text), matches[0].end() + 500)
            section_content = text[start_pos:end_pos].strip()
            
            sections.append({
                "title": section_name,
                "content": section_content[:300] + "..." if len(section_content) > 300 else section_content,
                "risk_count": len([r for r in RISK_PATTERNS.keys() if any(k.lower() in section_content.lower() for k in RISK_PATTERNS[r]["keywords"])])
            })
    
    return sections

def determine_overall_risk(risk_summary: Dict) -> str:
    """
    Determine overall contract risk level.
    """
    if risk_summary["HIGH"] >= 3:
        return "HIGH"
    elif risk_summary["HIGH"] >= 1 or risk_summary["MEDIUM"] >= 5:
        return "MEDIUM-HIGH"
    elif risk_summary["MEDIUM"] >= 2:
        return "MEDIUM"
    else:
        return "LOW"

def generate_recommendations(detailed_risks: List[Dict], risk_summary: Dict) -> List[str]:
    """
    Generate actionable recommendations based on risk analysis.
    """
    recommendations = []
    
    # Priority recommendations based on high-risk items
    high_priority_risks = [r for r in detailed_risks if r["priority_score"] >= 8]
    if high_priority_risks:
        recommendations.append(f"⚠️ URGENT: Address {len(high_priority_risks)} high-priority risk(s) before signing")
    
    # Category-specific recommendations
    risk_categories = {}
    for risk in detailed_risks:
        category = risk["risk_category"]
        if category not in risk_categories:
            risk_categories[category] = 0
        risk_categories[category] += 1
    
    if risk_categories.get("Financial Risk", 0) >= 2:
        recommendations.append("💰 Consider adding financial protection clauses (liability caps, insurance requirements)")
    
    if risk_categories.get("IP Risk", 0) >= 1:
        recommendations.append("🔒 Review intellectual property provisions carefully - consider IP attorney consultation")
    
    # General recommendations
    if risk_summary["HIGH"] + risk_summary["MEDIUM"] >= 5:
        recommendations.append("📋 Recommend full legal review before execution")
    
    recommendations.append("✅ Negotiate the highlighted terms to better balance risk allocation")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
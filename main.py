import re
import os
# import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pdf_utils.pdf_functions import extract_text_from_pdf
from typing import List, Dict, Optional
import logging
from cachetools import TTLCache
# import asyncio
# from typing import List, Dict
import json
from google import genai

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalMind Backend API",
    description="API for LegalMind an AI Application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "https://legal-mind-eight.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RiskAnalysis(BaseModel):
    sentence: str
    risk_category: str
    risk_level: str
    specific_risk: str
    explanation: str
    negotiation_tip: str
    potential_consequences: List[str]
    negotiation_strategies: List[str]
    red_flags: List[str]
    sentence_index: int
    section_context: Optional[str] = None

class ContractAnalyzer:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour
        # Expanded risk keywords for local analysis
        self.risk_keywords = {
            "liability": ("Liability", "High", ["Unlimited liability could lead to significant financial loss"]),
            "termination": ("Termination", "Medium", ["Abrupt termination could disrupt operations"]),
            "indemnification": ("Indemnification", "High", ["Broad indemnification may expose parties to unforeseen liabilities"]),
            "warranties": ("Warranties", "Medium", ["Unclear or broad warranties may increase risk of breach claims"]),
            "intellectual property": ("Intellectual Property", "High", ["IP clauses may result in loss of ownership or usage rights"]),
            "confidentiality": ("Confidentiality", "Medium", ["Weak confidentiality terms may lead to data leaks or reputational harm"]),
            "payment terms": ("Payment Terms", "Medium", ["Unfavorable payment terms can impact cash flow and financial planning"]),
            "force majeure": ("Force Majeure", "Low", ["Lack of force majeure provisions may increase exposure to uncontrollable events"]),
            "governing law": ("Governing Law", "Low", ["Unfavorable jurisdiction may complicate dispute resolution"]),
            "dispute resolution": ("Dispute Resolution", "Medium", ["Unclear dispute resolution process may lead to costly litigation"]),
            "assignment": ("Assignment", "Medium", ["Assignment clauses may allow unwanted transfer of obligations"]),
            "limitation of liability": ("Limitation of Liability", "High", ["Absence of liability caps may result in excessive damages"]),
            "exclusivity": ("Exclusivity", "Medium", ["Exclusive agreements may restrict future business opportunities"]),
            "non-compete": ("Non-Compete", "Medium", ["Non-compete clauses may limit future employment or business options"]),
            "audit": ("Audit", "Low", ["Audit rights may lead to privacy or operational concerns"])
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def _analyze_sentence_locally(self, sentence: str, index: int) -> List[Dict]:
        """Perform basic local risk analysis on a sentence."""
        risks = []
        sentence_lower = sentence.lower()
        for keyword, (category, level, consequences) in self.risk_keywords.items():
            if keyword in sentence_lower:
                risks.append({
                    "sentence": sentence,
                    "risk_category": category,
                    "risk_level": level,
                    "specific_risk": f"Potential {keyword} issue",
                    "explanation": f"This sentence mentions '{keyword}', indicating a potential {category.lower()} risk.",
                    "negotiation_tip": f"Consider negotiating the terms related to {keyword}.",
                    "potential_consequences": consequences,
                    "negotiation_strategies": [f"Propose a cap on {keyword} exposure"],
                    "red_flags": [f"Check for {keyword} clauses"],
                    "sentence_index": index,
                    "section_context": None
                })
        return risks

    async def analyze_contract_risks(self, text: str) -> List[Dict]:
        """Analyze contract text for risks, enhancing with Gemini API."""
        risks = []
        sentences = self._split_into_sentences(text)
        risk_data_batch = []

        # Collect all risks locally
        for i, sentence in enumerate(sentences):
            sentence_risks = self._analyze_sentence_locally(sentence, i)
            risk_data_batch.extend(sentence_risks)

        # Enhance with Gemini API if risks are found
        if risk_data_batch:
            enhanced_risks = await self._enhance_with_gemini_api(risk_data_batch)
            risks.extend(enhanced_risks)

        return risks

    async def _enhance_with_gemini_api(self, risk_data: List[Dict]) -> List[Dict]:
        """Enhance risk data with Gemini API explanations and tips."""
        cached_results = {}
        uncached_risks = []
        batch_input = []

        # Check cache for each risk
        for risk in risk_data:
            cache_key = f"{risk['sentence_index']}_{risk['sentence']}_{risk['risk_category']}"
            if cache_key in self.cache:
                cached_results[cache_key] = self.cache[cache_key]
            else:
                uncached_risks.append(risk)
                batch_input.append(risk)

        logger.info(f"Processing {len(risk_data)} risks: {len(cached_results)} cached, {len(uncached_risks)} to fetch")

        # If all risks are cached, return cached results
        if not uncached_risks:
            return [cached_results[f"{r['sentence']}_{r['risk_category']}"] for r in risk_data]

        # Batch API call for uncached risks (split if too large, e.g., >50)
        batch_size = 50
        api_responses = []
        for i in range(0, len(uncached_risks), batch_size):
            batch = batch_input[i:i + batch_size]
            responses = await self._call_gemini_api(batch)
            api_responses.extend(responses)

        # Process API responses
        for i, risk in enumerate(uncached_risks):
            api_response = api_responses[i] if i < len(api_responses) else {}
            explanation = api_response.get("explanation", risk["explanation"])
            negotiation_tip = api_response.get("negotiation_tip", 
                                              f"Consider negotiating the terms related to {risk['risk_category'].lower()}.")
            risk["explanation"] = explanation
            risk["negotiation_tip"] = negotiation_tip
            cache_key = f"{risk['sentence']}_{risk['risk_category']}"
            self.cache[cache_key] = risk

        # Combine cached and newly fetched results
        result = []
        uncached_index = 0
        for r in risk_data:
            cache_key = f"{r['sentence']}_{r['risk_category']}"
            if cache_key in cached_results:
                result.append(cached_results[cache_key])
            else:
                result.append(uncached_risks[uncached_index])
                uncached_index += 1
        return result



    async def _call_gemini_api(self, batch_input: List[Dict]) -> List[Dict]:
        """Make an asynchronous call to the Gemini API."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        client = genai.Client(api_key=api_key)
        
        try:
            # Process each risk item in the batch
            results = []
            
            for risk in batch_input:
                # Create a comprehensive prompt for analysis
                prompt = f"""
                Analyze the following contract clause for risks:

                Contract Text: "{risk['sentence']}"
                Risk Category: {risk['risk_category']}
                Initial Analysis: {risk['explanation']}

                Please provide:
                1. Risk level assessment (HIGH/MEDIUM/LOW)
                2. Specific concerns about this clause
                3. Negotiation strategies to mitigate risks
                4. Priority score (1-10)

                Respond in JSON format:
                {{
                    "risk_level": "HIGH|MEDIUM|LOW",
                    "specific_concerns": ["concern1", "concern2"],
                    "negotiation_strategies": ["strategy1", "strategy2"],
                    "priority_score": 1-10
                }}
                """
                
                # Make the API call
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                
                try:
                    # Parse the JSON response
                    analysis_result = json.loads(response.text)
                    results.append(analysis_result)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    logger.warning(f"Failed to parse Gemini response as JSON: {response.text}")
                    results.append({
                        "risk_level": "MEDIUM",
                        "specific_concerns": ["Analysis unavailable"],
                        "negotiation_strategies": ["Consult legal professional"],
                        "priority_score": 5
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            # Return fallback data to maintain structure
            return [{
                "risk_level": "MEDIUM",
                "specific_concerns": ["API analysis unavailable"],
                "negotiation_strategies": ["Manual review recommended"],
                "priority_score": 5
            } for _ in batch_input]
    
# Initialize analyzer
analyzer = ContractAnalyzer()

@app.post("/extract_and_analyze")
async def extract_and_analyze(file: UploadFile = File(...)):
    """Extract text from PDF and analyze for contract risks with Gemini API integration."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File is not a PDF")

    content = await file.read()
    try:
        # Extract text
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Analyze contract using ContractAnalyzer with async API call
        risks = await analyzer.analyze_contract_risks(text)
        analysis = [RiskAnalysis(**risk) for risk in risks]

        # Calculate summary statistics
        high_count = sum(1 for risk in analysis if risk.risk_level == "HIGH")
        medium_count = sum(1 for risk in analysis if risk.risk_level == "MEDIUM")
        low_count = sum(1 for risk in analysis if risk.risk_level == "LOW")
        
        # Determine overall risk level
        if high_count > 0:
            overall_risk = "HIGH"
        elif medium_count > 2:
            overall_risk = "MEDIUM-HIGH"
        elif medium_count > 0:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"


        return {
            "filename": file.filename,
            "extracted_text": text,
            "analysis": [risk.model_dump() for risk in analysis],
            "risks_found": len(analysis),
            "summary": {
                "total_risks": len(analysis),
                "high_risk_count": high_count,
                "medium_risk_count": medium_count,
                "low_risk_count": low_count,
                "overall_risk_level": overall_risk
            },
            "sections": [],  # Add your section extraction logic here
            "recommendations": [],  # Add your recommendations logic here
            "overall_summary": f"Contract analysis complete. Found {len(analysis)} risks with {high_count} high-priority issues."
        }
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
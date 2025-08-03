import os
import asyncio
import json
import hashlib
import logging
from typing import List, Dict, Optional

from cachetools import TTLCache
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_utils.pdf_functions import extract_text_from_pdf
from contracts.contract_analyzer import LightweightContractAnalyzer
from google.genai import types

# --------------------------------------------------------------------------- #
# Google Gemini setup
# --------------------------------------------------------------------------- #
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini SDK not installed – running without LLM boosts")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# FastAPI initialisation
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="LegalMind Backend API",
    description="AI-Powered Legal Contract Analysis (Serverless-Optimised)",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://legal-mind-eight.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic models used by the API
# --------------------------------------------------------------------------- #
class EnhancedRiskAnalysis(BaseModel):
    sentence: str
    risk_category: str
    risk_level: str
    risk_type: str
    description: str
    specific_concerns: List[str]
    negotiation_strategies: List[str]
    priority_score: int
    confidence_score: Optional[float] = 0.0
    legal_concepts: Optional[List[str]] = []
    entities: Optional[List[Dict]] = []
    mitigation_strategies: Optional[List[str]] = []
    negotiation_tactics: Optional[List[str]] = []
    alternative_language: Optional[str] = ""
    cost_implications: Optional[str] = ""


class ContractSection(BaseModel):
    title: str
    content: str
    risk_count: int


class RiskSummary(BaseModel):
    total_risks: int
    critical_risk_count: Optional[int] = 0
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
    document_complexity_score: Optional[float] = 0.0
    party_power_balance: Optional[float] = 0.5
    compliance_coverage: Optional[Dict[str, float]] = {}


# --------------------------------------------------------------------------- #
# In-memory caches
# --------------------------------------------------------------------------- #
analyzer_cache: TTLCache = TTLCache(maxsize=1, ttl=3_600)     # 1 hour
ai_cache: TTLCache = TTLCache(maxsize=500, ttl=1_800)         # 30 minutes


def get_analyzer() -> LightweightContractAnalyzer:
    """Return a cached instance of the lightweight analyzer."""
    if "analyzer" not in analyzer_cache:
        analyzer_cache["analyzer"] = LightweightContractAnalyzer()
    return analyzer_cache["analyzer"]


# --------------------------------------------------------------------------- #
# Generative processor that adds Gemini-powered context
# --------------------------------------------------------------------------- #
class GenerativeContractProcessor:
    def __init__(self) -> None:
        self.gemini_client = self._initialise_gemini()

    @staticmethod
    def _initialise_gemini():
        """Create a Gemini client if the SDK and key are present."""
        if not GEMINI_AVAILABLE:
            return None

        api_key = "AIzaSyCWqH4CpR1EfWcmF-yiq26xrwxyooPcrDs"
        if not api_key:
            logger.warning("GEMINI_API_KEY not set – running without LLM boosts")
            return None

        try:
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialised")
            return client
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini initialisation failed: %s", exc)
            return None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    async def analyse_contract(self, text: str) -> Dict:
        analyzer = get_analyzer()

        try:
            analyses, doc_profile = analyzer.analyze_contract_advanced(text)

            # Convert to API-friendly dicts and augment with AI
            api_analyses: List[Dict] = []
            for analysis in analyses:
                base = {
                    "sentence": analysis.sentence,
                    "risk_category": analysis.risk_category,
                    "risk_level": analysis.risk_level.value,
                    "risk_type": analysis.clause_type.value.replace("_", " ").title(),
                    "description": analysis.risk_explanation,
                    "specific_concerns": [analysis.risk_explanation],
                    "negotiation_strategies": analysis.mitigation_strategies,
                    "priority_score": analysis.negotiation_priority,
                    "confidence_score": analysis.confidence_score,
                    "legal_concepts": analysis.legal_concepts,
                    "entities": analysis.entities,
                    "mitigation_strategies": analysis.mitigation_strategies,
                    "negotiation_tactics": [],
                    "alternative_language": "",
                    "cost_implications": "",
                }

                # Enrich with Gemini
                if self.gemini_client:
                    enriched = await self._generate_contextual_analysis(
                        sentence=analysis.sentence,
                        risk_category=analysis.risk_category,
                        risk_level=analysis.risk_level.value,
                        entities=analysis.entities,
                    )

                    base.update(
                        {
                            "description": enriched.get(
                                "risk_explanation", base["description"]
                            ),
                            "specific_concerns": enriched.get(
                                "specific_concerns", base["specific_concerns"]
                            ),
                            "negotiation_strategies": enriched.get(
                                "mitigation_strategies",
                                base["negotiation_strategies"],
                            ),
                            "negotiation_tactics": enriched.get("negotiation_tactics", []),
                            "alternative_language": enriched.get("alternative_language", ""),
                            "cost_implications": enriched.get("cost_implications", ""),
                        }
                    )

                api_analyses.append(base)

            # Global recommendations
            recommendations = await self._generate_contract_recommendations(
                text, api_analyses
            )

            sections = self._create_sections(api_analyses)

            return {
                "analyses": api_analyses,
                "doc_profile": doc_profile,
                "sections": sections,
                "recommendations": recommendations,
            }

        except Exception as exc:  # pragma: no cover
            logger.error("Contract analysis failed: %s", exc)
            return {
                "analyses": [],
                "doc_profile": None,
                "sections": [],
                "recommendations": [
                    "⚠️ Analysis failed – please retry or seek human review."
                ],
            }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    async def _generate_contextual_analysis(
        self,
        sentence: str,
        risk_category: str,
        risk_level: str,
        entities: List[Dict],
    ) -> Dict:
        """Invoke Gemini (with caching) to deepen risk commentary."""
        cache_key = hashlib.md5(
            f"{sentence}_{risk_category}_{risk_level}".encode()
        ).hexdigest()
        if cache_key in ai_cache:
            return ai_cache[cache_key]

        if not self.gemini_client:
            return self._get_fallback_analysis(risk_category, risk_level)

        prompt = self._build_clause_prompt(
            sentence, risk_category, risk_level, entities
        )
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=800)
            )

            cleaned = self._strip_triple_backticks(response.text.strip())
            generated = json.loads(cleaned)
            ai_cache[cache_key] = generated
            return generated
        except (json.JSONDecodeError, Exception) as exc:  # pragma: no cover
            logger.warning("Gemini parsing failed: %s", exc)
            return self._get_fallback_analysis(risk_category, risk_level)

    @staticmethod
    def _build_clause_prompt(
        sentence: str, risk_category: str, risk_level: str, entities: List[Dict]
    ) -> str:
        entity_context = (
            ", ".join(f"{e['label']}: {e['text']}" for e in entities) if entities else "None"
        )
        return f"""
You are a legal AI assistant analysing a contract clause. Produce a detailed JSON assessment.

CLAUSE: "{sentence}"
RISK TYPE: {risk_category}
RISK LEVEL: {risk_level}
EXTRACTED ENTITIES: {entity_context}

Return JSON:
{{
  "risk_explanation": "...",
  "specific_concerns": ["..."],
  "mitigation_strategies": ["..."],
  "negotiation_tactics": ["..."],
  "alternative_language": "...",
  "cost_implications": "..."
}}

Avoid generic statements; ground every comment in the clause content.
""".strip()

    async def _generate_contract_recommendations(
        self, text: str, analyses: List[Dict]
    ) -> List[str]:
        if not self.gemini_client:
            return self._get_fallback_recommendations(analyses)

        prompt = self._build_recommendation_prompt(analyses)

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                generation_config={"temperature": 0.4, "max_output_tokens": 600},
            )

            cleaned = self._strip_triple_backticks(response.text.strip())
            recs = json.loads(cleaned)
            return recs if isinstance(recs, list) else self._get_fallback_recommendations(
                analyses
            )
        except (json.JSONDecodeError, Exception):
            return self._get_fallback_recommendations(analyses)

    @staticmethod
    def _build_recommendation_prompt(analyses: List[Dict]) -> str:
        # Summarise top few risks per level for the LLM
        levels = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for a in analyses:
            levels.setdefault(a["risk_level"], []).append(a)

        parts: List[str] = []
        for lvl, items in levels.items():
            if items:
                parts.append(f"{lvl} RISKS ({len(items)}):")
                for i in items[:3]:
                    parts.append(f"- {i['risk_category']}: {i['sentence'][:100]}...")
                if len(items) > 3:
                    parts.append(f"...and {len(items) - 3} more")
        risk_summary = "\n".join(parts)

        return f"""
You are a legal AI assistant. Based on the following analysis summary, provide 3-5 actionable, prioritised recommendations:

{risk_summary}

Output JSON list:
["Recommendation 1", "Recommendation 2", ...]
Prefix with emoji 🚨/⚠️/🔍/✅ according to criticality.
""".strip()

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_triple_backticks(text: str) -> str:
        """Remove optional `````` wrappers from LLM output."""
        if text.startswith("```"):
            text = text.lstrip("`").rstrip("`")
        return text.strip()

    @staticmethod
    def _get_fallback_analysis(risk_category: str, risk_level: str) -> Dict:
        return {
            "risk_explanation": f"This {risk_category.lower()} clause carries {risk_level.lower()} risk.",
            "specific_concerns": ["Requires legal review"],
            "mitigation_strategies": ["Seek professional legal advice"],
            "negotiation_tactics": ["Negotiate balanced terms"],
            "alternative_language": "Consider more neutral wording",
        }

    @staticmethod
    def _get_fallback_recommendations(analyses: List[Dict]) -> List[str]:
        high = sum(1 for a in analyses if a["risk_level"] == "HIGH")
        critical = sum(1 for a in analyses if a["risk_level"] == "CRITICAL")
        total = len(analyses)

        recs: List[str] = []
        if critical:
            recs.append("🚨 Critical risks discovered – urgent legal attention needed")
        if high > 2:
            recs.append("⚠️ Several high-risk clauses – engage counsel for renegotiation")
        if total > 8:
            recs.append("📋 Complex contract with numerous risks – comprehensive review advised")
        if not recs:
            recs.append("✅ Manageable risk profile – proceed with standard review")
        return recs

    @staticmethod
    def _create_sections(analyses: List[Dict]) -> List[Dict]:
        grouped: Dict[str, List[Dict]] = {}
        for a in analyses:
            grouped.setdefault(a["risk_type"], []).append(a)

        sections: List[Dict] = []
        for title, items in grouped.items():
            examples = " | ".join(
                (s := it["sentence"])[:100] + ("..." if len(s) > 100 else "")
                for it in items[:2]
            )
            sections.append({"title": title, "content": examples, "risk_count": len(items)})
        return sections


# Single, module-level processor instance
processor = GenerativeContractProcessor()

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/extract_and_analyze", response_model=EnhancedExtractAndAnalyzeResponse)
async def extract_and_analyze(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text")

        result = await processor.analyse_contract(text)

        analyses = result["analyses"]
        dp = result["doc_profile"]
        sections = result["sections"]
        recs = result["recommendations"]

        critical = sum(1 for r in analyses if r["risk_level"] == "CRITICAL")
        high = sum(1 for r in analyses if r["risk_level"] == "HIGH")
        medium = sum(1 for r in analyses if r["risk_level"] == "MEDIUM")
        low = sum(1 for r in analyses if r["risk_level"] == "LOW")

        if critical:
            overall = "CRITICAL"
        elif high > 3:
            overall = "HIGH"
        elif high:
            overall = "MEDIUM-HIGH"
        elif medium > 2:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        return EnhancedExtractAndAnalyzeResponse(
            filename=file.filename,
            extracted_text=text,
            analysis=[EnhancedRiskAnalysis(**a) for a in analyses],
            summary=RiskSummary(
                total_risks=len(analyses),
                critical_risk_count=critical,
                high_risk_count=high,
                medium_risk_count=medium,
                low_risk_count=low,
                overall_risk_level=overall,
            ),
            sections=[ContractSection(**s) for s in sections],
            recommendations=recs,
            overall_summary=f"{len(analyses)} risks found – {critical} critical, {high} high.",
            document_complexity_score=getattr(dp, "legal_complexity_score", 0.3) if dp else 0.3,
            party_power_balance=getattr(dp, "party_power_balance", 0.5) if dp else 0.5,
            compliance_coverage=getattr(dp, "compliance_coverage", {}) if dp else {},
        )

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("Processing error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version + "-generative",
        "analyzer": "lightweight-generative",
        "gemini_available": processor.gemini_client is not None,
        "memory_optimized": True,
        "serverless_ready": True,
    }


@app.get("/")
async def root():
    return {
        "message": "LegalMind Backend API – AI-Powered Contract Analysis",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

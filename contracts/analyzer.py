import re
import os
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.cache import cache_result
from contracts.patterns import ContractPatterns
from contracts.nlp_pipeline import LightweightNLPPipeline

# Optional Gemini integration
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ClauseType(Enum):
    TERMINATION = "Termination"
    LIABILITY = "Liability"
    PAYMENT = "Payment"
    PENALTY = "Penalty"
    CONFIDENTIALITY = "Confidentiality"
    INTELLECTUAL_PROPERTY = "Intellectual Property"
    INDEMNIFICATION = "Indemnification"
    GOVERNING_LAW = "Governing Law"
    ARBITRATION = "Arbitration"
    FORCE_MAJEURE = "Force Majeure"
    GENERAL = "General"

@dataclass
class RiskItem:
    sentence: str
    risk_category: str
    risk_level: RiskLevel
    clause_type: ClauseType
    description: str
    concerns: List[str]
    strategies: List[str]
    priority: int
    confidence: float

class HybridContractAnalyzer:
    """Hybrid analyzer with lightweight NLP pipeline"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.nlp_pipeline = LightweightNLPPipeline()
        self.gemini_client = self._initialize_gemini()
        
    def _initialize_gemini(self) -> Optional[object]:
        """Initialize Gemini client safely"""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini SDK not available")
            return None
            
        # api_key = os.getenv("GEMINI_API_KEY")
        api_key = "AIzaSyCWqH4CpR1EfWcmF-yiq26xrwxyooPcrDs"
        
        if not api_key:
            logger.warning("GEMINI_API_KEY not configured")
            return None
            
        try:
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None
    
    async def analyze_contract(self, text: str) -> Dict:
        """Main analysis method with NLP pipeline"""
        try:
            # Step 1: Use NLP pipeline to extract risky portions
            risky_portions = self.nlp_pipeline.extract_risky_portions(text, max_portions=10)
            logger.info(f"Extracted {len(risky_portions)} risky portions")
            
            # Step 2: Rule-based analysis on risky portions only
            rule_based_risks = []
            for portion in risky_portions:
                sentence = portion['sentence']
                risks = self._analyze_sentence_with_rules(sentence)
                rule_based_risks.extend(risks)
            
            # Step 3: AI enhancement on top 3 highest-priority risks only
            enhanced_risks = []
            top_risks = sorted(rule_based_risks, key=lambda x: x.priority, reverse=True)[:3]
            
            for risk in rule_based_risks:
                if self.gemini_client and risk in top_risks:
                    # Create focused prompt with key phrases only
                    relevant_portion = next((p for p in risky_portions if p['sentence'] == risk.sentence), None)
                    if relevant_portion:
                        enhanced_risk = await self._enhance_with_focused_ai(risk, relevant_portion)
                        enhanced_risks.append(enhanced_risk)
                    else:
                        enhanced_risks.append(self._convert_to_api_format(risk))
                else:
                    enhanced_risks.append(self._convert_to_api_format(risk))
            
            # Step 4: Generate summary and recommendations
            summary = self._generate_summary(enhanced_risks)
            sections = self._extract_sections(enhanced_risks)
            recommendations = await self._generate_recommendations(enhanced_risks, text)
            
            return {
                "analyses": enhanced_risks,
                "summary": summary,
                "sections": sections,
                "recommendations": recommendations,
                "overall_summary": self._create_overall_summary(enhanced_risks),
                "complexity_score": self._calculate_complexity(text),
                "power_balance": self._assess_power_balance(text)
            }
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            raise
    
    def _analyze_sentence_with_rules(self, sentence: str) -> List[RiskItem]:
        """Analyze a single sentence using rule-based patterns"""
        risks = []
        sentence_lower = sentence.lower()
        
        # Track if any pattern matched
        pattern_matched = False
        
        for pattern_data in self.patterns.get_all_patterns():
            for pattern in pattern_data["patterns"]:
                if re.search(pattern, sentence_lower):
                    risk = RiskItem(
                        sentence=sentence,
                        risk_category=pattern_data["category"],
                        risk_level=RiskLevel[pattern_data["risk_level"]],
                        clause_type=ClauseType[pattern_data["clause_type"]],
                        description=pattern_data["description"],
                        concerns=pattern_data["concerns"],
                        strategies=pattern_data["strategies"],
                        priority=pattern_data["priority"],
                        confidence=pattern_data["confidence"]
                    )
                    risks.append(risk)
                    pattern_matched = True
                    break  # Avoid duplicate matches for same category
        
        # If no specific patterns matched, create a generic risk item
        if not pattern_matched:
            risks.append(RiskItem(
                sentence=sentence,
                risk_category="General Risk",
                risk_level=RiskLevel.MEDIUM,  # Default to medium
                clause_type=ClauseType.GENERAL,
                description="Potentially risky clause identified by AI analysis",
                concerns=["Requires legal review", "May contain unfavorable terms"],
                strategies=["Review clause carefully", "Consider negotiation", "Seek legal advice"],
                priority=5,  # Medium priority
                confidence=0.7  # Lower confidence for generic matches
            ))
        
        return risks

    
    @cache_result(ttl=1800)
    async def _enhance_with_focused_ai(self, risk: RiskItem, portion_data: Dict) -> Dict:
        """Enhance risk analysis using focused AI with minimal data"""
        if not self.gemini_client:
            return self._convert_to_api_format(risk)
        
        try:
            # Create minimal prompt with key phrases only
            key_phrases = portion_data.get('key_phrases', [])[:3]  # Limit to top 3
            focused_prompt = self._build_minimal_prompt(risk, key_phrases)
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=focused_prompt,
                config={
                    "temperature": 0.3, 
                    "max_output_tokens": 300  # Reduced token limit
                }
            )
            
            ai_enhancement = self._parse_ai_response(response.text)
            return self._merge_analyses(risk, ai_enhancement)
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return self._convert_to_api_format(risk)
    
    def _build_minimal_prompt(self, risk: RiskItem, key_phrases: List[str]) -> str:
        """Build minimal prompt focusing only on key risk elements"""
        phrases_text = ", ".join(key_phrases[:3]) if key_phrases else "N/A"
        
        return f"""
Risk: {risk.risk_category}
Level: {risk.risk_level.value}
Key phrases: {phrases_text}

Provide brief JSON response:
{{
  "enhanced_description": "Brief explanation",
  "specific_concerns": ["concern1", "concern2"],
  "negotiation_strategies": ["strategy1", "strategy2"],
  "alternative_language": "Brief suggested wording"
}}
""".strip()
    
    def _parse_ai_response(self, response_text: str) -> dict:
        """Parse AI response safely with better error handling and robust JSON extraction."""
        try:
            # Clean response: strip whitespace and remove any leading/trailing non-JSON text
            cleaned = response_text.strip()

            # Step 1: Remove markdown code blocks if present
            # This regex captures content between ``````
            # Uses DOTALL to handle multi-line, and greedy match for the largest block
            code_block_match = re.search(r'``````', cleaned, re.DOTALL | re.IGNORECASE)
            if code_block_match:
                cleaned = code_block_match.group(1)
            else:
                # If no code block, try to find the first valid JSON-like structure
                # This matches objects {}, arrays [], or simple values, allowing nested structures
                json_match = re.search(r'(?:\{.*?\} | \[\s*.*?]\s* | "(?:[^"\\]|\\.)*" | \d+\.?\d* | true | false | null)', cleaned, re.DOTALL | re.VERBOSE)
                if json_match:
                    cleaned = json_match.group(0)
                # If still no match, assume the whole cleaned text is the JSON

            # Step 2: Attempt to parse as JSON
            parsed = json.loads(cleaned)
            
            # Ensure it's a dict (as per your original return type); convert if needed
            if not isinstance(parsed, dict):
                if isinstance(parsed, list):
                    parsed = {"data": parsed}  # Wrap arrays in a dict for consistency
                else:
                    raise ValueError(f"Parsed JSON is not a dict or list: {type(parsed)}")

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Original response text: {response_text}")
            logger.debug(f"Cleaned text: {cleaned}")
            return {}
        except Exception as e:
            logger.warning(f"Unexpected error parsing AI response: {e}")
            return {}
    
    def _merge_analyses(self, rule_risk: RiskItem, ai_enhancement: Dict) -> Dict:
        """Merge rule-based and AI analyses"""
        return {
            "sentence": rule_risk.sentence,
            "risk_category": rule_risk.risk_category,
            "risk_level": rule_risk.risk_level.value,
            "risk_type": rule_risk.clause_type.value,
            "description": ai_enhancement.get("enhanced_description", rule_risk.description),
            "specific_concerns": ai_enhancement.get("specific_concerns", rule_risk.concerns),
            "negotiation_strategies": ai_enhancement.get("negotiation_strategies", rule_risk.strategies),
            "priority_score": rule_risk.priority,
            "confidence_score": rule_risk.confidence,
            "legal_concepts": [rule_risk.risk_category],
            "entities": [],
            "mitigation_strategies": rule_risk.strategies,
            "alternative_language": ai_enhancement.get("alternative_language", ""),
            "cost_implications": ""
        }
    
    def _convert_to_api_format(self, risk: RiskItem) -> Dict:
        """Convert RiskItem to API format without AI enhancement"""
        return {
            "sentence": risk.sentence,
            "risk_category": risk.risk_category,
            "risk_level": risk.risk_level.value,
            "risk_type": risk.clause_type.value,
            "description": risk.description,
            "specific_concerns": risk.concerns,
            "negotiation_strategies": risk.strategies,
            "priority_score": risk.priority,
            "confidence_score": risk.confidence,
            "legal_concepts": [risk.risk_category],
            "entities": [],
            "mitigation_strategies": risk.strategies,
            "alternative_language": "",
            "cost_implications": ""
        }
    
    def _generate_summary(self, risks: List[Dict]) -> Dict:
        """Generate risk summary statistics"""
        total = len(risks)
        critical = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high = sum(1 for r in risks if r["risk_level"] == "HIGH")
        medium = sum(1 for r in risks if r["risk_level"] == "MEDIUM")
        low = sum(1 for r in risks if r["risk_level"] == "LOW")
        
        # Determine overall risk level
        if critical > 0:
            overall = "CRITICAL"
        elif high > 3:
            overall = "HIGH"
        elif high > 0:
            overall = "MEDIUM-HIGH"
        elif medium > 2:
            overall = "MEDIUM"
        else:
            overall = "LOW"
        
        # Risk distribution by category
        risk_distribution = {}
        for risk in risks:
            category = risk["risk_category"]
            risk_distribution[category] = risk_distribution.get(category, 0) + 1
        
        return {
            "total_risks": total,
            "critical_risk_count": critical,
            "high_risk_count": high,
            "medium_risk_count": medium,
            "low_risk_count": low,
            "overall_risk_level": overall,
            "risk_distribution": risk_distribution
        }
    
    def _extract_sections(self, risks: List[Dict]) -> List[Dict]:
        """Extract contract sections from identified risks"""
        sections_map = {}
        
        for risk in risks:
            section_type = risk["risk_type"]
            if section_type not in sections_map:
                sections_map[section_type] = {
                    "sentences": [],
                    "count": 0
                }
            
            sections_map[section_type]["sentences"].append(risk["sentence"][:100] + "...")
            sections_map[section_type]["count"] += 1
        
        sections = []
        for section_type, data in sections_map.items():
            content = " | ".join(data["sentences"][:2])
            if len(data["sentences"]) > 2:
                content += f" | ... ({len(data['sentences']) - 2} more)"
            
            sections.append({
                "title": section_type,
                "content": content,
                "risk_count": data["count"],
                "section_type": section_type.lower().replace(" ", "_")
            })
        
        return sorted(sections, key=lambda x: x["risk_count"], reverse=True)
    
    async def _generate_recommendations(self, risks: List[Dict], full_text: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        critical_count = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high_count = sum(1 for r in risks if r["risk_level"] == "HIGH")
        total_risks = len(risks)
        
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count} critical risk(s) detected - immediate legal review required")
        
        if high_count > 3:
            recommendations.append(f"⚠️ {high_count} high-risk clauses identified - negotiate these terms before signing")
        
        if total_risks > 10:
            recommendations.append("📋 Complex contract with numerous risks - consider comprehensive legal consultation")
        
        # Specific category recommendations
        risk_categories = [r["risk_category"] for r in risks]
        if "Liability Risk" in risk_categories:
            recommendations.append("🛡️ Review liability clauses for balanced risk allocation")
        
        if "Payment Risk" in risk_categories:
            recommendations.append("💰 Examine payment terms and penalty clauses carefully")
        
        if not recommendations:
            recommendations.append("✅ Manageable risk profile - standard legal review recommended")
        
        return recommendations
    
    def _create_overall_summary(self, risks: List[Dict]) -> str:
        """Create overall analysis summary"""
        total = len(risks)
        critical = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high = sum(1 for r in risks if r["risk_level"] == "HIGH")
        
        if critical > 0:
            return f"Analysis complete: {total} total risks with {critical} critical and {high} high-priority issues requiring immediate attention."
        elif high > 0:
            return f"Analysis complete: {total} total risks with {high} high-priority issues requiring careful review."
        else:
            return f"Analysis complete: {total} total risks identified with manageable risk profile."
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate document complexity score"""
        word_count = len(text.split())
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        complexity = min((avg_sentence_length - 10) / 30, 1.0)
        return max(complexity, 0.0)
    
    def _assess_power_balance(self, text: str) -> float:
        """Assess power balance between parties"""
        text_lower = text.lower()
        
        strong_language = len(re.findall(r'\b(shall|must|required|mandatory|obligated)\b', text_lower))
        weak_language = len(re.findall(r'\b(may|can|should|recommended|suggested)\b', text_lower))
        
        if strong_language + weak_language == 0:
            return 0.5
        
        balance = weak_language / (strong_language + weak_language)
        return balance

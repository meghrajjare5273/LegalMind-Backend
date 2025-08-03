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
    """Hybrid analyzer combining rule-based patterns with AI enhancement"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.gemini_client = self._initialize_gemini()
        
    def _initialize_gemini(self) -> Optional[object]:
        """Initialize Gemini client safely"""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini SDK not available")
            return None
            
        api_key = os.getenv("GEMINI_API_KEY")
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
        """Main analysis method combining rule-based and AI approaches"""
        try:
            # Step 1: Rule-based analysis
            sentences = self._split_into_sentences(text)
            rule_based_risks = self._analyze_with_rules(sentences)
            
            # Step 2: AI enhancement (if available)
            enhanced_risks = []
            for risk in rule_based_risks:
                if self.gemini_client:
                    enhanced_risk = await self._enhance_with_ai(risk)
                    enhanced_risks.append(enhanced_risk)
                else:
                    enhanced_risks.append(self._convert_to_api_format(risk))
            
            # Step 3: Generate summary and recommendations
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
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved regex"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def _analyze_with_rules(self, sentences: List[str]) -> List[RiskItem]:
        """Analyze sentences using rule-based patterns"""
        risks = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check each pattern category
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
                        break  # Avoid duplicate matches
        
        return risks
    
    @cache_result(ttl=1800)  # 30 minutes cache
    async def _enhance_with_ai(self, risk: RiskItem) -> Dict:
        """Enhance risk analysis using Gemini AI"""
        if not self.gemini_client:
            return self._convert_to_api_format(risk)
        
        try:
            prompt = self._build_enhancement_prompt(risk)
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 600}
            )
            
            ai_enhancement = self._parse_ai_response(response.text)
            return self._merge_analyses(risk, ai_enhancement)
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return self._convert_to_api_format(risk)
    
    def _build_enhancement_prompt(self, risk: RiskItem) -> str:
        """Build prompt for AI enhancement"""
        return f"""
Analyze this contract clause and enhance the risk assessment:

CLAUSE: "{risk.sentence}"
IDENTIFIED RISK: {risk.risk_category}
CURRENT LEVEL: {risk.risk_level.value}

Provide enhanced analysis in JSON format:
{{
  "enhanced_description": "More detailed explanation",
  "specific_concerns": ["concern1", "concern2"],
  "negotiation_strategies": ["strategy1", "strategy2"],
  "alternative_language": "Suggested safer wording",
  "cost_implications": "Financial impact assessment"
}}

Focus on practical, actionable insights.
""".strip()
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse AI response safely"""
        try:
            # Clean response
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```','', cleaned)
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON")
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
            "cost_implications": ai_enhancement.get("cost_implications", "")
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
            
            sections_map[section_type]["sentences"].append(risk["sentence"])
            sections_map[section_type]["count"] += 1
        
        sections = []
        for section_type, data in sections_map.items():
            # Create representative content (first 2 sentences)
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
        
        # Priority-based recommendations
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count} critical risk(s) detected - immediate legal review required")
        
        if high_count > 3:
            recommendations.append(f"⚠️ {high_count} high-risk clauses identified - negotiate these terms before signing")
        
        if total_risks > 10:
            recommendations.append("📋 Complex contract with numerous risks - consider comprehensive legal consultation")
        
        # Specific category recommendations
        risk_categories = [r["risk_category"] for r in risks]
        if "Liability" in risk_categories:
            recommendations.append("🛡️ Review liability clauses for balanced risk allocation")
        
        if "Payment" in risk_categories:
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
        # Simple complexity metrics
        word_count = len(text.split())
        avg_sentence_length = word_count / max(len(re.split(r'[.!?]', text)), 1)
        
        # Normalize to 0-1 scale
        complexity = min((avg_sentence_length - 10) / 30, 1.0)
        return max(complexity, 0.0)
    
    def _assess_power_balance(self, text: str) -> float:
        """Assess power balance between parties"""
        # Look for imbalanced language patterns
        text_lower = text.lower()
        
        strong_language = len(re.findall(r'\b(shall|must|required|mandatory|obligated)\b', text_lower))
        weak_language = len(re.findall(r'\b(may|can|should|recommended|suggested)\b', text_lower))
        
        if strong_language + weak_language == 0:
            return 0.5  # Neutral
        
        balance = weak_language / (strong_language + weak_language)
        return balance

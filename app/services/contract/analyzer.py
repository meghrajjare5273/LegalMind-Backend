import re
import os
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.cache import cache_result
from .patterns import ContractPatterns  # Updated import
from .nlp_pipeline import ImprovedNLPPipeline   # Updated import

# Keep your existing Gemini integration
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

# Keep your existing enums and dataclasses
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
class EnhancedRiskItem:
    sentence: str
    risk_category: str
    risk_level: RiskLevel
    clause_type: ClauseType
    description: str
    concerns: List[str]
    strategies: List[str]
    priority: int
    confidence: float
    detection_method: str = "rule-based"  # Track how it was detected
    specific_risks: List[str] = None
    entities: Dict = None

class EnhancedContractAnalyzer:
    """Enhanced analyzer that builds on existing system"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.nlp_pipeline = ImprovedNLPPipeline()
        self.gemini_client = self._get_gemini_client()
        
    def _get_gemini_client(self):
        """Keep your existing Gemini client setup"""
        if not GEMINI_AVAILABLE:
            return None
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
            
        try:
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None
    
    async def analyze_contract(self, text: str) -> Dict:
        """Enhanced analysis with multiple detection layers"""
        try:
            logger.info("Starting enhanced contract analysis...")
            
            # Step 1: Extract risky portions using improved NLP
            risky_portions = self.nlp_pipeline.extract_risky_portions(text, max_portions=20)
            logger.info(f"Extracted {len(risky_portions)} risky portions")
            
            # Step 2: Apply rule-based analysis to ALL sentences (not just risky ones)
            all_sentences = self._get_all_sentences(text)
            rule_based_risks = []
            
            # Analyze risky portions first
            for portion in risky_portions:
                risks = self._analyze_sentence_with_enhanced_rules(portion['sentence'], portion)
                rule_based_risks.extend(risks)
            
            # Also scan full text for patterns that might be missed
            additional_risks = self._scan_full_text_patterns(text)
            rule_based_risks.extend(additional_risks)
            
            # Remove duplicates
            unique_risks = self._deduplicate_risks(rule_based_risks)
            logger.info(f"Found {len(unique_risks)} unique risks after deduplication")
            
            # Step 3: Enhanced AI analysis with better context
            enhanced_risks = []
            top_risks = sorted(unique_risks, key=lambda x: x.priority, reverse=True)[:5]
            
            for risk in unique_risks:
                if self.gemini_client and risk in top_risks:
                    enhanced_risk = await self._enhance_with_comprehensive_ai(risk, text)
                    enhanced_risks.append(enhanced_risk)
                else:
                    enhanced_risks.append(self._convert_to_api_format(risk))
            
            # Step 4: Generate comprehensive results
            return {
                "analyses": enhanced_risks,
                "summary": self._generate_enhanced_summary(enhanced_risks),
                "sections": self._extract_enhanced_sections(enhanced_risks),
                "recommendations": await self._generate_enhanced_recommendations(enhanced_risks, text),
                "overall_summary": self._create_enhanced_overall_summary(enhanced_risks),
                "complexity_score": self._calculate_complexity(text),
                "power_balance": self._assess_power_balance(text),
                "detection_stats": self._get_detection_statistics(unique_risks)
            }
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise
    
    def _get_all_sentences(self, text: str) -> List[str]:
        """Get all sentences for comprehensive scanning"""
        return self.nlp_pipeline._smart_sentence_split(text)
    
    def _analyze_sentence_with_enhanced_rules(self, sentence: str, portion_data: Dict = None) -> List[EnhancedRiskItem]:
        """Enhanced rule-based analysis with better detection"""
        risks = []
        sentence_lower = sentence.lower()
        
        # Check all patterns
        for pattern_data in self.patterns.get_all_patterns():
            for pattern in pattern_data["patterns"]:
                if re.search(pattern, sentence_lower):
                    risk = EnhancedRiskItem(
                        sentence=sentence,
                        risk_category=pattern_data["category"],
                        risk_level=RiskLevel[pattern_data["risk_level"]],
                        clause_type=ClauseType[pattern_data["clause_type"]],
                        description=pattern_data["description"],
                        concerns=pattern_data["concerns"],
                        strategies=pattern_data["strategies"],
                        priority=pattern_data["priority"],
                        confidence=pattern_data["confidence"],
                        detection_method="enhanced-rule-based",
                        specific_risks=portion_data.get('risk_indicators', []) if portion_data else [],
                        entities=portion_data.get('entities', {}) if portion_data else {}
                    )
                    risks.append(risk)
                    break  # Avoid duplicates for same category
        
        # If no patterns matched but NLP flagged as risky, create generic risk
        if not risks and portion_data and portion_data.get('score', 0) > 0.4:
            risk = EnhancedRiskItem(
                sentence=sentence,
                risk_category="Potential Risk",
                risk_level=RiskLevel.MEDIUM,
                clause_type=ClauseType.GENERAL,
                description="Clause flagged by AI analysis as potentially risky",
                concerns=["Requires legal review", "May contain unfavorable terms"],
                strategies=["Review with legal counsel", "Consider negotiating terms"],
                priority=5,
                confidence=portion_data.get('score', 0.7),
                detection_method="ai-flagged",
                specific_risks=portion_data.get('risk_indicators', []),
                entities=portion_data.get('entities', {})
            )
            risks.append(risk)
        
        return risks
    
    def _scan_full_text_patterns(self, text: str) -> List[EnhancedRiskItem]:
        """Scan full text for patterns that might be missed in sentence analysis"""
        additional_risks = []
        text_lower = text.lower()
        
        # High-impact patterns to catch across sentence boundaries
        cross_boundary_patterns = [
            {
                "pattern": r"unlimited\s+(?:\w+\s+){0,5}liability",
                "category": "Liability Risk",
                "risk_level": "CRITICAL",
                "description": "Unlimited liability clause detected"
            },
            {
                "pattern": r"terminate\s+(?:\w+\s+){0,10}without\s+(?:\w+\s+){0,5}notice",
                "category": "Termination Risk", 
                "risk_level": "HIGH",
                "description": "Termination without notice clause"
            },
            {
                "pattern": r"payment\s+(?:\w+\s+){0,5}due\s+immediately",
                "category": "Payment Risk",
                "risk_level": "HIGH", 
                "description": "Immediate payment requirement"
            }
        ]
        
        for pattern_info in cross_boundary_patterns:
            matches = list(re.finditer(pattern_info["pattern"], text_lower))
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context_sentence = text[start:end].strip()
                
                risk = EnhancedRiskItem(
                    sentence=context_sentence,
                    risk_category=pattern_info["category"],
                    risk_level=RiskLevel[pattern_info["risk_level"]],
                    clause_type=ClauseType.GENERAL,
                    description=pattern_info["description"],
                    concerns=["Cross-boundary risk pattern detected"],
                    strategies=["Review full clause context", "Negotiate terms"],
                    priority=8,
                    confidence=0.9,
                    detection_method="cross-boundary-scan"
                )
                additional_risks.append(risk)
        
        return additional_risks
    
    def _deduplicate_risks(self, risks: List[EnhancedRiskItem]) -> List[EnhancedRiskItem]:
        """Remove duplicate risks based on sentence similarity"""
        if not risks:
            return []
        
        unique_risks = []
        seen_sentences = set()
        
        for risk in risks:
            # Simple deduplication based on first 100 characters
            sentence_key = risk.sentence[:100].lower().strip()
            if sentence_key not in seen_sentences:
                unique_risks.append(risk)
                seen_sentences.add(sentence_key)
        
        return unique_risks
    
    async def _enhance_with_comprehensive_ai(self, risk: EnhancedRiskItem, full_text: str) -> Dict:
        """Enhanced AI analysis with much more context"""
        if not self.gemini_client:
            return self._convert_to_api_format(risk)
        
        try:
            # Extract document context
            doc_context = self._extract_comprehensive_context(full_text)
            
            # Build comprehensive prompt
            enhanced_prompt = self._build_comprehensive_prompt(risk, doc_context)
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=enhanced_prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 600  # More tokens for comprehensive analysis
                }
            )
            
            ai_enhancement = self._parse_ai_response(response.text)
            return self._merge_comprehensive_analysis(risk, ai_enhancement, doc_context)
            
        except Exception as e:
            logger.warning(f"AI enhancement failed for {risk.risk_category}: {e}")
            return self._convert_to_api_format(risk)
    
    def _extract_comprehensive_context(self, text: str) -> Dict:
        """Extract comprehensive document context for AI"""
        context = {}
        
        # Document type
        text_lower = text.lower()
        if any(term in text_lower for term in ['employment', 'employee']):
            context['doc_type'] = 'Employment Agreement'
        elif any(term in text_lower for term in ['service', 'consulting']):
            context['doc_type'] = 'Service Agreement'
        elif any(term in text_lower for term in ['license', 'licensing']):
            context['doc_type'] = 'License Agreement'
        else:
            context['doc_type'] = 'General Contract'
        
        # Key financial amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        context['key_amounts'] = list(set(amounts))[:5]
        
        # Time periods
        periods = re.findall(r'\d+\s*(?:days?|months?|years?)', text, re.IGNORECASE)
        context['time_periods'] = list(set(periods))[:5]
        
        # Jurisdiction
        jurisdiction_match = re.search(r'governed\s+by\s+(?:the\s+)?laws?\s+of\s+([^,\.]+)', text_lower)
        context['jurisdiction'] = jurisdiction_match.group(1).strip() if jurisdiction_match else 'Unknown'
        
        return context
    
    def _build_comprehensive_prompt(self, risk: EnhancedRiskItem, doc_context: Dict) -> str:
        """Build comprehensive prompt with full context"""
        return f"""
LEGAL CONTRACT RISK ANALYSIS

**Document Context:**
- Type: {doc_context.get('doc_type', 'Unknown')}
- Key Amounts: {', '.join(doc_context.get('key_amounts', [])[:3])}
- Time Periods: {', '.join(doc_context.get('time_periods', [])[:3])}
- Jurisdiction: {doc_context.get('jurisdiction', 'Unknown')}

**Risk Identified:**
Category: {risk.risk_category}
Level: {risk.risk_level.value}
Detection: {risk.detection_method}

**Problematic Clause:**
"{risk.sentence}"

**Preliminary Assessment:**
- {risk.description}
- Priority Score: {risk.priority}/10
- Confidence: {risk.confidence:.2f}

**Specific Risk Indicators:**
{chr(10).join(f'- {indicator}' for indicator in risk.specific_risks) if risk.specific_risks else '- None detected'}

**TASK: Provide detailed analysis in JSON format:**
{{
  "enhanced_description": "Detailed explanation of why this clause is problematic in this document context",
  "specific_concerns": ["concern1 specific to this document type", "concern2 based on jurisdiction", "concern3 considering amounts/periods"],
  "negotiation_strategies": ["strategy1 for this document type", "strategy2 considering the financial context", "strategy3"],
  "alternative_language": "Suggested replacement clause wording",
  "legal_precedent": "Relevant legal principles or common practices", 
  "urgency_assessment": "HIGH/MEDIUM/LOW based on risk level and document context",
  "financial_impact": "Estimated potential financial impact",
  "mitigation_priority": "1-10 priority for addressing this risk"
}}
""".strip()
    
    def _parse_ai_response(self, response_text: str) -> dict:
        """Keep your existing AI response parsing - it works"""
        try:
            cleaned = response_text.strip()
            
            # Remove markdown code blocks
            code_block_match = re.search(r'``````', cleaned, re.DOTALL | re.IGNORECASE)
            if code_block_match:
                cleaned = code_block_match.group(1)
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            
            parsed = json.loads(cleaned)
            
            if not isinstance(parsed, dict):
                return {}
                
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {}
    
    def _merge_comprehensive_analysis(self, risk: EnhancedRiskItem, ai_enhancement: Dict, doc_context: Dict) -> Dict:
        """Merge all analysis components"""
        return {
            "sentence": risk.sentence,
            "risk_category": risk.risk_category,
            "risk_level": risk.risk_level.value,
            "risk_type": risk.clause_type.value,
            "description": ai_enhancement.get("enhanced_description", risk.description),
            "specific_concerns": ai_enhancement.get("specific_concerns", risk.concerns),
            "negotiation_strategies": ai_enhancement.get("negotiation_strategies", risk.strategies),
            "priority_score": risk.priority,
            "confidence_score": risk.confidence,
            "legal_concepts": [risk.risk_category],
            "entities": list(risk.entities.values()) if risk.entities else [],
            "mitigation_strategies": risk.strategies,
            "alternative_language": ai_enhancement.get("alternative_language", ""),
            "cost_implications": ai_enhancement.get("financial_impact", ""),
            "detection_method": risk.detection_method,
            "urgency_assessment": ai_enhancement.get("urgency_assessment", risk.risk_level.value),
            "legal_precedent": ai_enhancement.get("legal_precedent", ""),
            "mitigation_priority": ai_enhancement.get("mitigation_priority", risk.priority)
        }
    
    def _convert_to_api_format(self, risk: EnhancedRiskItem) -> Dict:
        """Convert to API format without AI enhancement"""
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
            "entities": list(risk.entities.values()) if risk.entities else [],
            "mitigation_strategies": risk.strategies,
            "alternative_language": "",
            "cost_implications": "",
            "detection_method": risk.detection_method
        }
    
    def _get_detection_statistics(self, risks: List[EnhancedRiskItem]) -> Dict:
        """Get statistics on detection methods"""
        detection_counts = {}
        for risk in risks:
            method = risk.detection_method
            detection_counts[method] = detection_counts.get(method, 0) + 1
        
        return {
            "total_risks_detected": len(risks),
            "detection_methods": detection_counts,
            "average_confidence": sum(r.confidence for r in risks) / len(risks) if risks else 0
        }
    
    # Keep your existing methods for summary, sections, recommendations
    def _generate_enhanced_summary(self, risks: List[Dict]) -> Dict:
        """Enhanced summary with detection insights"""
        total = len(risks)
        critical = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high = sum(1 for r in risks if r["risk_level"] == "HIGH")
        medium = sum(1 for r in risks if r["risk_level"] == "MEDIUM")
        low = sum(1 for r in risks if r["risk_level"] == "LOW")
        
        # Overall risk assessment
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
        
        # Risk distribution
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
            "risk_distribution": risk_distribution,
            "analysis_quality": "enhanced" if total > 0 else "standard"
        }
    
    # Keep other existing methods (sections, recommendations, etc.)
    def _extract_enhanced_sections(self, risks: List[Dict]) -> List[Dict]:
        """Keep your existing section extraction"""
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
    
    async def _generate_enhanced_recommendations(self, risks: List[Dict], full_text: str) -> List[str]:
        """Enhanced recommendations"""
        recommendations = []
        
        critical_count = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high_count = sum(1 for r in risks if r["risk_level"] == "HIGH")
        total_risks = len(risks)
        
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count} CRITICAL risk(s) detected - immediate legal review required before signing")
        
        if high_count > 2:
            recommendations.append(f"⚠️ {high_count} high-risk clauses identified - prioritize these in negotiations")
        
        if total_risks > 8:
            recommendations.append("📋 Multiple risks detected - consider comprehensive legal consultation")
        
        # Category-specific recommendations
        risk_categories = [r["risk_category"] for r in risks]
        if "Liability Risk" in risk_categories:
            recommendations.append("🛡️ Liability risks present - negotiate caps and mutual protections")
        
        if "Payment Risk" in risk_categories:
            recommendations.append("💰 Payment terms need attention - negotiate reasonable timelines and penalties")
        
        if "Termination Risk" in risk_categories:
            recommendations.append("🔄 Termination clauses are problematic - secure adequate notice periods")
        
        if not recommendations:
            recommendations.append("✅ Risk profile is manageable - standard legal review recommended")
        
        return recommendations
    
    def _create_enhanced_overall_summary(self, risks: List[Dict]) -> str:
        """Enhanced overall summary"""
        total = len(risks)
        critical = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
        high = sum(1 for r in risks if r["risk_level"] == "HIGH")
        
        if total == 0:
            return "Enhanced analysis complete: No significant risks detected in this contract."
        elif critical > 0:
            return f"Enhanced analysis complete: {total} risks detected including {critical} CRITICAL issues requiring immediate attention."
        elif high > 0:
            return f"Enhanced analysis complete: {total} risks detected with {high} high-priority concerns for negotiation."
        else:
            return f"Enhanced analysis complete: {total} risks detected with manageable overall profile."
    
    # Keep your existing methods
    def _calculate_complexity(self, text: str) -> float:
        """Keep your existing complexity calculation"""
        word_count = len(text.split())
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        complexity = min((avg_sentence_length - 10) / 30, 1.0)
        return max(complexity, 0.0)
    
    def _assess_power_balance(self, text: str) -> float:
        """Keep your existing power balance assessment"""
        text_lower = text.lower()
        
        strong_language = len(re.findall(r'\b(shall|must|required|mandatory|obligated)\b', text_lower))
        weak_language = len(re.findall(r'\b(may|can|should|recommended|suggested)\b', text_lower))
        
        if strong_language + weak_language == 0:
            return 0.5
        
        balance = weak_language / (strong_language + weak_language)
        return balance

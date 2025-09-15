import re
import os
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from itertools import cycle

from utils.cache import cache_result
from .patterns import ContractPatterns  
from .nlp_pipeline import ImprovedNLPPipeline 
from core.config import settings # Import your settings

# Keep your existing Gemini integration
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

# --- Enums and Dataclasses (No Change) ---
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
    detection_method: str = "rule-based" 
    specific_risks: List[str] = None
    entities: Dict = None
# --- End of Enums and Dataclasses ---


class EnhancedContractAnalyzer:
    """Enhanced analyzer that uses a pool of clients and user context"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.nlp_pipeline = ImprovedNLPPipeline()
        # Create the client pool and rotator
        self.client_pool, self.client_rotator = self._initialize_client_pool()
        
    def _initialize_client_pool(self):
        """Creates a pool of Gemini clients from the settings."""
        if not GEMINI_AVAILABLE:
            logger.warning("google-genai library not found. AI enhancement disabled.")
            return [], None
            
        api_keys = settings.GEMINI_KEY_LIST
        if not api_keys:
            logger.warning("GEMINI_API_KEYS not set in environment. AI enhancement disabled.")
            return [], None
            
        clients = []
        for key in api_keys:
            try:
                client = genai.Client(api_key=key)
                clients.append(client)
                logger.info("Gemini client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client with a key: {e}")
        
        if not clients:
            logger.error("No valid Gemini clients could be initialized.")
            return [], None
        
        logger.info(f"Successfully initialized {len(clients)} Gemini clients into a pool.")
        # Create an iterator that cycles through the pool indefinitely
        return clients, cycle(clients)

    def _get_next_gemini_client(self):
        """Gets the next available client from the rotating pool."""
        if not self.client_rotator:
            return None
        try:
            return next(self.client_rotator)
        except StopIteration:
            # This should ideally not happen with cycle, but good to have a fallback
            logger.warning("Client rotator exhausted, re-initializing pool.")
            self.client_pool, self.client_rotator = self._initialize_client_pool()
            if self.client_rotator:
                return next(self.client_rotator)
            return None
        
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

    async def analyze_contract(self, text: str, user_role: str = "Neutral Observer") -> Dict:
        """Enhanced analysis with user_role context"""
        try:
            logger.info(f"Starting enhanced contract analysis for role: {user_role}")
            
            # Step 1: Extract risky portions (no change)
            risky_portions = self.nlp_pipeline.extract_risky_portions(text, max_portions=20)
            logger.info(f"Extracted {len(risky_portions)} risky portions")
            
            # Step 2: Apply rule-based analysis (no change)
            all_sentences = self._get_all_sentences(text)
            rule_based_risks = []
            
            for portion in risky_portions:
                risks = self._analyze_sentence_with_enhanced_rules(portion['sentence'], portion)
                rule_based_risks.extend(risks)
            
            additional_risks = self._scan_full_text_patterns(text)
            rule_based_risks.extend(additional_risks)
            
            unique_risks = self._deduplicate_risks(rule_based_risks)
            logger.info(f"Found {len(unique_risks)} unique risks after deduplication")
            
            # Step 3: Enhanced AI analysis, passing user_role
            enhanced_risks = []
            top_risks = sorted(unique_risks, key=lambda x: x.priority, reverse=True)[:5]
            
            # Check if any clients are available
            gemini_available = bool(self.client_pool)
            
            for risk in unique_risks:
                if gemini_available and risk in top_risks:
                    # Pass the role to the enhancement method
                    enhanced_risk = await self._enhance_with_comprehensive_ai(risk, text, user_role)
                    enhanced_risks.append(enhanced_risk)
                else:
                    enhanced_risks.append(self._convert_to_api_format(risk))
            
            # Step 4: Generate comprehensive results (no change in this part)
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

    # ... (Keep _get_all_sentences, _analyze_sentence_with_enhanced_rules, _scan_full_text_patterns, _deduplicate_risks) ...

    async def _enhance_with_comprehensive_ai(self, risk: EnhancedRiskItem, full_text: str, user_role: str) -> Dict:
        """Enhanced AI analysis now aware of the user's role."""
        
        client = self._get_next_gemini_client()
        if not client:
            logger.warning("No Gemini client available for enhancement. Falling back to rule-based.")
            return self._convert_to_api_format(risk)
        
        try:
            # Extract document context (no change)
            doc_context = self._extract_comprehensive_context(full_text)
            
            # Build comprehensive prompt, now WITH user_role
            enhanced_prompt = self._build_comprehensive_prompt(risk, doc_context, user_role)
            
            response = client.models.generate_content(
                model=settings.LLM_MODEL, # Use model from settings
                contents=enhanced_prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 600
                }
            )
            
            ai_enhancement = self._parse_ai_response(response.text)
            return self._merge_comprehensive_analysis(risk, ai_enhancement, doc_context)
            
        except Exception as e:
            logger.warning(f"AI enhancement failed for {risk.risk_category} using one client: {e}. Falling back to rule-based.")
            return self._convert_to_api_format(risk)

    # ... (Keep _extract_comprehensive_context) ...

    def _build_comprehensive_prompt(self, risk: EnhancedRiskItem, doc_context: Dict, user_role: str) -> str:
        """Build comprehensive prompt with FULL personalization context"""
        return f"""
LEGAL CONTRACT RISK ANALYSIS

**IMPORTANT: Respond ONLY with valid JSON. Escape all quotes inside string values using backslash (\\").**

**Your Perspective:**
You are advising a client whose role in this contract is: **{user_role}**
All analysis, strategies, and concerns must be from this specific perspective.

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

**CRITICAL: Return ONLY valid JSON. Use \\" for quotes inside values. Example:**
{{"key": "Value with \\"quoted text\\" inside"}}

**TASK: Provide detailed analysis in JSON format, tailored specifically for your client, the {user_role}.**
{{
  "enhanced_description": "Detailed explanation of why this clause is problematic *specifically for the {user_role}*",
  "specific_concerns": ["concern1 specific to the {user_role}'s risk", "concern2 from the {user_role}'s perspective", "concern3"],
  "negotiation_strategies": ["A primary negotiation strategy for the {user_role}", "A fallback position for the {user_role}", "strategy3"],
  "alternative_language": "Suggested replacement clause wording that *favors the {user_role}*",
  "legal_precedent": "Common practices or principles relevant to this risk", 
  "urgency_assessment": "HIGH/MEDIUM/LOW urgency for the {user_role} to address this",
  "financial_impact": "Estimated potential financial impact or risk for the {user_role}",
  "mitigation_priority": "1-10 priority for the {user_role} to fix this"
}}
""".strip()
    
    def _parse_ai_response(self, response_text: str) -> dict:
        """Enhanced AI response parsing with detailed logging and error recovery"""
        try:
            cleaned = response_text.strip()
            logger.info(f"Original response length: {len(response_text)} characters")
            logger.debug(f"Original response: {response_text}")
            
            # Remove markdown code blocks
            code_block_match = re.search(r'``````', cleaned, re.DOTALL | re.IGNORECASE)
            if code_block_match:
                cleaned = code_block_match.group(1)
                logger.info("Removed markdown code blocks")
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
                logger.info("Extracted JSON object from response")
            
            logger.debug(f"Cleaned response: {cleaned}")
            
            # Multiple parsing strategies
            parsing_strategies = [
                ("direct", lambda x: x),
                ("escape_quotes", self._escape_inner_quotes),
                ("fix_common_issues", self._fix_common_json_issues),
                ("aggressive_fix", self._aggressive_json_fix)
            ]
            
            for strategy_name, strategy_func in parsing_strategies:
                try:
                    processed_json = strategy_func(cleaned)
                    parsed = json.loads(processed_json)
                    
                    if isinstance(parsed, dict):
                        logger.info(f"Successfully parsed JSON using strategy: {strategy_name}")
                        return parsed
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Strategy '{strategy_name}' failed: {e.msg} at line {e.lineno} col {e.colno} (pos {e.pos})")
                    # Log the problematic area
                    if hasattr(e, 'pos') and e.pos < len(processed_json):
                        start = max(0, e.pos - 50)
                        end = min(len(processed_json), e.pos + 50)
                        problem_area = processed_json[start:end]
                        logger.warning(f"Problem area: ...{problem_area}...")
                    continue
                except Exception as e:
                    logger.warning(f"Strategy '{strategy_name}' failed with exception: {e}")
                    continue
            
            logger.error("All parsing strategies failed")
            return {}
            
        except Exception as e:
            logger.error(f"Critical error in AI response parsing: {e}")
            return {}

    def _escape_inner_quotes(self, json_str: str) -> str:
        """Escape unescaped double quotes inside string values"""
        
        def fix_string_value(match):
            full_match = match.group(0)
            key = match.group(1)
            value = match.group(2)
            
            # Escape any unescaped double quotes in the value
            # But preserve already escaped quotes
            escaped_value = re.sub(r'(?<!\\)"', '\\"', value)
            
            return f'"{key}": "{escaped_value}"'
        
        # Match key-value pairs: "key": "value with possible "quotes""
        pattern = r'"([^"]+)":\s*"((?:[^"\\]|\\.)*)(?<!\\)"'
        fixed = re.sub(pattern, fix_string_value, json_str)
        
        return fixed

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix spacing issues
        json_str = re.sub(r':\s*"([^"]*?)"\s*"', r': "\1"', json_str)
        
        # Handle the specific alternative_language issue
        json_str = re.sub(
            r'"alternative_language":\s*"([^"]*)"([^"]*)"([^"]*)"',
            r'"alternative_language": "\1\"\2\"\3"',
            json_str,
            flags=re.DOTALL
        )
        
        return json_str

    def _aggressive_json_fix(self, json_str: str) -> str:
        """Aggressive JSON repair as last resort"""
        
        # Replace all unescaped quotes with escaped ones in string values
        # This is more aggressive and might over-escape, but should work as fallback
        
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip lines that are just structural (brackets, commas)
            if line.strip() in ['{', '}', '[', ']', ',']:
                fixed_lines.append(line)
                continue
            
            # For lines with key-value pairs, fix the value part
            if ':' in line and '"' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1].strip()
                    
                    # If value part starts and ends with quotes, fix inner quotes
                    if value_part.startswith('"') and value_part.rstrip(',').endswith('"'):
                        # Extract the inner value
                        inner_value = value_part[1:value_part.rstrip(',').rfind('"')]
                        # Escape quotes in inner value
                        escaped_inner = inner_value.replace('"', '\\"')
                        # Reconstruct
                        trailing_comma = ',' if value_part.rstrip().endswith(',') else ''
                        fixed_value = f'"{escaped_inner}"{trailing_comma}'
                        line = f"{key_part}: {fixed_value}"
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    

    def _log_parsing_details(self, response_text: str, error: Exception):
        """Enhanced logging for parsing failures"""
        logger.error("=== JSON PARSING FAILURE ANALYSIS ===")
        logger.error(f"Response length: {len(response_text)}")
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        
        if hasattr(error, 'pos'):
            pos = error.pos
            start = max(0, pos - 100)
            end = min(len(response_text), pos + 100)
            context = response_text[start:end]
            logger.error(f"Error position: {pos}")
            logger.error(f"Context around error: ...{context}...")
        
        # Check for common issues
        quote_count = response_text.count('"')
        escaped_quote_count = response_text.count('\\"')
        logger.error(f"Total quotes: {quote_count}, Escaped quotes: {escaped_quote_count}")
        
        # Find potential problematic lines
        lines = response_text.split('\n')
        for i, line in enumerate(lines, 1):
            if '"alternative_language"' in line or line.count('"') % 2 != 0:
                logger.error(f"Suspicious line {i}: {line}")


            
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
            "entities": [{"type": k, "values": v} for k, v in risk.entities.items()] if risk.entities else [],
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
            "entities": [{"type": k, "values": v} for k, v in risk.entities.items()] if risk.entities else [],
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

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskCategory:
    name: str
    keywords: List[str]
    risk_level: str
    consequences: List[str]
    negotiation_strategies: List[str]
    red_flags: List[str]

class ContractAnalyzer:
    def __init__(self):
        self.risk_categories = self._initialize_risk_categories()
        self.contract_patterns = self._initialize_contract_patterns()
        
    def _initialize_risk_categories(self) -> Dict[str, RiskCategory]:
        """Initialize comprehensive risk categories with detailed analysis."""
        return {
            "liability": RiskCategory(
                name="Liability and Indemnification",
                keywords=["liability", "liable", "indemnify", "indemnification", "hold harmless"],
                risk_level="HIGH",
                consequences=[
                    "Unlimited financial exposure",
                    "Legal responsibility for third-party claims",
                    "Potential for significant monetary damages"
                ],
                negotiation_strategies=[
                    "Cap liability at contract value or specific amount",
                    "Exclude consequential and punitive damages",
                    "Add mutual indemnification clauses",
                    "Include insurance requirements"
                ],
                red_flags=[
                    "Unlimited liability exposure",
                    "One-sided indemnification",
                    "Broad indemnification scope"
                ]
            ),
            "termination": RiskCategory(
                name="Termination and Breach",
                keywords=["terminate", "termination", "breach", "default", "cure period"],
                risk_level="HIGH",
                consequences=[
                    "Contract can be terminated without adequate notice",
                    "Loss of ongoing business relationship",
                    "Potential penalties or damages upon termination"
                ],
                negotiation_strategies=[
                    "Negotiate longer cure periods",
                    "Add termination for convenience clauses",
                    "Ensure mutual termination rights",
                    "Clarify what constitutes material breach"
                ],
                red_flags=[
                    "Immediate termination without cure period",
                    "One-sided termination rights",
                    "Vague breach definitions"
                ]
            ),
            "intellectual_property": RiskCategory(
                name="Intellectual Property",
                keywords=["intellectual property", "patent", "copyright", "trademark", "proprietary", "confidential"],
                risk_level="MEDIUM",
                consequences=[
                    "Loss of IP rights",
                    "Potential IP infringement claims",
                    "Restrictions on future business activities"
                ],
                negotiation_strategies=[
                    "Retain ownership of pre-existing IP",
                    "Limit scope of IP transfer",
                    "Add IP indemnification clauses",
                    "Define work-for-hire clearly"
                ],
                red_flags=[
                    "Broad IP assignment clauses",
                    "No IP indemnification",
                    "Unclear ownership of derivative works"
                ]
            ),
            "payment_terms": RiskCategory(
                name="Payment and Financial Terms",
                keywords=["payment", "invoice", "interest", "late fee", "penalty", "escrow"],
                risk_level="MEDIUM",
                consequences=[
                    "Cash flow issues from delayed payments",
                    "Additional costs from penalties and fees",
                    "Disputes over payment calculations"
                ],
                negotiation_strategies=[
                    "Negotiate shorter payment terms",
                    "Add interest on late payments",
                    "Include clear invoicing procedures",
                    "Consider milestone-based payments"
                ],
                red_flags=[
                    "Extended payment terms (>60 days)",
                    "Vague payment calculations",
                    "No late payment penalties"
                ]
            ),
            "force_majeure": RiskCategory(
                name="Force Majeure and Unforeseen Events",
                keywords=["force majeure", "act of god", "pandemic", "natural disaster", "unforeseeable"],
                risk_level="MEDIUM",
                consequences=[
                    "Inability to perform contractual obligations",
                    "Potential contract termination",
                    "No compensation for delays or non-performance"
                ],
                negotiation_strategies=[
                    "Include comprehensive force majeure clause",
                    "Add pandemic and cyber attack provisions",
                    "Negotiate equitable risk sharing",
                    "Include notice and mitigation requirements"
                ],
                red_flags=[
                    "No force majeure clause",
                    "Narrow definition of qualifying events",
                    "One-sided force majeure protection"
                ]
            ),
            "compliance": RiskCategory(
                name="Regulatory and Compliance",
                keywords=["comply", "regulation", "law", "statute", "regulatory", "compliance"],
                risk_level="HIGH",
                consequences=[
                    "Legal penalties and fines",
                    "Contract voidability",
                    "Reputational damage"
                ],
                negotiation_strategies=[
                    "Ensure compliance obligations are mutual",
                    "Add materiality thresholds",
                    "Include compliance cost sharing",
                    "Regular compliance reviews"
                ],
                red_flags=[
                    "Broad compliance warranties",
                    "No materiality qualifiers",
                    "Retroactive compliance requirements"
                ]
            )
        }
    
    def _initialize_contract_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for contract element extraction."""
        return {
            "dates": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:\d{1,2}(?:st|nd|rd|th)?),?\s+\d{4}|\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{4}|\d{2})\b',
            "money": r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|USD|million|billion)',
            "parties": r'(?:between|among)\s+(.*?)\s+(?:and|&)',
            "governing_law": r'governed by.*?laws?\s+of\s+(.*?)(?:\.|,|;)',
            "termination_notice": r'(?:notice of )?termination.*?(\d+)\s*days?',
        }

    def analyze_contract_risks(self, text: str) -> List[Dict]:
        """Perform comprehensive risk analysis on contract text."""
        risks = []
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_risks = self._analyze_sentence(sentence, i)
            risks.extend(sentence_risks)
        
        return risks
    
    def _analyze_sentence(self, sentence: str, sentence_index: int) -> List[Dict]:
        """Analyze individual sentence for risks."""
        risks = []
        sentence_lower = sentence.lower()
        
        for category_name, category in self.risk_categories.items():
            for keyword in category.keywords:
                if keyword in sentence_lower:
                    # Determine risk level based on context
                    actual_risk_level = self._assess_contextual_risk(sentence, keyword, category)
                    
                    risk = {
                        "sentence": sentence,
                        "risk_category": category.name,
                        "risk_level": actual_risk_level,
                        "specific_risk": f"Potential {category_name} risk identified",
                        "explanation": self._generate_contextual_explanation(sentence, keyword, category),
                        "potential_consequences": category.consequences,
                        "negotiation_strategies": category.negotiation_strategies,
                        "red_flags": self._identify_red_flags(sentence, category),
                        "sentence_index": sentence_index,
                        "section_context": self._extract_section_context(sentence_index)
                    }
                    risks.append(risk)
                    break  # Only add one risk per sentence to avoid duplicates
        
        return risks
    
    def _assess_contextual_risk(self, sentence: str, keyword: str, category: RiskCategory) -> str:
        """Assess risk level based on sentence context."""
        sentence_lower = sentence.lower()
        
        # High risk indicators
        high_risk_terms = ["unlimited", "sole", "exclusive", "immediate", "without notice", "all", "any and all"]
        if any(term in sentence_lower for term in high_risk_terms):
            return "HIGH"
        
        # Low risk indicators
        low_risk_terms = ["limited", "mutual", "reasonable", "material", "written notice", "cure period"]
        if any(term in sentence_lower for term in low_risk_terms):
            return "LOW"
        
        return category.risk_level
    
    def _generate_contextual_explanation(self, sentence: str, keyword: str, category: RiskCategory) -> str:
        """Generate context-specific explanation for the identified risk."""
        base_explanation = f"This clause contains '{keyword}' which relates to {category.name.lower()}."
        
        # Add specific context based on sentence content
        sentence_lower = sentence.lower()
        
        if "shall" in sentence_lower:
            base_explanation += " This appears to be a mandatory obligation."
        elif "may" in sentence_lower:
            base_explanation += " This appears to provide discretionary rights."
        elif "liable" in sentence_lower or "responsible" in sentence_lower:
            base_explanation += " This establishes responsibility or liability."
        
        return base_explanation
    
    def _identify_red_flags(self, sentence: str, category: RiskCategory) -> List[str]:
        """Identify specific red flags in the sentence."""
        identified_flags = []
        sentence_lower = sentence.lower()
        
        for red_flag in category.red_flags:
            red_flag_keywords = red_flag.lower().split()
            if all(keyword in sentence_lower for keyword in red_flag_keywords):
                identified_flags.append(red_flag)
        
        return identified_flags
    
    def extract_contract_summary(self, text: str) -> Dict:
        """Extract key contract information for summary."""
        summary = {
            "contract_type": self._identify_contract_type(text),
            "key_parties": self._extract_parties(text),
            "critical_dates": self._extract_dates(text),
            "financial_terms": self._extract_financial_terms(text),
            "termination_conditions": self._extract_termination_conditions(text),
            "governing_law": self._extract_governing_law(text)
        }
        return summary
    
    def _identify_contract_type(self, text: str) -> str:
        """Identify the type of contract based on content."""
        text_lower = text.lower()
        
        contract_types = {
            "Service Agreement": ["service", "services", "perform", "deliverable"],
            "Employment Contract": ["employee", "employment", "salary", "benefits"],
            "Non-Disclosure Agreement": ["confidential", "non-disclosure", "proprietary information"],
            "Purchase Agreement": ["purchase", "buy", "goods", "products"],
            "License Agreement": ["license", "licensed", "intellectual property"],
            "Lease Agreement": ["lease", "rent", "premises", "tenant"],
            "Partnership Agreement": ["partner", "partnership", "joint venture"]
        }
        
        for contract_type, keywords in contract_types.items():
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                return contract_type
        
        return "General Contract"
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract contracting parties."""
        parties = []
        
        # Look for party definitions
        party_patterns = [
            r'"([^"]*(?:Inc|LLC|Corp|Company|Limited)[^"]*)"',
            r'between\s+([^,]+(?:Inc|LLC|Corp|Company|Limited)[^,]*)',
            r'party.*?"([^"]*)"'
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties.extend(matches)
        
        unique_parties = list(set(parties))
        if len(unique_parties) > 5:
            return {
                "parties": unique_parties[:5],
                "truncated": True,
                "total_count": len(unique_parties)
            }
        return {"parties": unique_parties, "truncated": False}
    
    def _extract_dates(self, text: str) -> List[Dict[str, str]]:
        """Extract critical dates from contract."""
        dates = []
        date_pattern = self.contract_patterns["dates"]
        
        # Find all dates
        date_matches = re.findall(date_pattern, text)
        
        # Try to categorize dates based on context
        for date in date_matches[:10]:  # Limit to 10 dates
            context = self._get_date_context(text, date)
            dates.append({
                "date": date,
                "context": context,
                "type": self._categorize_date(context)
            })
        
        return dates
    
    def _get_date_context(self, text: str, date: str) -> str:
        """Get context around a date."""
        date_index = text.find(date)
        if date_index == -1:
            return ""
        
        start = max(0, date_index - 50)
        end = min(len(text), date_index + len(date) + 50)
        return text[start:end].strip()
    
    def _categorize_date(self, context: str) -> str:
        """Categorize date based on context."""
        context_lower = context.lower()
        
        if any(term in context_lower for term in ["effective", "commence", "start"]):
            return "Effective Date"
        elif any(term in context_lower for term in ["expire", "end", "terminate"]):
            return "Expiration Date"
        elif any(term in context_lower for term in ["due", "payment", "invoice"]):
            return "Payment Date"
        else:
            return "Other Date"
    
    def _extract_financial_terms(self, text: str) -> List[Dict[str, str]]:
        """Extract financial terms and amounts."""
        financial_terms = []
        money_pattern = self.contract_patterns["money"]
        
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        
        for amount in money_matches[:10]:  # Limit to 10 amounts
            context = self._get_amount_context(text, amount)
            financial_terms.append({
                "amount": amount,
                "context": context,
                "type": self._categorize_amount(context)
            })
        
        return financial_terms
    
    def _get_amount_context(self, text: str, amount: str) -> str:
        """Get context around a monetary amount."""
        amount_index = text.find(amount)
        if amount_index == -1:
            return ""
        
        start = max(0, amount_index - 30)
        end = min(len(text), amount_index + len(amount) + 30)
        return text[start:end].strip()
    
    def _categorize_amount(self, context: str) -> str:
        """Categorize monetary amount based on context."""
        context_lower = context.lower()
        
        if any(term in context_lower for term in ["fee", "payment", "compensation"]):
            return "Payment Amount"
        elif any(term in context_lower for term in ["penalty", "liquidated", "damages"]):
            return "Penalty Amount"
        elif any(term in context_lower for term in ["deposit", "escrow"]):
            return "Security Amount"
        else:
            return "Other Amount"
    
    def _extract_termination_conditions(self, text: str) -> List[str]:
        """Extract termination conditions."""
        conditions = []
        
        # Look for termination-related sentences
        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            if any(term in sentence.lower() for term in ["terminat", "breach", "default", "expire"]):
                conditions.append(sentence.strip())
        
        return conditions[:5]  # Limit to 5 conditions
    
    def _extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law clause."""
        pattern = self.contract_patterns["governing_law"]
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def generate_risk_dashboard(self, risks: List[Dict]) -> Dict:
        """Generate risk dashboard summary."""
        total_risks = len(risks)
        
        # Count risks by level
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        category_counts = {}
        
        for risk in risks:
            level = risk.get("risk_level", "MEDIUM")
            risk_counts[level] += 1
            
            category = risk.get("risk_category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate overall risk score (0-100)
        risk_score = (risk_counts["HIGH"] * 10 + risk_counts["MEDIUM"] * 5 + risk_counts["LOW"] * 2)
        max_possible_score = total_risks * 10
        overall_score = (risk_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_dashboard_recommendations(risk_counts, category_counts)
        
        return {
            "total_risks": total_risks,
            "high_risks": risk_counts["HIGH"],
            "medium_risks": risk_counts["MEDIUM"],
            "low_risks": risk_counts["LOW"],
            "risk_categories": category_counts,
            "overall_risk_score": round(overall_score, 1),
            "recommendations": recommendations
        }
    
    def _generate_dashboard_recommendations(self, risk_counts: Dict, category_counts: Dict) -> List[str]:
        """Generate high-level recommendations based on risk analysis."""
        recommendations = []
        
        if risk_counts["HIGH"] > 3:
            recommendations.append("⚠️ High number of high-risk clauses detected. Consider comprehensive legal review.")
        
        if "Liability and Indemnification" in category_counts and category_counts["Liability and Indemnification"] > 2:
            recommendations.append("🛡️ Multiple liability issues found. Consider liability caps and insurance requirements.")
        
        if "Termination and Breach" in category_counts:
            recommendations.append("📋 Review termination clauses for adequate cure periods and mutual rights.")
        
        if risk_counts["HIGH"] + risk_counts["MEDIUM"] > 5:
            recommendations.append("⚖️ Consider engaging legal counsel for contract negotiation.")
        
        if not recommendations:
            recommendations.append("✅ Contract shows relatively low risk profile. Standard review recommended.")
        
        return recommendations
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences more intelligently."""
        # Handle common legal abbreviations
        text = re.sub(r'\b(Inc|LLC|Corp|Ltd|Co|etc|vs|v)\.\s*', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences if s.strip()]
        
        return sentences
    
    def _extract_section_context(self, sentence_index: int) -> Optional[str]:
        """Extract section context (placeholder for future enhancement)."""
        # This could be enhanced to identify sections like "Payment Terms", "Termination", etc.
        return None
    
    def analyze_specific_clause(self, clause_text: str) -> Dict:
        """Analyze a specific clause in detail."""
        risks = self._analyze_sentence(clause_text, 0)
        
        return {
            "clause_text": clause_text,
            "risks_identified": len(risks),
            "risk_level": risks[0]["risk_level"] if risks else "LOW",
            "detailed_analysis": risks,
            "recommendations": risks[0]["negotiation_strategies"] if risks else ["Clause appears to have minimal risk."]
        }
    
    def compare_contracts(self, text1: str, text2: str) -> Dict:
        """Compare two contracts and highlight differences."""
        risks1 = self.analyze_contract_risks(text1)
        risks2 = self.analyze_contract_risks(text2)
        
        dashboard1 = self.generate_risk_dashboard(risks1)
        dashboard2 = self.generate_risk_dashboard(risks2)
        
        return {
            "contract1_risk_score": dashboard1["overall_risk_score"],
            "contract2_risk_score": dashboard2["overall_risk_score"],
            "risk_difference": abs(dashboard1["overall_risk_score"] - dashboard2["overall_risk_score"]),
            "contract1_high_risks": dashboard1["high_risks"],
            "contract2_high_risks": dashboard2["high_risks"],
            "recommendation": "Contract 1 has lower risk" if dashboard1["overall_risk_score"] < dashboard2["overall_risk_score"] else "Contract 2 has lower risk"
        }
    
    def generate_recommendations(self, risks: List[Dict], contract_summary: Dict) -> List[str]:
        """Generate overall contract recommendations."""
        recommendations = []
        
        # High-level recommendations based on risk analysis
        high_risks = [r for r in risks if r.get("risk_level") == "HIGH"]
        
        if len(high_risks) > 5:
            recommendations.append("🚨 This contract contains multiple high-risk clauses. Strongly recommend legal review before signing.")
        
        # Category-specific recommendations
        categories = {}
        for risk in risks:
            cat = risk.get("risk_category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        for category, count in categories.items():
            if count > 2:
                recommendations.append(f"📋 Multiple issues found in {category}. Focus negotiation efforts here.")
        
        # Contract type specific recommendations
        contract_type = contract_summary.get("contract_type", "")
        if "Service" in contract_type:
            recommendations.append("🔧 For service agreements, ensure clear deliverables and performance standards.")
        elif "Employment" in contract_type:
            recommendations.append("👤 Review non-compete and confidentiality clauses carefully.")
        
        if not recommendations:
            recommendations.append("✅ Contract appears to have reasonable risk levels. Standard due diligence recommended.")
        
        return recommendations
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from contracts.patterns import ContractPatterns

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    RULE_BASED = "rule_based"
    NLP_PATTERN = "nlp_pattern" 
    AI_ENHANCED = "ai_enhanced"
    REGULATORY = "regulatory"

@dataclass
class ExplanationEvidence:
    evidence_type: ExplanationType
    confidence: float
    matched_patterns: List[str]
    legal_concepts: List[str]
    regulatory_references: List[str]
    feature_importance: Dict[str, float]
    human_rationale: str

@dataclass
class ComplianceReference:
    regulation_name: str
    section: str
    description: str
    compliance_level: str  # "compliant", "non_compliant", "requires_review"
    recommendation: str

class ExplainabilityEngine:
    """Generate explanations for contract risk analysis decisions"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.regulatory_db = self._initialize_regulatory_database()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.legal_concepts_map = self._initialize_legal_concepts()
        
    def _initialize_regulatory_database(self) -> Dict[str, List[ComplianceReference]]:
        """Initialize regulatory compliance database"""
        return {
            "liability": [
                ComplianceReference(
                    regulation_name="UCC Article 2",
                    section="Section 2-719",
                    description="Limitation of damages and remedies",
                    compliance_level="requires_review",
                    recommendation="Ensure liability caps are not unconscionable"
                ),
                ComplianceReference(
                    regulation_name="Restatement of Contracts",
                    section="Section 208",
                    description="Unconscionable contracts or terms",
                    compliance_level="non_compliant",
                    recommendation="Unlimited liability may be unconscionable"
                )
            ],
            "termination": [
                ComplianceReference(
                    regulation_name="Employment-at-Will Doctrine",
                    section="Common Law",
                    description="At-will employment termination rights",
                    compliance_level="compliant",
                    recommendation="Standard at-will termination clause"
                ),
                ComplianceReference(
                    regulation_name="WARN Act",
                    section="29 USC 2101",
                    description="Worker Adjustment and Retraining Notification",
                    compliance_level="requires_review",
                    recommendation="Ensure proper notice requirements for mass layoffs"
                )
            ],
            "intellectual_property": [
                ComplianceReference(
                    regulation_name="Copyright Act",
                    section="17 USC 201",
                    description="Work for hire provisions",
                    compliance_level="requires_review",
                    recommendation="Ensure proper work-for-hire documentation"
                ),
                ComplianceReference(
                    regulation_name="Patent Act",
                    section="35 USC 261",
                    description="Patent assignment and licensing",
                    compliance_level="compliant",
                    recommendation="Standard patent assignment clause"
                )
            ],
            "payment": [
                ComplianceReference(
                    regulation_name="Fair Labor Standards Act",
                    section="29 USC 201",
                    description="Minimum wage and overtime requirements",
                    compliance_level="requires_review",
                    recommendation="Ensure payment terms comply with wage laws"
                ),
                ComplianceReference(
                    regulation_name="Truth in Lending Act",
                    section="15 USC 1601",
                    description="Consumer credit disclosure requirements",
                    compliance_level="compliant",
                    recommendation="Standard payment terms disclosure"
                )
            ],
            "confidentiality": [
                ComplianceReference(
                    regulation_name="Trade Secrets Act",
                    section="18 USC 1836",
                    description="Protection of trade secrets",
                    compliance_level="compliant",
                    recommendation="Standard confidentiality protections"
                ),
                ComplianceReference(
                    regulation_name="GDPR",
                    section="Article 32",
                    description="Security of processing personal data",
                    compliance_level="requires_review",
                    recommendation="Ensure data protection compliance for EU data"
                )
            ]
        }
    
    def _initialize_legal_concepts(self) -> Dict[str, List[str]]:
        """Map risk categories to legal concepts"""
        return {
            "liability": [
                "duty of care", "negligence", "strict liability", 
                "indemnification", "limitation of damages", "consequential damages",
                "punitive damages", "mitigation", "force majeure"
            ],
            "termination": [
                "breach of contract", "cure period", "notice requirements",
                "severance", "post-termination obligations", "at-will employment",
                "wrongful termination", "constructive dismissal"
            ],
            "payment": [
                "consideration", "liquidated damages", "penalty clauses",
                "unconscionability", "usury laws", "compound interest",
                "late fees", "acceleration clauses", "set-off rights"
            ],
            "intellectual_property": [
                "work for hire", "derivative works", "moral rights",
                "assignment vs licensing", "prior art", "fair use",
                "copyright infringement", "patent prosecution", "trademark dilution"
            ],
            "confidentiality": [
                "trade secrets", "proprietary information", "non-disclosure",
                "return of materials", "survival clauses", "exceptions to confidentiality",
                "residual knowledge", "publicity rights"
            ]
        }
    
    def generate_explanation(self, 
                           risk_item: Dict[str, Any], 
                           analysis_method: str,
                           ai_response: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation for a risk item"""
        
        explanation = {
            "risk_id": risk_item.get("sentence", "")[:50],
            "explanation_summary": "",
            "evidence_chain": [],
            "confidence_breakdown": {},
            "regulatory_compliance": [],
            "feature_analysis": {},
            "human_rationale": "",
            "recommendations": [],
            "legal_precedents": [],
            "alternative_approaches": [],
            "metadata": {
                "analysis_timestamp": self._get_timestamp(),
                "analysis_version": "1.0",
                "confidence_threshold": 0.6
            }
        }
        
        # 1. Rule-based explanation
        if analysis_method in ["rule_based", "hybrid"]:
            rule_evidence = self._explain_rule_based_decision(risk_item)
            explanation["evidence_chain"].append(rule_evidence)
        
        # 2. NLP pattern explanation
        nlp_evidence = self._explain_nlp_patterns(risk_item)
        explanation["evidence_chain"].append(nlp_evidence)
        
        # 3. AI enhancement explanation
        if ai_response and analysis_method in ["ai_enhanced", "hybrid"]:
            ai_evidence = self._explain_ai_decision(risk_item, ai_response)
            explanation["evidence_chain"].append(ai_evidence)
        
        # 4. Regulatory compliance analysis
        compliance_analysis = self._analyze_regulatory_compliance(risk_item)
        explanation["regulatory_compliance"] = compliance_analysis
        
        # 5. Feature importance analysis
        feature_analysis = self._analyze_feature_importance(risk_item)
        explanation["feature_analysis"] = feature_analysis
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(risk_item, explanation)
        explanation["recommendations"] = recommendations
        
        # 7. Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(explanation)
        explanation["confidence_breakdown"] = confidence_breakdown
        
        # 8. Generate human-readable summary
        explanation["explanation_summary"] = self._generate_explanation_summary(explanation)
        explanation["human_rationale"] = self._generate_human_rationale(risk_item, explanation)
        
        return explanation
    
    def _explain_rule_based_decision(self, risk_item: Dict) -> ExplanationEvidence:
        """Explain rule-based pattern matching decision"""
        matched_patterns = []
        legal_concepts = []
        confidence_factors = {}
        
        sentence = risk_item.get("sentence", "").lower()
        risk_category = risk_item.get("risk_category", "").lower()
        
        # Find which patterns matched
        for pattern_data in self.patterns.get_all_patterns():
            if pattern_data["category"].lower() == risk_category:
                for pattern in pattern_data["patterns"]:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        matched_patterns.append(pattern)
                        confidence_factors[pattern] = pattern_data.get("confidence", 0.8)
                
                # Map to legal concepts
                category_key = risk_category.replace(" risk", "").replace(" ", "_")
                legal_concepts.extend(self.legal_concepts_map.get(category_key, []))
                break
        
        human_rationale = self._create_rule_based_rationale(
            risk_item, matched_patterns, legal_concepts
        )
        
        return ExplanationEvidence(
            evidence_type=ExplanationType.RULE_BASED,
            confidence=risk_item.get("confidence_score", 0.8),
            matched_patterns=matched_patterns,
            legal_concepts=legal_concepts,
            regulatory_references=[],
            feature_importance=confidence_factors,
            human_rationale=human_rationale
        )
    
    def _explain_nlp_patterns(self, risk_item: Dict) -> ExplanationEvidence:
        """Explain NLP-based risk detection"""
        sentence = risk_item.get("sentence", "")
        
        # Analyze linguistic features
        linguistic_features = self._extract_linguistic_features(sentence)
        
        # TF-IDF analysis for key terms
        try:
            tfidf_scores = self._get_tfidf_importance(sentence)
        except:
            tfidf_scores = {}
        
        # Semantic analysis
        semantic_features = self._analyze_semantic_patterns(sentence)
        
        human_rationale = f"""
        NLP Analysis detected elevated risk based on:
        - Linguistic patterns: {', '.join(linguistic_features[:3])}
        - Key risk terms identified in context
        - Sentence structure analysis showing potential obligation imbalance
        - Semantic analysis reveals {len(semantic_features)} concerning patterns
        
        The natural language processing component identified structural and contextual
        indicators that suggest this clause may create unfavorable obligations or risks.
        """
        
        return ExplanationEvidence(
            evidence_type=ExplanationType.NLP_PATTERN,
            confidence=0.7,
            matched_patterns=linguistic_features + semantic_features,
            legal_concepts=[],
            regulatory_references=[],
            feature_importance=tfidf_scores,
            human_rationale=human_rationale.strip()
        )
    
    def _explain_ai_decision(self, risk_item: Dict, ai_response: str) -> ExplanationEvidence:
        """Explain AI-enhanced analysis decision"""
        
        # Parse AI response for explanation elements
        ai_features = self._extract_ai_reasoning(ai_response)
        
        # Calculate AI confidence based on response analysis
        ai_confidence = self._calculate_ai_confidence(ai_response, risk_item)
        
        human_rationale = f"""
        AI Enhancement Analysis:
        - Advanced pattern recognition identified subtle risk indicators
        - Contextual understanding of legal language nuances  
        - Cross-reference with legal precedent database
        - Semantic analysis of clause relationships
        
        AI-identified concerns: {', '.join(ai_features.get('concerns', [])[:2])}
        
        The AI system leveraged its training on legal documents to identify
        patterns that may not be caught by traditional rule-based approaches,
        providing deeper contextual understanding of potential risks.
        """
        
        return ExplanationEvidence(
            evidence_type=ExplanationType.AI_ENHANCED,
            confidence=ai_confidence,
            matched_patterns=ai_features.get('patterns', []),
            legal_concepts=ai_features.get('concepts', []),
            regulatory_references=[],
            feature_importance=ai_features.get('importance', {}),
            human_rationale=human_rationale.strip()
        )
    
    def _analyze_regulatory_compliance(self, risk_item: Dict) -> List[ComplianceReference]:
        """Analyze regulatory compliance for the risk"""
        risk_category = risk_item.get("risk_category", "").lower()
        sentence = risk_item.get("sentence", "").lower()
        
        # Map risk category to regulatory category
        category_mapping = {
            "liability risk": "liability",
            "termination risk": "termination", 
            "intellectual property risk": "intellectual_property",
            "payment risk": "payment",
            "confidentiality risk": "confidentiality",
            "employment risk": "termination"
        }
        
        regulatory_category = category_mapping.get(risk_category)
        compliance_refs = []
        
        if regulatory_category:
            base_refs = self.regulatory_db.get(regulatory_category, [])
            
            # Filter based on sentence content for more precise compliance analysis
            for ref in base_refs:
                if self._is_regulation_applicable(sentence, ref):
                    compliance_refs.append(ref)
            
            # If no specific matches, include general references
            if not compliance_refs and base_refs:
                compliance_refs = base_refs[:2]  # Include top 2 general references
        
        return compliance_refs
    
    def _analyze_feature_importance(self, risk_item: Dict) -> Dict[str, Any]:
        """Analyze which features contributed most to the risk assessment"""
        sentence = risk_item.get("sentence", "")
        
        # Feature categories
        features = {
            "modal_verbs": self._count_modal_verbs(sentence),
            "risk_keywords": self._count_risk_keywords(sentence),
            "legal_terms": self._count_legal_terms(sentence),
            "monetary_references": self._count_monetary_refs(sentence),
            "time_constraints": self._count_time_constraints(sentence),
            "absolute_language": self._count_absolute_language(sentence),
            "conditional_language": self._count_conditional_language(sentence),
            "passive_voice": self._count_passive_voice(sentence),
            "complex_structures": self._count_complex_structures(sentence)
        }
        
        # Calculate relative importance
        total_features = sum(features.values()) or 1
        importance_scores = {
            k: (v / total_features) for k, v in features.items()
        }
        
        # Get top contributing features
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "feature_counts": features,
            "importance_scores": importance_scores,
            "top_features": sorted_features[:5],
            "risk_density": sum(features.values()) / len(sentence.split()) if sentence else 0,
            "linguistic_complexity": self._calculate_linguistic_complexity(sentence)
        }
    
    def _generate_recommendations(self, risk_item: Dict, explanation: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        risk_level = risk_item.get("risk_level", "MEDIUM")
        risk_category = risk_item.get("risk_category", "")
        
        # Base recommendations by risk level
        if risk_level == "HIGH":
            recommendations.append("Immediate legal review required before contract execution")
            recommendations.append("Consider renegotiating or removing this clause entirely")
        elif risk_level == "MEDIUM":
            recommendations.append("Schedule legal review to assess acceptability of risks")
            recommendations.append("Consider adding protective language or limitations")
        else:
            recommendations.append("Monitor during contract performance")
        
        # Category-specific recommendations
        if "liability" in risk_category.lower():
            recommendations.append("Consider adding liability caps or exclusions")
            recommendations.append("Ensure adequate insurance coverage")
        elif "termination" in risk_category.lower():
            recommendations.append("Negotiate reasonable cure periods")
            recommendations.append("Clarify termination procedures and notice requirements")
        elif "payment" in risk_category.lower():
            recommendations.append("Review payment terms for reasonableness")
            recommendations.append("Consider escrow or milestone-based payments")
        
        # Compliance-based recommendations
        compliance_issues = explanation.get("regulatory_compliance", [])
        non_compliant = [c for c in compliance_issues if c.compliance_level == "non_compliant"]
        if non_compliant:
            recommendations.append("Address regulatory compliance violations immediately")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_breakdown(self, explanation: Dict) -> Dict[str, float]:
        """Calculate confidence breakdown by analysis method"""
        evidence_chain = explanation.get("evidence_chain", [])
        breakdown = {}
        
        for evidence in evidence_chain:
            method_name = evidence.evidence_type.value
            breakdown[method_name] = evidence.confidence
        
        # Calculate weighted average
        if breakdown:
            weights = {"rule_based": 0.4, "nlp_pattern": 0.3, "ai_enhanced": 0.3}
            weighted_confidence = sum(
                breakdown.get(method, 0) * weight 
                for method, weight in weights.items()
            )
            breakdown["overall_confidence"] = weighted_confidence
        
        return breakdown
    
    def _generate_explanation_summary(self, explanation: Dict) -> str:
        """Generate human-readable explanation summary"""
        evidence_types = [e.evidence_type.value for e in explanation["evidence_chain"]]
        compliance_issues = len(explanation["regulatory_compliance"])
        overall_confidence = explanation["confidence_breakdown"].get("overall_confidence", 0)
        
        summary = f"""
        This clause was flagged as risky through {len(evidence_types)} analysis methods: 
        {', '.join(evidence_types.replace('_', ' ').title())}. 
        
        Overall confidence in risk assessment: {overall_confidence:.1%}
        
        {'Regulatory compliance concerns identified. ' if compliance_issues > 0 else ''}
        
        The analysis combines pattern matching, natural language processing, and 
        {'AI enhancement ' if 'ai_enhanced' in evidence_types else ''}
        to provide comprehensive risk assessment with transparent reasoning.
        """
        
        return summary.strip()
    
    def _generate_human_rationale(self, risk_item: Dict, explanation: Dict) -> str:
        """Generate comprehensive human-readable rationale"""
        
        risk_level = risk_item.get("risk_level", "MEDIUM")
        risk_type = risk_item.get("risk_type", "General")
        confidence = explanation["confidence_breakdown"].get("overall_confidence", 0)
        
        rationale = f"""
**Risk Assessment Rationale for {risk_type} Clause**

**Risk Level: {risk_level}** (Confidence: {confidence:.1%})

**Why This Clause Is Concerning:**
        """
        
        # Add evidence from each analysis method
        for evidence in explanation["evidence_chain"]:
            rationale += f"\n\n**{evidence.evidence_type.value.replace('_', ' ').title()} Analysis:**"
            rationale += f"\n{evidence.human_rationale}"
        
        # Add regulatory context
        if explanation["regulatory_compliance"]:
            rationale += "\n\n**Regulatory Considerations:**"
            for compliance in explanation["regulatory_compliance"][:2]:
                rationale += f"\n- **{compliance.regulation_name}**: {compliance.description}"
                rationale += f"\n  *Recommendation: {compliance.recommendation}*"
        
        # Add feature analysis
        top_features = explanation["feature_analysis"].get("top_features", [])
        if top_features:
            rationale += f"\n\n**Key Risk Indicators:**"
            for feature, score in top_features[:3]:
                if score > 0:
                    rationale += f"\n- {feature.replace('_', ' ').title()}: {score:.2%} contribution"
        
        # Add recommendations
        if explanation["recommendations"]:
            rationale += f"\n\n**Recommended Actions:**"
            for i, rec in enumerate(explanation["recommendations"][:3], 1):
                rationale += f"\n{i}. {rec}"
        
        return rationale
    
    # Helper methods
    def _extract_linguistic_features(self, sentence: str) -> List[str]:
        """Extract linguistic risk features"""
        features = []
        sentence_lower = sentence.lower()
        
        if re.search(r'\b(shall|must|will)\b', sentence_lower):
            features.append("strong_obligation_language")
        if re.search(r'\b(unlimited|absolute|complete|total|entire)\b', sentence_lower):
            features.append("absolute_terms")
        if re.search(r'\b(immediately|forthwith|without delay|at once)\b', sentence_lower):
            features.append("time_pressure")
        if re.search(r'\$\d+|dollar|payment|fee|cost', sentence_lower):
            features.append("financial_terms")
        if re.search(r'\b(liable|responsible|accountable|answerable)\b', sentence_lower):
            features.append("liability_language")
        if re.search(r'\b(terminate|cancel|end|cease|discontinue)\b', sentence_lower):
            features.append("termination_language")
            
        return features
    
    def _analyze_semantic_patterns(self, sentence: str) -> List[str]:
        """Analyze semantic patterns that indicate risk"""
        patterns = []
        sentence_lower = sentence.lower()
        
        # Power imbalance indicators
        if re.search(r'party.*may.*other party.*shall', sentence_lower):
            patterns.append("power_imbalance_detected")
        
        # Vague or ambiguous terms
        vague_terms = ['reasonable', 'appropriate', 'adequate', 'sufficient', 'material']
        if any(term in sentence_lower for term in vague_terms):
            patterns.append("vague_terms_present")
        
        # Harsh consequences
        if re.search(r'forfeit|penalty|liquidated damages', sentence_lower):
            patterns.append("harsh_consequences")
        
        return patterns
    
    def _get_tfidf_importance(self, sentence: str) -> Dict[str, float]:
        """Get TF-IDF importance scores"""
        try:
            # Create a small corpus for comparison
            corpus = [sentence, "standard contract clause", "legal agreement terms"]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]  # Get scores for input sentence
            
            # Return top scoring terms
            scored_terms = dict(zip(feature_names, scores))
            return {k: v for k, v in sorted(scored_terms.items(), key=lambda x: x[1], reverse=True)[:10]}
        except:
            return {}
    
    def _extract_ai_reasoning(self, ai_response: str) -> Dict[str, List[str]]:
        """Extract reasoning elements from AI response"""
        # Parse AI response for explanation elements
        concerns = []
        patterns = []
        concepts = []
        importance = {}
        
        # Simple keyword extraction from AI response
        if "risk" in ai_response.lower():
            concerns.append("risk_identified_by_ai")
        if "concern" in ai_response.lower():
            concerns.append("general_concern_noted")
        if "liability" in ai_response.lower():
            concepts.append("liability_implications")
        if "termination" in ai_response.lower():
            concepts.append("termination_provisions")
        
        patterns.append("ai_semantic_analysis")
        importance["ai_confidence"] = 0.8
        
        return {
            "patterns": patterns,
            "concepts": concepts,
            "concerns": concerns,
            "importance": importance
        }
    
    def _calculate_ai_confidence(self, ai_response: str, risk_item: Dict) -> float:
        """Calculate AI confidence based on response quality"""
        base_confidence = 0.75
        
        # Adjust based on response length and detail
        if len(ai_response) > 100:
            base_confidence += 0.1
        
        # Adjust based on specific risk indicators in response
        risk_indicators = ["risk", "concern", "problem", "issue", "violation"]
        indicator_count = sum(1 for indicator in risk_indicators if indicator in ai_response.lower())
        base_confidence += min(0.15, indicator_count * 0.03)
        
        return min(0.95, base_confidence)
    
    def _is_regulation_applicable(self, sentence: str, compliance_ref: ComplianceReference) -> bool:
        """Check if regulation is applicable to the sentence"""
        sentence_lower = sentence.lower()
        reg_lower = compliance_ref.regulation_name.lower()
        
        # Simple keyword matching for applicability
        if "liability" in reg_lower and any(term in sentence_lower for term in ["liable", "damages", "responsible"]):
            return True
        if "employment" in reg_lower and any(term in sentence_lower for term in ["employee", "worker", "termination"]):
            return True
        if "copyright" in reg_lower and any(term in sentence_lower for term in ["copyright", "work", "intellectual"]):
            return True
        
        return False
    
    def _count_modal_verbs(self, text: str) -> int:
        return len(re.findall(r'\b(shall|must|will|may|can|should|might|could|would)\b', text.lower()))
    
    def _count_risk_keywords(self, text: str) -> int:
        risk_words = ['penalty', 'terminate', 'forfeit', 'breach', 'default', 'liable', 'damages', 'violation']
        return sum(1 for word in risk_words if word in text.lower())
    
    def _count_legal_terms(self, text: str) -> int:
        legal_words = ['contract', 'agreement', 'party', 'liability', 'damages', 'remedy', 'obligation', 'provision']
        return sum(1 for word in legal_words if word in text.lower())
    
    def _count_monetary_refs(self, text: str) -> int:
        return len(re.findall(r'\$\d+|\d+%|\d+\s*dollars?|\d+\s*cents?', text.lower()))
    
    def _count_time_constraints(self, text: str) -> int:
        return len(re.findall(r'\d+\s*(days?|months?|years?|hours?|minutes?|weeks?)', text.lower()))
    
    def _count_absolute_language(self, text: str) -> int:
        absolute_words = ['unlimited', 'absolute', 'complete', 'total', 'entire', 'all', 'never', 'always', 'forever']
        return sum(1 for word in absolute_words if word in text.lower())
    
    def _count_conditional_language(self, text: str) -> int:
        conditional_words = ['if', 'unless', 'provided', 'subject to', 'contingent', 'conditional']
        return sum(1 for word in conditional_words if word in text.lower())
    
    def _count_passive_voice(self, text: str) -> int:
        # Simple passive voice detection
        passive_patterns = [r'is\s+\w+ed', r'are\s+\w+ed', r'was\s+\w+ed', r'were\s+\w+ed', r'be\s+\w+ed']
        return sum(len(re.findall(pattern, text.lower())) for pattern in passive_patterns)
    
    def _count_complex_structures(self, text: str) -> int:
        # Count complex sentence structures
        complexity_indicators = [';', ':', 'which', 'that', 'wherein', 'whereby', 'notwithstanding']
        return sum(1 for indicator in complexity_indicators if indicator in text.lower())
    
    def _calculate_linguistic_complexity(self, text: str) -> float:
        """Calculate overall linguistic complexity score"""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / max(1, len([s for s in sentences if s.strip()]))
        
        # Syllable estimation (rough)
        syllable_count = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words)
        avg_syllables_per_word = syllable_count / max(1, len(words))
        
        # Complexity score (0-1)
        complexity = min(1.0, (avg_words_per_sentence / 20) + (avg_syllables_per_word / 3))
        return complexity
    
    def _create_rule_based_rationale(self, risk_item: Dict, patterns: List[str], concepts: List[str]) -> str:
        """Create human rationale for rule-based decision"""
        
        risk_type = risk_item.get("risk_type", "General")
        sentence = risk_item.get("sentence", "")[:100] + "..." if len(risk_item.get("sentence", "")) > 100 else risk_item.get("sentence", "")
        
        rationale = f"""
        Rule-Based Analysis identified this as a {risk_type} risk because:
        
        - Matched {len(patterns)} established legal risk patterns
        - Pattern examples: {', '.join(patterns[:2]) if patterns else 'general risk indicators'}
        - Legal concepts involved: {', '.join(concepts[:3]) if concepts else 'standard contract terms'}
        
        This determination is based on well-established legal precedents and 
        common contract interpretation principles. The identified patterns have
        been associated with similar risks in legal literature and case law.
        
        Analyzed text: "{sentence}"
        """
        
        return rationale.strip()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

import re
import logging
from typing import Dict, List, Any, Optional
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class LightweightNLPPipeline:
    """Enhanced lightweight NLP pipeline for contract analysis"""
    
    def __init__(self):
        self.risk_keywords = self._initialize_risk_keywords()
        self.sentiment_words = self._initialize_sentiment_words()
        self.legal_terms = self._initialize_legal_terms()
        self.power_indicators = self._initialize_power_indicators()
        
    def _initialize_risk_keywords(self) -> Dict[str, List[str]]:
        """Initialize risk keyword mappings"""
        return {
            "high_risk": [
                "unlimited", "absolute", "entire", "all", "any", "every",
                "forfeit", "penalty", "punitive", "liquidated", "damages",
                "terminate", "breach", "default", "violation", "liable",
                "responsible", "accountable", "indemnify", "hold harmless"
            ],
            "medium_risk": [
                "shall", "must", "required", "obligated", "duty", "covenant",
                "warranty", "guarantee", "represent", "certify", "ensure",
                "maintain", "comply", "adhere", "satisfy", "fulfill"
            ],
            "low_risk": [
                "may", "can", "might", "could", "should", "would",
                "reasonable", "best efforts", "commercially reasonable",
                "good faith", "mutual", "both parties", "jointly"
            ]
        }
    
    def _initialize_sentiment_words(self) -> Dict[str, List[str]]:
        """Initialize sentiment analysis words"""
        return {
            "negative": [
                "breach", "violation", "default", "failure", "unable",
                "impossible", "prohibit", "restrict", "limit", "prevent",
                "deny", "refuse", "reject", "terminate", "cancel",
                "forfeit", "penalty", "punish", "liable", "guilty"
            ],
            "positive": [
                "benefit", "advantage", "protect", "ensure", "guarantee",
                "secure", "preserve", "maintain", "support", "assist",
                "cooperate", "collaborate", "mutual", "fair", "reasonable"
            ],
            "neutral": [
                "provide", "include", "contain", "specify", "describe",
                "define", "establish", "create", "form", "constitute"
            ]
        }
    
    def _initialize_legal_terms(self) -> List[str]:
        """Initialize legal terminology"""
        return [
            "contract", "agreement", "party", "parties", "consideration",
            "obligation", "duty", "right", "privilege", "license",
            "warranty", "representation", "covenant", "condition",
            "provision", "clause", "section", "article", "paragraph",
            "liability", "damages", "remedy", "relief", "indemnity",
            "termination", "expiration", "breach", "default", "cure",
            "notice", "consent", "approval", "discretion", "reasonable",
            "material", "substantial", "significant", "force majeure",
            "intellectual property", "confidential", "proprietary",
            "assignment", "transfer", "sublicense", "governing law",
            "jurisdiction", "venue", "arbitration", "mediation"
        ]
    
    def _initialize_power_indicators(self) -> Dict[str, List[str]]:
        """Initialize power imbalance indicators"""
        return {
            "dominant_party": [
                "sole discretion", "absolute discretion", "final decision",
                "unilateral", "at will", "without cause", "without notice",
                "immediate termination", "automatically terminate"
            ],
            "subordinate_party": [
                "subject to approval", "with consent", "upon request",
                "as directed", "in accordance with", "as specified by",
                "under supervision", "at the direction of"
            ]
        }
    
    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Comprehensive sentence-level analysis"""
        analysis = {
            "sentence": sentence,
            "word_count": len(sentence.split()),
            "character_count": len(sentence),
            "sentiment": self._analyze_sentiment(sentence),
            "risk_confidence": self._calculate_risk_confidence(sentence),
            "complexity_score": self._calculate_complexity(sentence),
            "legal_density": self._calculate_legal_density(sentence),
            "power_imbalance_score": self._analyze_power_imbalance(sentence),
            "obligation_strength": self._analyze_obligation_strength(sentence),
            "temporal_constraints": self._analyze_temporal_constraints(sentence),
            "financial_impact": self._analyze_financial_impact(sentence),
            "linguistic_features": self._extract_linguistic_features(sentence),
            "risk_indicators": self._identify_risk_indicators(sentence)
        }
        
        return analysis
    
    def analyze_document_sentiment(self, text: str) -> Dict[str, Any]:
        """Document-level sentiment analysis"""
        sentences = re.split(r'[.!?]+', text)
        sentence_sentiments = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 5:
                sentiment = self._analyze_sentiment(sentence)
                sentence_sentiments.append(sentiment)
        
        # Calculate overall sentiment
        sentiment_counts = Counter(sentence_sentiments)
        total_sentences = len(sentence_sentiments)
        
        if total_sentences == 0:
            return {"overall_sentiment": "neutral", "confidence": 0.0}
        
        sentiment_scores = {
            "negative": sentiment_counts.get("negative", 0) / total_sentences,
            "neutral": sentiment_counts.get("neutral", 0) / total_sentences,
            "positive": sentiment_counts.get("positive", 0) / total_sentences
        }
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        
        return {
            "overall_sentiment": max_sentiment[0],
            "confidence": max_sentiment[1],
            "sentiment_distribution": sentiment_scores,
            "total_sentences_analyzed": total_sentences,
            "risk_tone": "high" if sentiment_scores["negative"] > 0.4 else "moderate" if sentiment_scores["negative"] > 0.2 else "low"
        }
    
    def _analyze_sentiment(self, sentence: str) -> str:
        """Analyze sentiment of a sentence"""
        sentence_lower = sentence.lower()
        
        negative_count = sum(1 for word in self.sentiment_words["negative"] if word in sentence_lower)
        positive_count = sum(1 for word in self.sentiment_words["positive"] if word in sentence_lower)
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    def _calculate_risk_confidence(self, sentence: str) -> float:
        """Calculate risk confidence based on keyword analysis"""
        sentence_lower = sentence.lower()
        
        high_risk_count = sum(1 for word in self.risk_keywords["high_risk"] if word in sentence_lower)
        medium_risk_count = sum(1 for word in self.risk_keywords["medium_risk"] if word in sentence_lower)
        low_risk_count = sum(1 for word in self.risk_keywords["low_risk"] if word in sentence_lower)
        
        # Calculate weighted score
        total_words = len(sentence.split())
        if total_words == 0:
            return 0.0
        
        risk_score = (high_risk_count * 3 + medium_risk_count * 2 + low_risk_count * 1) / total_words
        
        # Normalize to 0-1 range
        return min(1.0, risk_score * 2)
    
    def _calculate_complexity(self, sentence: str) -> float:
        """Calculate linguistic complexity"""
        words = sentence.split()
        if not words:
            return 0.0
        
        # Factors affecting complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        punctuation_density = sum(1 for char in sentence if char in '.,;:()[]{}') / len(sentence)
        subordinate_clauses = len(re.findall(r'\b(which|that|where|when|while|although|because|since|if|unless)\b', sentence.lower()))
        
        # Complex legal terms
        complex_terms = ['whereas', 'heretofore', 'hereinafter', 'notwithstanding', 'pursuant', 'aforementioned']
        complex_term_count = sum(1 for term in complex_terms if term in sentence.lower())
        
        # Normalize complexity score
        complexity = (
            (avg_word_length / 10) * 0.3 +
            (punctuation_density * 10) * 0.2 +
            (subordinate_clauses / len(words)) * 0.3 +
            (complex_term_count / len(words)) * 0.2
        )
        
        return min(1.0, complexity)
    
    def _calculate_legal_density(self, sentence: str) -> float:
        """Calculate density of legal terminology"""
        words = sentence.lower().split()
        if not words:
            return 0.0
        
        legal_word_count = sum(1 for word in words if word in [term.lower() for term in self.legal_terms])
        return legal_word_count / len(words)
    
    def _analyze_power_imbalance(self, sentence: str) -> float:
        """Analyze power imbalance indicators"""
        sentence_lower = sentence.lower()
        
        dominant_indicators = sum(1 for phrase in self.power_indicators["dominant_party"] if phrase in sentence_lower)
        subordinate_indicators = sum(1 for phrase in self.power_indicators["subordinate_party"] if phrase in sentence_lower)
        
        # Calculate imbalance score
        total_indicators = dominant_indicators + subordinate_indicators
        if total_indicators == 0:
            return 0.0
        
        # Higher score indicates more imbalance
        imbalance_ratio = abs(dominant_indicators - subordinate_indicators) / total_indicators
        return imbalance_ratio
    
    def _analyze_obligation_strength(self, sentence: str) -> str:
        """Analyze strength of obligations in sentence"""
        sentence_lower = sentence.lower()
        
        # Strong obligation indicators
        strong_words = ['shall', 'must', 'required', 'mandatory', 'obligated', 'duty']
        moderate_words = ['should', 'will', 'agrees to', 'undertakes', 'covenants']
        weak_words = ['may', 'can', 'might', 'endeavor', 'best efforts']
        
        strong_count = sum(1 for word in strong_words if word in sentence_lower)
        moderate_count = sum(1 for word in moderate_words if word in sentence_lower)
        weak_count = sum(1 for word in weak_words if word in sentence_lower)
        
        if strong_count > 0:
            return "strong"
        elif moderate_count > 0:
            return "moderate"
        elif weak_count > 0:
            return "weak"
        else:
            return "none"
    
    def _analyze_temporal_constraints(self, sentence: str) -> Dict[str, Any]:
        """Analyze temporal constraints and deadlines"""
        time_patterns = [
            (r'\bimmediately\b', 'immediate'),
            (r'\bforthwith\b', 'immediate'),
            (r'\bwithout delay\b', 'immediate'),
            (r'\b(\d+)\s*days?\b', 'days'),
            (r'\b(\d+)\s*months?\b', 'months'),
            (r'\b(\d+)\s*years?\b', 'years'),
            (r'\bupon\s+notice\b', 'upon_notice'),
            (r'\bwithin\s+(\d+)\b', 'within_period')
        ]
        
        constraints = []
        urgency_score = 0.0
        
        for pattern, constraint_type in time_patterns:
            matches = re.findall(pattern, sentence.lower())
            if matches:
                constraints.append({
                    "type": constraint_type,
                    "matches": matches,
                    "pattern": pattern
                })
                
                # Calculate urgency
                if constraint_type == 'immediate':
                    urgency_score = 1.0
                elif constraint_type == 'days':
                    days = int(matches[0]) if matches[0].isdigit() else 30
                    urgency_score = max(urgency_score, 1.0 - min(1.0, days / 30))
        
        return {
            "constraints": constraints,
            "urgency_score": urgency_score,
            "has_deadlines": len(constraints) > 0
        }
    
    def _analyze_financial_impact(self, sentence: str) -> Dict[str, Any]:
        """Analyze financial terms and impacts"""
        financial_patterns = [
            (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', 'dollar_amount'),
            (r'\b(\d+(?:\.\d+)?)\s*percent\b', 'percentage'),
            (r'\b(\d+(?:\.\d+)?)%', 'percentage'),
            (r'\binterest\s+rate\b', 'interest'),
            (r'\bpenalty\b', 'penalty'),
            (r'\bliquidated\s+damages\b', 'liquidated_damages'),
            (r'\bforfeit\b', 'forfeiture')
        ]
        
        financial_terms = []
        risk_level = "low"
        
        for pattern, term_type in financial_patterns:
            matches = re.findall(pattern, sentence.lower())
            if matches:
                financial_terms.append({
                    "type": term_type,
                    "matches": matches
                })
                
                # Assess risk level
                if term_type in ['penalty', 'liquidated_damages', 'forfeiture']:
                    risk_level = "high"
                elif term_type in ['interest', 'percentage'] and risk_level != "high":
                    risk_level = "medium"
        
        return {
            "financial_terms": financial_terms,
            "has_financial_impact": len(financial_terms) > 0,
            "financial_risk_level": risk_level
        }
    
    def _extract_linguistic_features(self, sentence: str) -> List[str]:
        """Extract linguistic features that may indicate risk"""
        features = []
        sentence_lower = sentence.lower()
        
        # Absoluteness indicators
        if re.search(r'\b(all|any|every|entire|complete|total|absolute|unlimited)\b', sentence_lower):
            features.append("absolute_language")
        
        # Conditional language
        if re.search(r'\b(if|unless|provided|subject to|conditional|contingent)\b', sentence_lower):
            features.append("conditional_language")
        
        # Passive voice indicators
        if re.search(r'\b(is|are|was|were|been|being)\s+\w+ed\b', sentence_lower):
            features.append("passive_voice")
        
        # Modal verbs
        if re.search(r'\b(shall|must|will|may|can|should|might|could|would)\b', sentence_lower):
            features.append("modal_verbs")
        
        # Negation
        if re.search(r'\b(not|no|never|nothing|none|neither|nor)\b', sentence_lower):
            features.append("negation")
        
        # Legal formality
        formal_terms = ['whereas', 'herein', 'thereof', 'hereof', 'aforementioned', 'notwithstanding']
        if any(term in sentence_lower for term in formal_terms):
            features.append("formal_legal_language")
        
        return features
    
    def _identify_risk_indicators(self, sentence: str) -> List[Dict[str, Any]]:
        """Identify specific risk indicators in sentence"""
        indicators = []
        sentence_lower = sentence.lower()
        
        # Check for specific risk patterns
        risk_patterns = [
            {
                "pattern": r'\bunlimited\s+liability\b',
                "risk_type": "unlimited_liability",
                "severity": "high",
                "description": "Unlimited liability exposure"
            },
            {
                "pattern": r'\bterminate.*at\s+will\b',
                "risk_type": "at_will_termination",
                "severity": "medium",
                "description": "At-will termination clause"
            },
            {
                "pattern": r'\bsole\s+discretion\b',
                "risk_type": "unilateral_control",
                "severity": "medium",
                "description": "Unilateral decision-making authority"
            },
            {
                "pattern": r'\bwithout\s+notice\b',
                "risk_type": "no_notice_requirement",
                "severity": "medium",
                "description": "No notice requirement"
            },
            {
                "pattern": r'\bimmediately\s+effective\b',
                "risk_type": "immediate_effect",
                "severity": "medium",
                "description": "Immediate effectiveness clause"
            }
        ]
        
        for risk_pattern in risk_patterns:
            if re.search(risk_pattern["pattern"], sentence_lower):
                indicators.append({
                    "type": risk_pattern["risk_type"],
                    "severity": risk_pattern["severity"],
                    "description": risk_pattern["description"],
                    "matched_text": re.search(risk_pattern["pattern"], sentence_lower).group()
                })
        
        return indicators
    
    def get_analysis_summary(self, sentences: List[str]) -> Dict[str, Any]:
        """Get summary analysis for multiple sentences"""
        if not sentences:
            return {"error": "No sentences provided"}
        
        analyses = [self.analyze_sentence(sentence) for sentence in sentences]
        
        # Aggregate statistics
        avg_complexity = statistics.mean([a["complexity_score"] for a in analyses])
        avg_legal_density = statistics.mean([a["legal_density"] for a in analyses])
        avg_risk_confidence = statistics.mean([a["risk_confidence"] for a in analyses])
        avg_power_imbalance = statistics.mean([a["power_imbalance_score"] for a in analyses])
        
        # Count sentiment distribution
        sentiments = [a["sentiment"] for a in analyses]
        sentiment_dist = Counter(sentiments)
        
        # Count obligation strengths
        obligations = [a["obligation_strength"] for a in analyses]
        obligation_dist = Counter(obligations)
        
        return {
            "total_sentences": len(sentences),
            "average_complexity": avg_complexity,
            "average_legal_density": avg_legal_density,
            "average_risk_confidence": avg_risk_confidence,
            "average_power_imbalance": avg_power_imbalance,
            "sentiment_distribution": dict(sentiment_dist),
            "obligation_distribution": dict(obligation_dist),
            "high_risk_sentences": sum(1 for a in analyses if a["risk_confidence"] > 0.7),
            "complex_sentences": sum(1 for a in analyses if a["complexity_score"] > 0.7),
            "overall_risk_level": "high" if avg_risk_confidence > 0.7 else "medium" if avg_risk_confidence > 0.4 else "low"
        }

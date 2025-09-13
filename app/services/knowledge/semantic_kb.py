import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskPattern:
    pattern_text: str
    risk_category: str
    risk_level: str
    clause_type: str
    description: str
    concerns: List[str]
    strategies: List[str]
    priority: int
    confidence: float
    embedding: Optional[np.ndarray] = None

class SemanticKnowledgeBase:
    """Semantic knowledge base for contract risk patterns"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.risk_patterns: List[RiskPattern] = []
        self.pattern_embeddings = None
        self.similarity_threshold = 0.75
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize with semantic risk patterns"""
        risk_examples = [
            {
                "pattern_text": "The company may terminate this agreement immediately without any notice for any reason or no reason at all",
                "risk_category": "Termination Risk",
                "clause_type": "TERMINATION",
                "risk_level": "CRITICAL",
                "description": "Allows arbitrary termination without notice or justification",
                "concerns": ["Sudden contract termination", "No protection against arbitrary decisions", "Business continuity risk"],
                "strategies": ["Negotiate minimum notice period", "Add termination for cause only", "Include cure period"],
                "priority": 9,
                "confidence": 0.95
            },
            {
                "pattern_text": "Contractor shall indemnify and hold harmless the company from any and all claims, damages, losses, and expenses of any nature whatsoever",
                "risk_category": "Liability Risk", 
                "clause_type": "LIABILITY",
                "risk_level": "CRITICAL",
                "description": "Unlimited indemnification with broad scope",
                "concerns": ["Unlimited financial exposure", "Covers all possible claims", "No reciprocal protection"],
                "strategies": ["Cap indemnification amount", "Limit to specific claim types", "Add mutual indemnification"],
                "priority": 10,
                "confidence": 0.98
            },
            {
                "pattern_text": "All intellectual property created during the term shall automatically transfer to and vest in the company",
                "risk_category": "Intellectual Property Risk",
                "clause_type": "INTELLECTUAL_PROPERTY", 
                "risk_level": "HIGH",
                "description": "Automatic transfer of all IP without compensation",
                "concerns": ["Loss of IP ownership", "No compensation for valuable IP", "Overly broad assignment"],
                "strategies": ["Limit to work-specific IP", "Retain pre-existing IP rights", "Negotiate IP compensation"],
                "priority": 8,
                "confidence": 0.92
            },
            {
                "pattern_text": "Payment is due immediately upon invoice with a 5% penalty per day for late payment",
                "risk_category": "Payment Risk",
                "clause_type": "PAYMENT",
                "risk_level": "HIGH", 
                "description": "Immediate payment requirement with excessive penalties",
                "concerns": ["No grace period", "Compounding daily penalties", "Cash flow pressure"],
                "strategies": ["Negotiate net-30 terms", "Reduce penalty rate", "Add grace period"],
                "priority": 7,
                "confidence": 0.88
            },
            {
                "pattern_text": "This confidentiality obligation shall survive termination and continue in perpetuity",
                "risk_category": "Confidentiality Risk",
                "clause_type": "CONFIDENTIALITY",
                "risk_level": "MEDIUM",
                "description": "Perpetual confidentiality requirements",
                "concerns": ["Indefinite obligation", "Unclear scope", "Enforcement challenges"],
                "strategies": ["Limit to 5-7 years", "Define confidential information clearly", "Add standard exceptions"],
                "priority": 5,
                "confidence": 0.82
            }
        ]
        
        # Convert to RiskPattern objects
        for example in risk_examples:
            pattern = RiskPattern(**example)
            self.risk_patterns.append(pattern)
        
        # Generate embeddings for all patterns
        self._generate_embeddings()
        logger.info(f"Initialized semantic knowledge base with {len(self.risk_patterns)} patterns")
    
    def _generate_embeddings(self):
        """Generate embeddings for all risk patterns"""
        pattern_texts = [pattern.pattern_text for pattern in self.risk_patterns]
        embeddings = self.model.encode(pattern_texts, convert_to_numpy=True)
        
        # Store embeddings in patterns
        for i, pattern in enumerate(self.risk_patterns):
            pattern.embedding = embeddings[i]
        
        self.pattern_embeddings = embeddings
        
    def find_similar_risks(self, text: str, top_k: int = 3) -> List[Tuple[RiskPattern, float]]:
        """Find semantically similar risk patterns"""
        if not self.pattern_embeddings is not None:
            return []
        
        # Generate embedding for input text
        query_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self.pattern_embeddings, query_embedding) / (
            np.linalg.norm(self.pattern_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top matches above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= self.similarity_threshold:
                results.append((self.risk_patterns[idx], similarity_score))
        
        return results
    
    def get_contextual_examples(self, risk_category: str, limit: int = 2) -> List[RiskPattern]:
        """Get examples for a specific risk category"""
        return [
            pattern for pattern in self.risk_patterns 
            if pattern.risk_category == risk_category
        ][:limit]

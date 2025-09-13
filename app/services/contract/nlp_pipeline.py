import re
import nltk
import logging
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

class LightweightNLPPipeline:
    """Lightweight NLP pipeline for contract analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.legal_keywords = self._get_legal_keywords()
        self.risk_indicators = self._get_risk_indicators()
        # Removed TfidfVectorizer initialization to reduce memory usage
        
    def _get_legal_keywords(self) -> List[str]:
        """Legal domain keywords for relevance scoring"""
        return [
            'contract', 'agreement', 'party', 'parties', 'liability', 'damages',
            'termination', 'breach', 'default', 'remedy', 'indemnification',
            'confidentiality', 'intellectual property', 'copyright', 'trademark',
            'patent', 'license', 'assignment', 'transfer', 'waiver', 'jurisdiction',
            'governing law', 'arbitration', 'dispute', 'resolution', 'penalty',
            'fee', 'payment', 'compensation', 'refund', 'force majeure'
        ]
    
    def _get_risk_indicators(self) -> Dict[str, List[str]]:
        """Risk indicator keywords by severity"""
        return {
            'critical': ['unlimited', 'absolute', 'complete', 'total', 'entire', 'all'],
            'high': ['exclusively', 'solely', 'perpetual', 'irrevocable', 'immediate'],
            'medium': ['substantial', 'significant', 'material', 'reasonable']
        }
    
    def extract_risky_portions(self, text: str, max_portions: int = 5) -> List[Dict]:
        """Extract only the riskiest portions of the contract"""
        
        # Split into sentences
        sentences = self._smart_sentence_split(text)
        
        # Score each sentence for risk
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._calculate_risk_score(sentence)
            if score > 0.3:  # Only consider sentences with meaningful risk
                sentence_scores.append({
                    'sentence': sentence,
                    'score': score,
                    'index': i,
                    'key_phrases': self._extract_key_phrases(sentence)
                })
        
        # Sort by risk score and return top portions
        sentence_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return sentence_scores[:max_portions]
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """Improved sentence splitting for legal documents"""
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Use NLTK for better sentence boundary detection
        sentences = sent_tokenize(text)
        
        # Filter out very short or very long sentences
        filtered_sentences = []
        for sentence in sentences:
            length = len(sentence.split())
            if 5 <= length <= 100:  # Reasonable sentence length
                filtered_sentences.append(sentence.strip())
        
        return filtered_sentences
    
    def _calculate_risk_score(self, sentence: str) -> float:
        """Calculate risk score for a sentence"""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Check for risk indicator keywords
        for risk_level, keywords in self.risk_indicators.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    if risk_level == 'critical':
                        score += 0.4
                    elif risk_level == 'high':
                        score += 0.3
                    elif risk_level == 'medium':
                        score += 0.2
        
        # Check for legal keywords (domain relevance)
        legal_keyword_count = sum(1 for keyword in self.legal_keywords 
                                if keyword in sentence_lower)
        score += min(legal_keyword_count * 0.1, 0.3)
        
        # Check for numerical values (often indicate penalties/payments)
        if re.search(r'\$\d+|\d+%|\d+\s*(days|months|years)', sentence):
            score += 0.2
        
        # Check for modal verbs indicating obligation
        modal_verbs = ['shall', 'must', 'will', 'may not', 'cannot']
        modal_count = sum(1 for modal in modal_verbs if modal in sentence_lower)
        score += min(modal_count * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _extract_key_phrases(self, sentence: str) -> List[str]:
        """Extract key phrases from a sentence"""
        # Tokenize and remove stopwords
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Get most frequent meaningful words
        word_freq = Counter(words)
        key_words = [word for word, freq in word_freq.most_common(5)]
        
        # Extract phrases with legal significance
        legal_phrases = []
        sentence_lower = sentence.lower()
        
        phrase_patterns = [
            r'terminate.*without.*(?:notice|cause)',
            r'unlimited.*liability',
            r'intellectual.*property',
            r'governing.*law',
            r'arbitration.*in',
            r'confidential.*information',
            r'payment.*due',
            r'penalty.*of'
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, sentence_lower)
            legal_phrases.extend(matches)
        
        return key_words + legal_phrases
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities using pattern matching"""
        entities = {
            'monetary_amounts': re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text),
            'percentages': re.findall(r'\d+(?:\.\d+)?%', text),
            'time_periods': re.findall(r'\d+\s*(?:days?|months?|years?)', text),
            'dates': re.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', text),
            'parties': re.findall(r'(?:company|corporation|llc|inc|ltd)\.?', text, re.I)
        }
        
        return {k: list(set(v)) for k, v in entities.items() if v}

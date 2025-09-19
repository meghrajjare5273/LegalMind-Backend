import re
import nltk
import logging
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# # Keep your existing imports
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)

# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
logger = logging.getLogger(__name__)

def setup_nltk_data():
    """Setup NLTK data with proper error handling for production deployment"""
    
    # Set NLTK data path to a persistent location
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Add the custom path to NLTK's search paths
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    # Download required NLTK data
    required_data = [
        ('punkt_tab', 'tokenizers/punkt_tab'),  # Updated tokenizer
        ('stopwords', 'corpora/stopwords')
    ]
    
    for data_name, data_path in required_data:
        try:
            nltk.data.find(data_path)
            logger.info(f"NLTK data '{data_name}' already available")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK data: {data_name}")
                nltk.download(data_name, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Successfully downloaded NLTK data: {data_name}")
            except Exception as e:
                logger.error(f"Failed to download NLTK data '{data_name}': {str(e)}")
                # Fallback: try without specifying download directory
                try:
                    nltk.download(data_name, quiet=True)
                    logger.info(f"Downloaded NLTK data '{data_name}' to default location")
                except Exception as fallback_error:
                    logger.error(f"Complete failure downloading '{data_name}': {str(fallback_error)}")

# Call this function during module initialization
setup_nltk_data()


# Replace the existing try/except block with:
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError as e:
    logger.error(f"Failed to import NLTK components: {str(e)}")
    raise


class ImprovedNLPPipeline:
    """Improved NLP pipeline that enhances rather than replaces existing functionality"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.legal_keywords = self._get_enhanced_legal_keywords()
        self.risk_indicators = self._get_enhanced_risk_indicators()
        self.risk_threshold = 0.25  # Much lower threshold
        
    def _get_enhanced_legal_keywords(self) -> List[str]:
        """Enhanced legal keywords with more comprehensive coverage"""
        return [
            # Core contract terms (keep yours + add more)
            'contract', 'agreement', 'party', 'parties', 'liability', 'damages',
            'termination', 'breach', 'default', 'remedy', 'indemnification',
            'confidentiality', 'intellectual property', 'copyright', 'trademark',
            'patent', 'license', 'assignment', 'transfer', 'waiver', 'jurisdiction',
            'governing law', 'arbitration', 'dispute', 'resolution', 'penalty',
            'fee', 'payment', 'compensation', 'refund', 'force majeure',
            
            # Risk-specific terms
            'unlimited', 'absolute', 'perpetual', 'irrevocable', 'exclusive',
            'sole', 'entire', 'complete', 'immediate', 'forthwith', 'without notice',
            'at will', 'discretion', 'indemnify', 'hold harmless', 'defend',
            
            # Financial terms
            'penalty', 'interest', 'late fee', 'collection costs', 'attorney fees',
            'non-refundable', 'upfront', 'advance payment', 'liquidated damages',
            
            # Time-sensitive terms
            'immediately', 'forthwith', 'without delay', 'business days', 'calendar days'
        ]
    
    def _get_enhanced_risk_indicators(self) -> Dict[str, List[str]]:
        """Enhanced risk indicators with lower barriers to detection"""
        return {
            'critical': [
                'unlimited', 'absolute', 'complete', 'total', 'entire', 'all damages',
                'without limitation', 'unlimited liability', 'unlimited indemnity'
            ],
            'high': [
                'exclusively', 'solely', 'perpetual', 'irrevocable', 'immediate',
                'without notice', 'at will', 'sole discretion', 'non-refundable',
                'no refund', 'penalty', 'late fee'
            ],
            'medium': [
                'substantial', 'significant', 'material', 'reasonable',
                'confidential', 'proprietary', 'governing law', 'jurisdiction'
            ]
        }
    
    def extract_risky_portions(self, text: str, max_portions: int = 15) -> List[Dict]:
        """Enhanced extraction with multiple detection methods"""
        
        # Split into sentences (keep your existing method)
        sentences = self._smart_sentence_split(text)
        logger.info(f"Analyzing {len(sentences)} sentences")
        
        # Score sentences using multiple methods
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Use multiple scoring methods
            rule_score = self._calculate_rule_based_score(sentence)
            context_score = self._calculate_contextual_score(sentence)
            entity_score = self._calculate_entity_score(sentence)
            
            # Combined score (weighted average)
            combined_score = (rule_score * 0.5 + context_score * 0.3 + entity_score * 0.2)
            
            # Lower threshold for inclusion
            if combined_score > self.risk_threshold:
                sentence_scores.append({
                    'sentence': sentence,
                    'score': combined_score,
                    'rule_score': rule_score,
                    'context_score': context_score,
                    'entity_score': entity_score,
                    'index': i,
                    'key_phrases': self._extract_key_phrases(sentence),
                    'risk_indicators': self._identify_specific_risks(sentence),
                    'entities': self._extract_sentence_entities(sentence)
                })
        
        # Sort by score and return
        sentence_scores.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Found {len(sentence_scores)} risky sentences above threshold {self.risk_threshold}")
        
        return sentence_scores[:max_portions]
    
    def _calculate_rule_based_score(self, sentence: str) -> float:
        """Calculate score based on rule patterns (similar to your existing)"""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Risk indicator keywords (keep your logic but make more sensitive)
        for risk_level, keywords in self.risk_indicators.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    if risk_level == 'critical':
                        score += 0.6  # Higher scores for detection
                    elif risk_level == 'high':
                        score += 0.4
                    elif risk_level == 'medium':
                        score += 0.2
        
        # Legal keywords (domain relevance)
        legal_keyword_count = sum(1 for keyword in self.legal_keywords 
                                if keyword in sentence_lower)
        score += min(legal_keyword_count * 0.08, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_contextual_score(self, sentence: str) -> float:
        """Calculate score based on contextual patterns"""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Numerical values (penalties/payments/timeframes)
        if re.search(r'\$[\d,]+', sentence):
            score += 0.3
        if re.search(r'\d+%', sentence):
            score += 0.2
        if re.search(r'\d+\s*(?:days?|hours?|months?)', sentence):
            score += 0.2
        
        # Modal verbs indicating obligation
        obligation_patterns = [
            r'\bshall\b', r'\bmust\b', r'\bwill\b', r'\bmay not\b', 
            r'\bcannot\b', r'\bprohibited\b', r'\brequired\b'
        ]
        for pattern in obligation_patterns:
            if re.search(pattern, sentence_lower):
                score += 0.15
        
        # Absolute/extreme language
        extreme_patterns = [
            r'\ball\b', r'\bany\b', r'\bevery\b', r'\bentire\b', r'\bcomplete\b',
            r'\bunlimited\b', r'\babsolute\b', r'\bperpetual\b', r'\bimmediate\b'
        ]
        extreme_count = sum(1 for pattern in extreme_patterns 
                          if re.search(pattern, sentence_lower))
        score += min(extreme_count * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_entity_score(self, sentence: str) -> float:
        """Score based on presence of relevant entities"""
        score = 0.0
        
        # Party names/roles
        if re.search(r'\b(?:company|corporation|contractor|client|vendor|supplier)\b', sentence, re.I):
            score += 0.2
        
        # Legal concepts
        legal_concepts = [
            'indemnification', 'liability', 'damages', 'breach', 'termination',
            'confidentiality', 'intellectual property', 'governing law'
        ]
        for concept in legal_concepts:
            if concept in sentence.lower():
                score += 0.15
                
        return min(score, 1.0)
    
    def _identify_specific_risks(self, sentence: str) -> List[str]:
        """Identify specific risk types in sentence"""
        risks = []
        sentence_lower = sentence.lower()
        
        # Termination risks
        if any(term in sentence_lower for term in ['terminate', 'termination', 'cancel', 'end']):
            if any(term in sentence_lower for term in ['without notice', 'immediately', 'at will']):
                risks.append("HIGH: Arbitrary termination risk")
        
        # Liability risks  
        if any(term in sentence_lower for term in ['liable', 'liability', 'indemnify', 'responsible']):
            if any(term in sentence_lower for term in ['unlimited', 'all', 'any', 'complete']):
                risks.append("CRITICAL: Unlimited liability exposure")
        
        # Payment risks
        if any(term in sentence_lower for term in ['payment', 'pay', 'fee', 'penalty']):
            if any(term in sentence_lower for term in ['immediately', 'upfront', 'advance', 'non-refundable']):
                risks.append("HIGH: Harsh payment terms")
        
        return risks
    
    def _extract_sentence_entities(self, sentence: str) -> Dict[str, List[str]]:
        """Extract entities from individual sentence"""
        entities = {
            'amounts': re.findall(r'\$[\d,]+(?:\.\d{2})?', sentence),
            'percentages': re.findall(r'\d+(?:\.\d+)?%', sentence),
            'timeframes': re.findall(r'\d+\s*(?:days?|months?|years?|hours?)', sentence),
            'parties': re.findall(r'\b(?:company|corporation|contractor|client|vendor|supplier)\b', sentence, re.I)
        }
        return {k: v for k, v in entities.items() if v}
    
    # Keep your existing methods for sentence splitting and key phrase extraction
    def _smart_sentence_split(self, text: str) -> List[str]:
        """Keep your existing implementation - it works!"""
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = sent_tokenize(text)
        
        filtered_sentences = []
        for sentence in sentences:
            length = len(sentence.split())
            if 5 <= length <= 120:  # Slightly wider range
                filtered_sentences.append(sentence.strip())
        
        return filtered_sentences
    
    def _extract_key_phrases(self, sentence: str) -> List[str]:
        """Keep your existing implementation"""
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        word_freq = Counter(words)
        key_words = [word for word, freq in word_freq.most_common(5)]
        
        # Legal phrases
        legal_phrases = []
        sentence_lower = sentence.lower()
        
        phrase_patterns = [
            r'terminate.*without.*(?:notice|cause)',
            r'unlimited.*liability',
            r'intellectual.*property', 
            r'governing.*law',
            r'payment.*due',
            r'penalty.*of'
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, sentence_lower)
            legal_phrases.extend(matches)
        
        return key_words + legal_phrases

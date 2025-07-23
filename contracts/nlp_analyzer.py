import spacy
import asyncio
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from dataclasses import dataclass, asdict
from enum import Enum
import re
import logging
from cachetools import TTLCache
import hashlib
from collections import defaultdict, Counter
import json
from datetime import datetime
from spacy.cli import download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ClauseType(Enum):
    LIABILITY = "LIABILITY"
    TERMINATION = "TERMINATION"
    PAYMENT = "PAYMENT"
    CONFIDENTIALITY = "CONFIDENTIALITY"
    IP_RIGHTS = "IP_RIGHTS"
    COMPLIANCE = "COMPLIANCE"
    GOVERNING_LAW = "GOVERNING_LAW"
    FORCE_MAJEURE = "FORCE_MAJEURE"
    GENERAL = "GENERAL"

@dataclass
class SemanticRiskFeatures:
    """Enhanced semantic features for risk analysis"""
    sentiment_score: float
    complexity_score: float
    legal_density: float
    obligation_strength: float
    temporal_urgency: float
    financial_impact: float
    party_balance: float

@dataclass
class LegalEntity:
    """Enhanced legal entity with context"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context_type: str
    legal_significance: float

@dataclass
class AdvancedRiskAnalysis:
    sentence: str
    risk_level: RiskLevel
    risk_category: str
    confidence_score: float
    entities: List[LegalEntity]
    semantic_similarity: float
    clause_type: ClauseType
    legal_concepts: List[str]
    negotiation_priority: int
    compliance_flags: List[str]
    semantic_features: SemanticRiskFeatures
    risk_explanation: str
    mitigation_strategies: List[str]
    related_clauses: List[int]
    legal_precedent_score: float

@dataclass
class DocumentSemanticProfile:
    """Document-wide semantic analysis"""
    overall_risk_distribution: Dict[str, float]
    semantic_clusters: List[Dict]
    key_themes: List[str]
    party_power_balance: float
    legal_complexity_score: float
    compliance_coverage: Dict[str, float]
    temporal_analysis: Dict[str, List[str]]

class ContractAnalyzer:
    def __init__(self, model_cache_dir: Optional[str] = None):
        self.cache = TTLCache(maxsize=2000, ttl=3600)
        self.model_cache_dir = model_cache_dir
        self._initialize_models()
        self._load_legal_knowledge_base()
        self._initialize_risk_vectors()
        
    def _initialize_models(self):
        """Initialize advanced NLP models with error handling"""
        try:
            # Load spaCy model with custom legal components
            # download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom legal entity ruler
            self._add_legal_entity_patterns()
            
            # Initialize BERT-based models
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Legal text classification with fallback
            try:
                self.legal_classifier = pipeline(
                    "text-classification",
                    model="nlpaueb/legal-bert-base-uncased",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"Legal BERT model not available, using fallback: {e}")
                self.legal_classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Sentence transformer for semantic embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load legal domain-specific embeddings if available
            try:
                self.legal_embeddings = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            except Exception:
                self.legal_embeddings = self.sentence_model
            
            logger.info("Advanced NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            self._initialize_fallback_models()

    def _initialize_fallback_models(self):
        """Initialize basic models as fallback"""
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_pipeline = None
        self.legal_classifier = None
        self.legal_embeddings = self.sentence_model

    def _add_legal_entity_patterns(self):
        """Add custom legal entity recognition patterns"""
        from spacy.matcher import Matcher
        
        matcher = Matcher(self.nlp.vocab)
        
        # Legal entity patterns
        patterns = [
            # Liability patterns
            [{"LOWER": {"IN": ["unlimited", "limited", "joint", "several"]}}, 
             {"LOWER": "liability"}],
            
            # Termination patterns
            [{"LOWER": {"IN": ["immediate", "30", "60", "90"]}}, 
             {"LOWER": {"IN": ["day", "days"]}}, 
             {"LOWER": {"IN": ["notice", "termination"]}}],
            
            # Payment patterns
            [{"LOWER": {"IN": ["net", "payment"]}}, 
             {"LIKE_NUM": True}, 
             {"LOWER": {"IN": ["days", "months"]}}],
            
            # IP patterns
            [{"LOWER": "intellectual"}, {"LOWER": "property"}],
            [{"LOWER": {"IN": ["patent", "copyright", "trademark", "trade"]}}, 
             {"LOWER": {"IN": ["secret", "secrets"]}, "OP": "?"}],
        ]
        
        for i, pattern in enumerate(patterns):
            matcher.add(f"LEGAL_ENTITY_{i}", [pattern])
        
        self.legal_matcher = matcher

    def _load_legal_knowledge_base(self):
        """Load comprehensive legal knowledge base"""
        self.legal_entities = {
            'liability_terms': {
                'terms': ['liable', 'liability', 'responsible', 'accountability', 'damages', 
                         'indemnify', 'indemnification', 'hold harmless', 'joint and several'],
                'risk_weight': 0.8,
                'category': 'LIABILITY'
            },
            'termination_terms': {
                'terms': ['terminate', 'termination', 'breach', 'default', 'expire', 
                         'dissolution', 'cancellation', 'revocation'],
                'risk_weight': 0.7,
                'category': 'TERMINATION'
            },
            'payment_terms': {
                'terms': ['payment', 'invoice', 'fee', 'compensation', 'remuneration', 
                         'penalty', 'late fee', 'interest', 'escrow'],
                'risk_weight': 0.6,
                'category': 'PAYMENT'
            },
            'confidentiality_terms': {
                'terms': ['confidential', 'proprietary', 'non-disclosure', 'trade secret', 
                         'confidentiality', 'proprietary information'],
                'risk_weight': 0.5,
                'category': 'CONFIDENTIALITY'
            },
            'intellectual_property': {
                'terms': ['patent', 'copyright', 'trademark', 'intellectual property', 
                         'IP', 'trade secret', 'proprietary technology'],
                'risk_weight': 0.7,
                'category': 'IP_RIGHTS'
            },
            'compliance_terms': {
                'terms': ['comply', 'regulation', 'statutory', 'legal requirement', 
                         'regulatory', 'compliance', 'audit'],
                'risk_weight': 0.6,
                'category': 'COMPLIANCE'
            },
            'force_majeure_terms': {
                'terms': ['force majeure', 'act of god', 'unforeseeable', 'pandemic', 
                         'natural disaster', 'war', 'terrorism'],
                'risk_weight': 0.4,
                'category': 'FORCE_MAJEURE'
            }
        }
        
        # High-risk clause patterns with semantic context
        self.high_risk_patterns = [
            {
                'pattern': r'\b(?:unlimited|sole|exclusive)\s+liability\b',
                'risk_score': 0.9,
                'explanation': 'Unlimited liability exposure without caps',
                'category': 'LIABILITY'
            },
            {
                'pattern': r'\bimmediate\s+termination\b',
                'risk_score': 0.8,
                'explanation': 'No cure period for contract breaches',
                'category': 'TERMINATION'
            },
            {
                'pattern': r'\bwithout\s+(?:notice|cure\s+period)\b',
                'risk_score': 0.8,
                'explanation': 'Lack of notice or opportunity to cure',
                'category': 'TERMINATION'
            },
            {
                'pattern': r'\ball\s+intellectual\s+property\b',
                'risk_score': 0.7,
                'explanation': 'Broad IP assignment without limitations',
                'category': 'IP_RIGHTS'
            },
            {
                'pattern': r'\bindemnify.*against\s+all\b',
                'risk_score': 0.9,
                'explanation': 'Broad indemnification without limitations',
                'category': 'LIABILITY'
            }
        ]

    def _initialize_risk_vectors(self):
        """Initialize semantic risk vectors for comparison"""
        risk_scenarios = [
            "unlimited liability and damages without caps or limitations",
            "immediate termination without notice or cure period",
            "exclusive intellectual property rights assignment",
            "broad indemnification obligations against all claims",
            "unilateral contract modification rights",
            "no limitation of consequential damages",
            "automatic renewal without opt-out provisions",
            "broad non-compete and non-solicitation clauses",
            "unrestricted audit rights and access",
            "governing law in unfavorable jurisdiction"
        ]
        
        try:
            self.risk_embeddings = self.legal_embeddings.encode(risk_scenarios)
            self.risk_scenario_labels = risk_scenarios
        except Exception as e:
            logger.error(f"Error initializing risk vectors: {e}")
            self.risk_embeddings = None
            self.risk_scenario_labels = []

    async def analyze_contract_advanced(self, text: str) -> Tuple[List[AdvancedRiskAnalysis], DocumentSemanticProfile]:
        """Advanced contract analysis with comprehensive semantic analysis"""
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"advanced_analysis_{text_hash}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Document preprocessing
            processed_text = self._preprocess_legal_text(text)
            
            # Parallel processing of different analysis tasks
            tasks = [
                self._extract_enhanced_entities(processed_text),
                self._classify_clauses_advanced(processed_text),
                self._assess_semantic_risks_comprehensive(processed_text),
                self._detect_compliance_issues_advanced(processed_text),
                self._analyze_document_semantics(processed_text)
            ]
            
            entities, clause_types, semantic_risks, compliance_flags, doc_profile = await asyncio.gather(*tasks)
            
            # Process sentences with advanced analysis
            sentences = self._intelligent_sentence_segmentation(processed_text)
            analyses = []
            
            # Batch process sentences for efficiency
            batch_size = 10
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_analyses = await self._analyze_sentence_batch(
                    batch, i, entities, clause_types, semantic_risks, compliance_flags
                )
                analyses.extend(batch_analyses)
            
            # Post-process and link related clauses
            analyses = self._link_related_clauses(analyses)
            
            # Cache results
            result = (analyses, doc_profile)
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            # Fallback to basic analysis
            fallback_analyses = await self._fallback_analysis(text)
            return fallback_analyses, self._create_basic_doc_profile()

    def _preprocess_legal_text(self, text: str) -> str:
        """Enhanced preprocessing for legal documents"""
        # Handle legal citations
        text = re.sub(r'\b\d+\s+U\.S\.C\.?\s*§?\s*\d+', '[LEGAL_CITATION]', text)
        
        # Handle case citations
        text = re.sub(r'\b\w+\s+v\.?\s+\w+,?\s+\d+', '[CASE_CITATION]', text)
        
        # Normalize legal abbreviations
        legal_abbrevs = {
            r'\bInc\.': 'Incorporated',
            r'\bLLC\.?': 'Limited Liability Company',
            r'\bCorp\.': 'Corporation',
            r'\bLtd\.': 'Limited',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
            r'\betc\.': 'et cetera'
        }
        
        for pattern, replacement in legal_abbrevs.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    async def _extract_enhanced_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities with enhanced context and significance scoring"""
        doc = self.nlp(text)
        entities = []
        
        # Extract spaCy entities
        for ent in doc.ents:
            legal_sig = self._calculate_legal_significance(ent.text, ent.label_)
            context_type = self._determine_context_type(ent, doc)
            
            entities.append(LegalEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=getattr(ent, 'confidence', 0.8),
                context_type=context_type,
                legal_significance=legal_sig
            ))
        
        # Extract custom legal entities
        matches = self.legal_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            entities.append(LegalEntity(
                text=span.text,
                label="LEGAL_TERM",
                start=span.start_char,
                end=span.end_char,
                confidence=0.9,
                context_type="LEGAL_CONCEPT",
                legal_significance=0.8
            ))
        
        return entities

    def _calculate_legal_significance(self, text: str, label: str) -> float:
        """Calculate legal significance score for entities"""
        base_scores = {
            'PERSON': 0.3,
            'ORG': 0.6,
            'MONEY': 0.8,
            'DATE': 0.4,
            'TIME': 0.3,
            'PERCENT': 0.5,
            'LAW': 0.9,
            'LEGAL_TERM': 0.8
        }
        
        base_score = base_scores.get(label, 0.3)
        
        # Boost for legal keywords
        legal_keywords = ['liability', 'damages', 'termination', 'breach', 'indemnity']
        if any(keyword in text.lower() for keyword in legal_keywords):
            base_score += 0.3
        
        return min(base_score, 1.0)

    def _determine_context_type(self, entity, doc) -> str:
        """Determine the context type of an entity"""
        # Analyze surrounding context
        sent = entity.sent
        context_words = [token.text.lower() for token in sent if not token.is_stop]
        
        context_patterns = {
            'OBLIGATION': ['shall', 'must', 'required', 'obligated'],
            'CONDITION': ['if', 'unless', 'provided', 'subject to'],
            'TEMPORAL': ['within', 'after', 'before', 'during'],
            'FINANCIAL': ['pay', 'payment', 'fee', 'cost', 'expense'],
            'LEGAL_STANDARD': ['reasonable', 'material', 'substantial']
        }
        
        for context_type, keywords in context_patterns.items():
            if any(keyword in context_words for keyword in keywords):
                return context_type
        
        return 'GENERAL'

    async def _classify_clauses_advanced(self, text: str) -> Dict[int, ClauseType]:
        """Advanced clause classification with semantic understanding"""
        sentences = self._intelligent_sentence_segmentation(text)
        clause_classifications = {}
        
        # Create embeddings for all sentences
        try:
            sentence_embeddings = self.legal_embeddings.encode(sentences)
        except Exception:
            sentence_embeddings = self.sentence_model.encode(sentences)
        
        # Define clause type templates
        clause_templates = {
            ClauseType.LIABILITY: [
                "party shall be liable for damages",
                "indemnification and hold harmless provisions",
                "limitation of liability and damages"
            ],
            ClauseType.TERMINATION: [
                "termination for cause or convenience",
                "breach and cure period provisions",
                "contract expiration and renewal"
            ],
            ClauseType.PAYMENT: [
                "payment terms and conditions",
                "invoicing and billing procedures",
                "late fees and interest charges"
            ],
            ClauseType.CONFIDENTIALITY: [
                "confidential and proprietary information",
                "non-disclosure obligations",
                "protection of trade secrets"
            ],
            ClauseType.IP_RIGHTS: [
                "intellectual property ownership",
                "patent and copyright licenses",
                "trademark usage rights"
            ],
            ClauseType.COMPLIANCE: [
                "regulatory compliance requirements",
                "legal and statutory obligations",
                "audit and inspection rights"
            ]
        }
        
        # Generate template embeddings
        template_embeddings = {}
        for clause_type, templates in clause_templates.items():
            try:
                embeddings = self.legal_embeddings.encode(templates)
                template_embeddings[clause_type] = np.mean(embeddings, axis=0)
            except Exception:
                embeddings = self.sentence_model.encode(templates)
                template_embeddings[clause_type] = np.mean(embeddings, axis=0)
        
        # Classify each sentence
        for i, sent_embedding in enumerate(sentence_embeddings):
            best_match = ClauseType.GENERAL
            best_score = 0.0
            
            for clause_type, template_embedding in template_embeddings.items():
                similarity = cosine_similarity([sent_embedding], [template_embedding])[0][0]
                if similarity > best_score and similarity > 0.3:  # Threshold for classification
                    best_score = similarity
                    best_match = clause_type
            
            clause_classifications[i] = best_match
        
        return clause_classifications

    async def _assess_semantic_risks_comprehensive(self, text: str) -> Dict[int, Dict[str, float]]:
        """Comprehensive semantic risk assessment with multiple dimensions"""
        sentences = self._intelligent_sentence_segmentation(text)
        semantic_risks = {}
        
        if self.risk_embeddings is None:
            # Fallback to basic pattern matching
            return self._assess_risks_basic_patterns(sentences)
        
        try:
            sentence_embeddings = self.legal_embeddings.encode(sentences)
        except Exception:
            sentence_embeddings = self.sentence_model.encode(sentences)
        
        for i, sent_embedding in enumerate(sentence_embeddings):
            # Calculate similarity to known risk patterns
            similarities = cosine_similarity([sent_embedding], self.risk_embeddings)[0]
            max_similarity = np.max(similarities)
            best_match_idx = np.argmax(similarities)
            
            # Calculate semantic features
            semantic_features = await self._calculate_semantic_features(sentences[i])
            
            # Combined risk score
            risk_score = (max_similarity * 0.4 + 
                         semantic_features.obligation_strength * 0.2 +
                         semantic_features.financial_impact * 0.2 +
                         (1 - semantic_features.party_balance) * 0.2)
            
            semantic_risks[i] = {
                'similarity_score': max_similarity,
                'risk_score': risk_score,
                'best_match': self.risk_scenario_labels[best_match_idx] if best_match_idx < len(self.risk_scenario_labels) else 'unknown',
                'semantic_features': asdict(semantic_features)
            }
        
        return semantic_risks

    async def _calculate_semantic_features(self, sentence: str) -> SemanticRiskFeatures:
        """Calculate detailed semantic features for risk analysis"""
        doc = self.nlp(sentence)
        
        # Sentiment analysis
        sentiment_score = 0.0
        if self.sentiment_pipeline:
            try:
                sentiment_results = self.sentiment_pipeline(sentence)
                if isinstance(sentiment_results, list) and len(sentiment_results) > 0:
                    # Handle different sentiment pipeline outputs
                    if isinstance(sentiment_results[0], list):
                        sentiment_results = sentiment_results[0]
                    
                    negative_score = next((item['score'] for item in sentiment_results 
                                         if 'negative' in item['label'].lower() or 
                                         item['label'] in ['1', '2']), 0.0)
                    sentiment_score = negative_score
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Complexity score (readability inverse)
        complexity_score = min(len(sentence.split()) / 20.0, 1.0)  # Normalize by 20 words
        
        # Legal density (legal terms per sentence)
        legal_terms = 0
        for category_info in self.legal_entities.values():
            legal_terms += sum(1 for term in category_info['terms'] 
                             if term.lower() in sentence.lower())
        legal_density = min(legal_terms / 10.0, 1.0)  # Normalize by 10 terms
        
        # Obligation strength (modal verbs and imperatives)
        obligation_words = ['shall', 'must', 'required', 'mandatory', 'obligated']
        obligation_strength = min(sum(1 for word in obligation_words 
                                    if word in sentence.lower()) / 3.0, 1.0)
        
        # Temporal urgency
        urgent_words = ['immediate', 'forthwith', 'without delay', 'promptly']
        temporal_urgency = min(sum(1 for word in urgent_words 
                                 if word in sentence.lower()) / 2.0, 1.0)
        
        # Financial impact (money-related terms)
        financial_words = ['dollar', 'payment', 'fee', 'cost', 'damage', 'penalty']
        financial_impact = min(sum(1 for word in financial_words 
                                 if word in sentence.lower()) / 3.0, 1.0)
        
        # Party balance (mutual vs unilateral obligations)
        mutual_words = ['mutual', 'both parties', 'each party']
        unilateral_words = ['party a', 'party b', 'contractor', 'client']
        mutual_score = sum(1 for word in mutual_words if word in sentence.lower())
        unilateral_score = sum(1 for word in unilateral_words if word in sentence.lower())
        party_balance = 0.5 if mutual_score == unilateral_score else (
            mutual_score / (mutual_score + unilateral_score + 1)
        )
        
        return SemanticRiskFeatures(
            sentiment_score=sentiment_score,
            complexity_score=complexity_score,
            legal_density=legal_density,
            obligation_strength=obligation_strength,
            temporal_urgency=temporal_urgency,
            financial_impact=financial_impact,
            party_balance=party_balance
        )

    def _assess_risks_basic_patterns(self, sentences: List[str]) -> Dict[int, Dict[str, float]]:
        """Fallback risk assessment using pattern matching"""
        risks = {}
        
        for i, sentence in enumerate(sentences):
            risk_score = 0.0
            best_match = "basic pattern analysis"
            
            # Check high-risk patterns
            for pattern_info in self.high_risk_patterns:
                if re.search(pattern_info['pattern'], sentence, re.IGNORECASE):
                    risk_score = max(risk_score, pattern_info['risk_score'])
                    best_match = pattern_info['explanation']
            
            # Check legal entity density
            legal_term_count = 0
            for category_info in self.legal_entities.values():
                for term in category_info['terms']:
                    if term.lower() in sentence.lower():
                        legal_term_count += 1
                        risk_score += category_info['risk_weight'] * 0.1
            
            risks[i] = {
                'similarity_score': min(risk_score, 1.0),
                'risk_score': min(risk_score, 1.0),
                'best_match': best_match,
                'semantic_features': {}
            }
        
        return risks

    async def _detect_compliance_issues_advanced(self, text: str) -> List[str]:
        """Advanced compliance issue detection"""
        compliance_flags = []
        text_lower = text.lower()
        
        # GDPR compliance checks
        privacy_terms = ['personal data', 'privacy', 'data protection', 'data subject']
        if any(term in text_lower for term in privacy_terms):
            if not any(term in text_lower for term in ['gdpr', 'data protection regulation', 'privacy policy']):
                compliance_flags.append('GDPR_COMPLIANCE_MISSING')
        
        # Jurisdiction and governing law
        if not any(term in text_lower for term in ['governing law', 'jurisdiction', 'venue']):
            compliance_flags.append('JURISDICTION_UNCLEAR')
        
        # Employment law compliance
        employment_terms = ['employee', 'employment', 'worker', 'contractor']
        if any(term in text_lower for term in employment_terms):
            if not any(term in text_lower for term in ['labor law', 'employment law', 'wage', 'overtime']):
                compliance_flags.append('EMPLOYMENT_LAW_GAPS')
        
        # Anti-corruption compliance
        if any(term in text_lower for term in ['government', 'official', 'public sector']):
            if not any(term in text_lower for term in ['anti-corruption', 'fcpa', 'bribery']):
                compliance_flags.append('ANTI_CORRUPTION_MISSING')
        
        # Accessibility compliance
        if any(term in text_lower for term in ['website', 'software', 'application']):
            if not any(term in text_lower for term in ['accessibility', 'ada', 'wcag']):
                compliance_flags.append('ACCESSIBILITY_COMPLIANCE_MISSING')
        
        return compliance_flags

    async def _analyze_document_semantics(self, text: str) -> DocumentSemanticProfile:
        """Analyze document-level semantic patterns and themes"""
        sentences = self._intelligent_sentence_segmentation(text)
        
        try:
            # Create sentence embeddings
            embeddings = self.legal_embeddings.encode(sentences)
            
            # Perform clustering to identify themes
            n_clusters = min(5, len(sentences) // 3)  # Adaptive cluster count
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
            else:
                clusters = [0] * len(sentences)
            
            # Analyze clusters
            semantic_clusters = []
            for i in range(n_clusters):
                cluster_sentences = [sentences[j] for j, c in enumerate(clusters) if c == i]
                cluster_theme = self._extract_cluster_theme(cluster_sentences)
                semantic_clusters.append({
                    'cluster_id': i,
                    'theme': cluster_theme,
                    'sentences': cluster_sentences[:3],  # Sample sentences
                    'size': len(cluster_sentences)
                })
            
            # Calculate overall metrics
            risk_distribution = self._calculate_risk_distribution(sentences)
            party_balance = self._analyze_party_power_balance(text)
            complexity_score = self._calculate_document_complexity(text)
            compliance_coverage = self._assess_compliance_coverage(text)
            temporal_analysis = self._analyze_temporal_elements(text)
            key_themes = self._extract_key_themes(text)
            
            return DocumentSemanticProfile(
                overall_risk_distribution=risk_distribution,
                semantic_clusters=semantic_clusters,
                key_themes=key_themes,
                party_power_balance=party_balance,
                legal_complexity_score=complexity_score,
                compliance_coverage=compliance_coverage,
                temporal_analysis=temporal_analysis
            )
            
        except Exception as e:
            logger.error(f"Document semantic analysis failed: {e}")
            return self._create_basic_doc_profile()

    def _extract_cluster_theme(self, sentences: List[str]) -> str:
        """Extract theme from sentence cluster"""
        if not sentences:
            return "General"
        
        # Count key terms across cluster
        term_counts = Counter()
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            term_counts.update(words)
        
        # Map to legal themes
        theme_keywords = {
            'Liability': ['liability', 'liable', 'damages', 'harm', 'loss'],
            'Payment': ['payment', 'fee', 'cost', 'invoice', 'billing'],
            'Termination': ['terminate', 'end', 'breach', 'default', 'expire'],
            'Intellectual Property': ['intellectual', 'property', 'patent', 'copyright'],
            'Confidentiality': ['confidential', 'proprietary', 'disclosure', 'secret'],
            'Compliance': ['comply', 'regulation', 'law', 'statutory', 'regulatory']
        }
        
        best_theme = "General"
        best_score = 0
        
        for theme, keywords in theme_keywords.items():
            score = sum(term_counts.get(keyword, 0) for keyword in keywords)
            if score > best_score:
                best_score = score
                best_theme = theme
        
        return best_theme

    def _calculate_risk_distribution(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate distribution of risk levels across document"""
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for sentence in sentences:
            # Basic risk assessment
            risk_score = 0
            for pattern_info in self.high_risk_patterns:
                if re.search(pattern_info['pattern'], sentence, re.IGNORECASE):
                    risk_score = max(risk_score, pattern_info['risk_score'])
            
            # Categorize risk
            if risk_score >= 0.8:
                risk_counts['CRITICAL'] += 1
            elif risk_score >= 0.6:
                risk_counts['HIGH'] += 1
            elif risk_score >= 0.3:
                risk_counts['MEDIUM'] += 1
            else:
                risk_counts['LOW'] += 1
        
        total = sum(risk_counts.values())
        return {k: v/total if total > 0 else 0 for k, v in risk_counts.items()}

    def _analyze_party_power_balance(self, text: str) -> float:
        """Analyze power balance between contracting parties"""
        text_lower = text.lower()
        
        # Count obligations for each party
        party_a_obligations = len(re.findall(r'\b(?:party a|contractor|vendor|supplier)\s+shall\b', text_lower))
        party_b_obligations = len(re.findall(r'\b(?:party b|client|customer|buyer)\s+shall\b', text_lower))
        
        # Count rights/benefits
        party_a_rights = len(re.findall(r'\b(?:party a|contractor|vendor|supplier)\s+(?:may|entitled)\b', text_lower))
        party_b_rights = len(re.findall(r'\b(?:party b|client|customer|buyer)\s+(?:may|entitled)\b', text_lower))
        
        # Calculate balance (0.5 = balanced, 0 = heavily favors party B, 1 = heavily favors party A)
        total_obligations = party_a_obligations + party_b_obligations
        total_rights = party_a_rights + party_b_rights
        
        if total_obligations + total_rights == 0:
            return 0.5  # No clear imbalance detected
        
        party_a_burden = (party_a_obligations - party_a_rights) if total_obligations + total_rights > 0 else 0
        party_b_burden = (party_b_obligations - party_b_rights) if total_obligations + total_rights > 0 else 0
        
        # Normalize to 0-1 scale
        max_burden = max(abs(party_a_burden), abs(party_b_burden), 1)
        balance_score = 0.5 + (party_b_burden - party_a_burden) / (2 * max_burden)
        
        return max(0, min(1, balance_score))

    def _calculate_document_complexity(self, text: str) -> float:
        """Calculate overall document complexity score"""
        doc = self.nlp(text)
        
        # Factors for complexity
        avg_sentence_length = np.mean([len(sent.text.split()) for sent in doc.sents])
        legal_term_density = sum(1 for token in doc if token.text.lower() in 
                               [term for category in self.legal_entities.values() 
                                for term in category['terms']]) / len(doc)
        
        unique_legal_concepts = len(set(token.text.lower() for token in doc 
                                      if token.text.lower() in 
                                      [term for category in self.legal_entities.values() 
                                       for term in category['terms']]))
        
        # Normalize and combine scores
        length_score = min(avg_sentence_length / 25, 1.0)  # Normalize by 25 words
        density_score = min(legal_term_density * 10, 1.0)  # Scale up density
        concept_score = min(unique_legal_concepts / 20, 1.0)  # Normalize by 20 concepts
        
        return (length_score + density_score + concept_score) / 3

    def _assess_compliance_coverage(self, text: str) -> Dict[str, float]:
        """Assess coverage of different compliance areas"""
        text_lower = text.lower()
        
        compliance_areas = {
            'data_protection': ['gdpr', 'privacy', 'data protection', 'personal data'],
            'employment_law': ['employment law', 'labor law', 'worker rights', 'wage'],
            'intellectual_property': ['ip', 'intellectual property', 'patent', 'copyright'],
            'anti_corruption': ['anti-corruption', 'bribery', 'fcpa', 'kickback'],
            'environmental': ['environmental', 'sustainability', 'green', 'carbon'],
            'accessibility': ['accessibility', 'ada', 'wcag', 'disability']
        }
        
        coverage = {}
        for area, keywords in compliance_areas.items():
            mentions = sum(1 for keyword in keywords if keyword in text_lower)
            coverage[area] = min(mentions / len(keywords), 1.0)
        
        return coverage

    def _analyze_temporal_elements(self, text: str) -> Dict[str, List[str]]:
        """Analyze temporal elements in the contract"""
        doc = self.nlp(text)
        
        temporal_elements = {
            'deadlines': [],
            'durations': [],
            'renewal_terms': [],
            'notice_periods': []
        }
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Deadlines
            if any(word in sent_text for word in ['deadline', 'due by', 'no later than']):
                temporal_elements['deadlines'].append(sent.text.strip())
            
            # Durations
            if re.search(r'\b\d+\s+(?:days?|months?|years?)\b', sent_text):
                temporal_elements['durations'].append(sent.text.strip())
            
            # Renewal terms
            if any(word in sent_text for word in ['renew', 'renewal', 'extend', 'extension']):
                temporal_elements['renewal_terms'].append(sent.text.strip())
            
            # Notice periods
            if re.search(r'\b\d+\s+days?\s+notice\b', sent_text):
                temporal_elements['notice_periods'].append(sent.text.strip())
        
        # Limit results
        for key in temporal_elements:
            temporal_elements[key] = temporal_elements[key][:3]
        
        return temporal_elements

    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from the document"""
        doc = self.nlp(text)
        
        # Count important terms
        term_counts = Counter()
        for token in doc:
            if (not token.is_stop and not token.is_punct and 
                len(token.text) > 3 and token.pos_ in ['NOUN', 'ADJ']):
                term_counts[token.lemma_.lower()] += 1
        
        # Get most common terms
        common_terms = [term for term, count in term_counts.most_common(10)]
        
        # Map to themes
        theme_mapping = {
            'service': 'Service Agreement',
            'payment': 'Financial Terms',
            'liability': 'Risk and Liability',
            'property': 'Intellectual Property',
            'confidential': 'Confidentiality',
            'termination': 'Contract Termination',
            'compliance': 'Regulatory Compliance'
        }
        
        themes = []
        for term in common_terms:
            for key, theme in theme_mapping.items():
                if key in term and theme not in themes:
                    themes.append(theme)
        
        return themes[:5]  # Return top 5 themes

    def _create_basic_doc_profile(self) -> DocumentSemanticProfile:
        """Create a basic document profile as fallback"""
        return DocumentSemanticProfile(
            overall_risk_distribution={'LOW': 0.7, 'MEDIUM': 0.2, 'HIGH': 0.1, 'CRITICAL': 0.0},
            semantic_clusters=[],
            key_themes=['General Contract'],
            party_power_balance=0.5,
            legal_complexity_score=0.3,
            compliance_coverage={},
            temporal_analysis={}
        )

    async def _analyze_sentence_batch(
        self,
        sentences: List[str],
        start_index: int,
        entities: List[LegalEntity],
        clause_types: Dict[int, ClauseType],
        semantic_risks: Dict[int, Dict[str, float]],
        compliance_flags: List[str]
    ) -> List[AdvancedRiskAnalysis]:
        """Analyze a batch of sentences efficiently"""
        analyses = []
        
        for i, sentence in enumerate(sentences):
            sentence_index = start_index + i
            analysis = await self._analyze_sentence_advanced(
                sentence, sentence_index, entities, clause_types, 
                semantic_risks, compliance_flags
            )
            if analysis:
                analyses.append(analysis)
        
        return analyses

    async def _analyze_sentence_advanced(
        self,
        sentence: str,
        index: int,
        entities: List[LegalEntity],
        clause_types: Dict[int, ClauseType],
        semantic_risks: Dict[int, Dict[str, float]],
        compliance_flags: List[str]
    ) -> Optional[AdvancedRiskAnalysis]:
        """Advanced sentence-level analysis with comprehensive features"""
        
        # Extract relevant entities for this sentence
        sentence_start = 0  # Approximate sentence position
        sentence_end = len(sentence)
        
        sentence_entities = [
            ent for ent in entities 
            if (ent.start >= sentence_start and ent.end <= sentence_end) or
               (sentence.lower() in ent.text.lower() or ent.text.lower() in sentence.lower())
        ]
        
        # Get semantic risk data
        risk_data = semantic_risks.get(index, {
            'similarity_score': 0.0,
            'risk_score': 0.0,
            'best_match': 'no match',
            'semantic_features': {}
        })
        
        # Determine risk level using comprehensive scoring
        risk_score = risk_data['risk_score']
        confidence = min(risk_data['similarity_score'] + 0.3, 1.0)
        
        # Pattern-based risk enhancement
        for pattern_info in self.high_risk_patterns:
            if re.search(pattern_info['pattern'], sentence, re.IGNORECASE):
                risk_score = max(risk_score, pattern_info['risk_score'])
                confidence += 0.2
        
        # Entity-based risk adjustment
        if sentence_entities:
            avg_significance = np.mean([ent.legal_significance for ent in sentence_entities])
            risk_score += avg_significance * 0.2
            confidence += 0.1
        
        # Determine final risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Skip very low-risk sentences to reduce noise
        if risk_level == RiskLevel.LOW and confidence < 0.4:
            return None
        
        # Identify legal concepts and generate explanation
        legal_concepts = self._identify_legal_concepts(sentence)
        risk_category = self._categorize_risk(sentence, legal_concepts)
        explanation = self._generate_risk_explanation(sentence, risk_data, legal_concepts)
        mitigation_strategies = self._generate_mitigation_strategies(risk_category, risk_level)
        
        # Extract semantic features
        semantic_features_dict = risk_data.get('semantic_features', {})
        semantic_features = SemanticRiskFeatures(
            sentiment_score=semantic_features_dict.get('sentiment_score', 0.0),
            complexity_score=semantic_features_dict.get('complexity_score', 0.0),
            legal_density=semantic_features_dict.get('legal_density', 0.0),
            obligation_strength=semantic_features_dict.get('obligation_strength', 0.0),
            temporal_urgency=semantic_features_dict.get('temporal_urgency', 0.0),
            financial_impact=semantic_features_dict.get('financial_impact', 0.0),
            party_balance=semantic_features_dict.get('party_balance', 0.5)
        )
        
        return AdvancedRiskAnalysis(
            sentence=sentence,
            risk_level=risk_level,
            risk_category=risk_category,
            confidence_score=min(confidence, 1.0),
            entities=sentence_entities,
            semantic_similarity=risk_data['similarity_score'],
            clause_type=clause_types.get(index, ClauseType.GENERAL),
            legal_concepts=legal_concepts,
            negotiation_priority=self._calculate_negotiation_priority(risk_level, confidence),
            compliance_flags=[flag for flag in compliance_flags 
                            if self._flag_relevant_to_sentence(sentence, flag)],
            semantic_features=semantic_features,
            risk_explanation=explanation,
            mitigation_strategies=mitigation_strategies,
            related_clauses=[],  # Will be populated by post-processing
            legal_precedent_score=self._calculate_precedent_score(sentence, legal_concepts)
        )

    def _identify_legal_concepts(self, sentence: str) -> List[str]:
        """Identify legal concepts in sentence with enhanced detection"""
        concepts = []
        sentence_lower = sentence.lower()
        
        # Check against knowledge base
        for concept_key, concept_info in self.legal_entities.items():
            if any(term in sentence_lower for term in concept_info['terms']):
                concept_name = concept_key.replace('_', ' ').title()
                if concept_name not in concepts:
                    concepts.append(concept_name)
        
        # Additional legal concept detection
        advanced_concepts = {
            'Force Majeure': ['force majeure', 'act of god', 'unforeseeable'],
            'Governing Law': ['governing law', 'jurisdiction', 'venue'],
            'Dispute Resolution': ['arbitration', 'mediation', 'dispute resolution'],
            'Assignment': ['assign', 'assignment', 'transfer'],
            'Severability': ['severability', 'severable', 'invalid provision'],
            'Entire Agreement': ['entire agreement', 'whole agreement', 'complete agreement']
        }
        
        for concept, keywords in advanced_concepts.items():
            if any(keyword in sentence_lower for keyword in keywords):
                if concept not in concepts:
                    concepts.append(concept)
        
        return concepts

    def _categorize_risk(self, sentence: str, legal_concepts: List[str]) -> str:
        """Enhanced risk categorization with semantic understanding"""
        sentence_lower = sentence.lower()
        
        # Priority-based categorization
        if any('liability' in concept.lower() or 'indemnif' in sentence_lower for concept in legal_concepts):
            return 'Liability and Indemnification Risk'
        elif any('payment' in concept.lower() or 'financial' in sentence_lower for concept in legal_concepts):
            return 'Financial and Payment Risk'
        elif any('termination' in concept.lower() or 'breach' in sentence_lower for concept in legal_concepts):
            return 'Contract Continuity Risk'
        elif any('intellectual' in concept.lower() or 'ip' in sentence_lower for concept in legal_concepts):
            return 'Intellectual Property Risk'
        elif any('compliance' in concept.lower() or 'regulatory' in sentence_lower for concept in legal_concepts):
            return 'Regulatory Compliance Risk'
        elif any('confidential' in concept.lower() or 'disclosure' in sentence_lower for concept in legal_concepts):
            return 'Confidentiality and Data Risk'
        elif 'Force Majeure' in legal_concepts:
            return 'Force Majeure and External Risk'
        elif 'Dispute Resolution' in legal_concepts:
            return 'Dispute Resolution Risk'
        else:
            return 'General Contractual Risk'

    def _generate_risk_explanation(self, sentence: str, risk_data: Dict, legal_concepts: List[str]) -> str:
        """Generate detailed risk explanation"""
        base_explanation = f"This clause presents risk due to {risk_data.get('best_match', 'contractual obligations')}."
        
        # Add concept-specific explanations
        if legal_concepts:
            concept_text = ', '.join(legal_concepts[:3])
            base_explanation += f" The clause involves {concept_text.lower()}, which requires careful attention."
        
        # Add pattern-specific explanations
        for pattern_info in self.high_risk_patterns:
            if re.search(pattern_info['pattern'], sentence, re.IGNORECASE):
                base_explanation += f" Specifically, {pattern_info['explanation'].lower()}."
                break
        
        return base_explanation

    def _generate_mitigation_strategies(self, risk_category: str, risk_level: RiskLevel) -> List[str]:
        """Generate context-specific mitigation strategies"""
        strategies = []
        
        # Category-specific strategies
        category_strategies = {
            'Liability and Indemnification Risk': [
                'Add liability caps to limit maximum exposure',
                'Exclude consequential and punitive damages',
                'Require adequate insurance coverage',
                'Add mutual indemnification clauses'
            ],
            'Financial and Payment Risk': [
                'Negotiate shorter payment terms',
                'Add interest on late payments',
                'Include clear dispute resolution for billing',
                'Consider escrow for large payments'
            ],
            'Contract Continuity Risk': [
                'Add adequate cure periods for breaches',
                'Include mutual termination rights',
                'Clarify material breach definitions',
                'Add termination for convenience clauses'
            ],
            'Intellectual Property Risk': [
                'Retain ownership of pre-existing IP',
                'Limit scope of IP assignments',
                'Add IP indemnification provisions',
                'Define work-for-hire clearly'
            ],
            'Regulatory Compliance Risk': [
                'Add materiality qualifiers to compliance obligations',
                'Include compliance cost-sharing provisions',
                'Regular compliance review procedures',
                'Add regulatory change notification requirements'
            ]
        }
        
        base_strategies = category_strategies.get(risk_category, [
            'Review clause carefully with legal counsel',
            'Consider negotiating more balanced terms',
            'Add appropriate limitations or qualifications'
        ])
        
        # Risk level adjustments
        if risk_level == RiskLevel.CRITICAL:
            strategies.append('CRITICAL: Requires immediate legal review before acceptance')
        elif risk_level == RiskLevel.HIGH:
            strategies.append('HIGH PRIORITY: Focus negotiation efforts on this clause')
        
        strategies.extend(base_strategies[:3])  # Limit to avoid overwhelming output
        return strategies

    def _calculate_precedent_score(self, sentence: str, legal_concepts: List[str]) -> float:
        """Calculate legal precedent relevance score"""
        # This is a simplified implementation
        # In a real system, this would query legal databases
        
        base_score = 0.3  # Default precedent relevance
        
        # Higher scores for well-established legal concepts
        established_concepts = [
            'Liability Terms', 'Payment Terms', 'Intellectual Property',
            'Force Majeure', 'Governing Law', 'Dispute Resolution'
        ]
        
        if any(concept in established_concepts for concept in legal_concepts):
            base_score += 0.3
        
        # Boost for specific legal language
        legal_language_indicators = ['pursuant to', 'in accordance with', 'subject to', 'notwithstanding']
        if any(indicator in sentence.lower() for indicator in legal_language_indicators):
            base_score += 0.2
        
        return min(base_score, 1.0)

    def _calculate_negotiation_priority(self, risk_level: RiskLevel, confidence: float) -> int:
        """Calculate negotiation priority with enhanced logic"""
        base_priority = {
            RiskLevel.CRITICAL: 10,
            RiskLevel.HIGH: 8,
            RiskLevel.MEDIUM: 5,
            RiskLevel.LOW: 2
        }
        
        priority = base_priority[risk_level]
        
        # Adjust based on confidence
        confidence_adjustment = int(confidence * 2)
        priority = min(priority + confidence_adjustment, 10)
        
        return max(priority, 1)

    def _flag_relevant_to_sentence(self, sentence: str, flag: str) -> bool:
        """Enhanced compliance flag relevance check"""
        relevance_map = {
            'GDPR_COMPLIANCE_MISSING': ['data', 'personal', 'privacy', 'information', 'process'],
            'JURISDICTION_UNCLEAR': ['law', 'court', 'dispute', 'govern', 'jurisdiction'],
            'EMPLOYMENT_LAW_GAPS': ['employee', 'worker', 'labor', 'employment', 'wage'],
            'ANTI_CORRUPTION_MISSING': ['government', 'official', 'public', 'authority'],
            'ACCESSIBILITY_COMPLIANCE_MISSING': ['website', 'software', 'application', 'interface']
        }
        
        keywords = relevance_map.get(flag, [])
        return any(keyword in sentence.lower() for keyword in keywords)

    def _link_related_clauses(self, analyses: List[AdvancedRiskAnalysis]) -> List[AdvancedRiskAnalysis]:
        """Link related clauses based on semantic similarity"""
        if len(analyses) < 2:
            return analyses
        
        try:
            # Create embeddings for all sentences
            sentences = [analysis.sentence for analysis in analyses]
            embeddings = self.legal_embeddings.encode(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Link related clauses
            for i, analysis in enumerate(analyses):
                related_indices = []
                for j in range(len(analyses)):
                    if i != j and similarity_matrix[i][j] > 0.7:  # High similarity threshold
                        related_indices.append(j)
                
                analysis.related_clauses = related_indices[:3]  # Limit to top 3 related clauses
            
        except Exception as e:
            logger.warning(f"Failed to link related clauses: {e}")
        
        return analyses

    def _intelligent_sentence_segmentation(self, text: str) -> List[str]:
        """Enhanced sentence segmentation for legal documents"""
        # Handle legal abbreviations and citations
        text = re.sub(r'\b(Inc|LLC|Corp|Ltd|Co|etc|vs|v|U\.S\.C|F\.R|C\.F\.R)\.\s*', r'\1<DOT> ', text)
        
        # Handle numbered sections
        text = re.sub(r'\b(\d+)\.\s*(\d+)\s*', r'\1<DOT>\2 ', text)
        
        # Use spaCy for segmentation
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            # Restore periods and clean up
            clean_sent = sent.text.replace('<DOT>', '.').strip()
            
            # Filter out very short sentences and headers
            if len(clean_sent) > 15 and not re.match(r'^\d+\.?\s*$', clean_sent):
                sentences.append(clean_sent)
        
        return sentences

    async def _fallback_analysis(self, text: str) -> List[AdvancedRiskAnalysis]:
        """Comprehensive fallback analysis when advanced methods fail"""
        logger.warning("Using comprehensive fallback analysis")
        
        sentences = self._intelligent_sentence_segmentation(text)
        analyses = []
        
        for i, sentence in enumerate(sentences):
            # Basic risk assessment
            risk_score = 0.0
            risk_explanation = "Basic pattern analysis"
            risk_category = "General Contractual Risk"
            
            # Check high-risk patterns
            for pattern_info in self.high_risk_patterns:
                if re.search(pattern_info['pattern'], sentence, re.IGNORECASE):
                    risk_score = max(risk_score, pattern_info['risk_score'])
                    risk_explanation = pattern_info['explanation']
                    risk_category = f"{pattern_info['category']} Risk"
            
            # Check legal entity mentions
            legal_concepts = []
            for concept_key, concept_info in self.legal_entities.items():
                if any(term.lower() in sentence.lower() for term in concept_info['terms']):
                    legal_concepts.append(concept_key.replace('_', ' ').title())
                    risk_score += concept_info['risk_weight'] * 0.2
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Skip very low risk sentences
            if risk_level == RiskLevel.LOW and risk_score < 0.2:
                continue
            
            # Create basic semantic features
            semantic_features = SemanticRiskFeatures(
                sentiment_score=0.0,
                complexity_score=min(len(sentence.split()) / 20.0, 1.0),
                legal_density=len(legal_concepts) / 5.0,
                obligation_strength=float('shall' in sentence.lower() or 'must' in sentence.lower()),
                temporal_urgency=float('immediate' in sentence.lower()),
                financial_impact=float(any(word in sentence.lower() for word in ['payment', 'fee', 'cost', 'damage'])),
                party_balance=0.5
            )
            
            # Create analysis
            analysis = AdvancedRiskAnalysis(
                sentence=sentence,
                risk_level=risk_level,
                risk_category=risk_category,
                confidence_score=0.6,  # Moderate confidence for fallback
                entities=[],
                semantic_similarity=risk_score,
                clause_type=ClauseType.GENERAL,
                legal_concepts=legal_concepts,
                negotiation_priority=min(int(risk_score * 10), 8),
                compliance_flags=[],
                semantic_features=semantic_features,
                risk_explanation=risk_explanation,
                mitigation_strategies=self._generate_mitigation_strategies(risk_category, risk_level),
                related_clauses=[],
                legal_precedent_score=0.3
            )
            
            analyses.append(analysis)
        
        return analyses

    def get_analysis_summary(self, analyses: List[AdvancedRiskAnalysis]) -> Dict:
        """Generate comprehensive analysis summary"""
        if not analyses:
            return {
                'total_risks': 0,
                'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
                'top_risks': [],
                'recommendations': ['No significant risks detected']
            }
        
        # Risk distribution
        risk_counts = Counter(analysis.risk_level.value for analysis in analyses)
        total_risks = len(analyses)
        
        # Top priority risks
        top_risks = sorted(analyses, key=lambda x: x.negotiation_priority, reverse=True)[:5]
        
        # Generate recommendations
        recommendations = []
        if risk_counts['CRITICAL'] > 0:
            recommendations.append(f"🚨 {risk_counts['CRITICAL']} critical risk(s) require immediate attention")
        if risk_counts['HIGH'] > 2:
            recommendations.append(f"⚠️ {risk_counts['HIGH']} high-risk clauses need careful review")
        if total_risks > 10:
            recommendations.append("📋 Consider comprehensive legal review due to complexity")
        
        # Category analysis
        category_counts = Counter(analysis.risk_category for analysis in analyses)
        top_category = category_counts.most_common(1)[0] if category_counts else ('General', 0)
        recommendations.append(f"🎯 Focus negotiation on {top_category[0]} ({top_category[1]} instances)")
        
        return {
            'total_risks': total_risks,
            'risk_distribution': dict(risk_counts),
            'top_risks': [asdict(risk) for risk in top_risks],
            'category_distribution': dict(category_counts),
            'recommendations': recommendations,
            'avg_confidence': np.mean([analysis.confidence_score for analysis in analyses]),
            'avg_negotiation_priority': np.mean([analysis.negotiation_priority for analysis in analyses])
        }

# # Usage example and testing
# if __name__ == "__main__":
#     async def test_analyzer():
#         analyzer = ContractAnalyzer()
        
#         sample_contract = """
#         This Service Agreement ("Agreement") is entered into on January 1, 2024, between Company A and Company B.
#         The Contractor shall be liable for all damages arising from performance of services without limitation.
#         Either party may terminate this agreement immediately without notice upon any breach.
#         All intellectual property created during the term shall be exclusively owned by Company A.
#         The Contractor shall indemnify Company A against all claims, damages, and expenses.
#         Payment shall be made within 90 days of invoice receipt.
#         This Agreement shall be governed by the laws of Delaware.
#         """
        
#         try:
#             analyses, doc_profile = await analyzer.analyze_contract_advanced(sample_contract)
#             summary = analyzer.get_analysis_summary(analyses)
            
#             print("=== ANALYSIS SUMMARY ===")
#             print(f"Total Risks Found: {summary['total_risks']}")
#             print(f"Risk Distribution: {summary['risk_distribution']}")
#             print(f"Average Confidence: {summary['avg_confidence']:.2f}")
            
#             print("\n=== TOP RISKS ===")
#             for i, risk in enumerate(analyses[:3], 1):
#                 print(f"{i}. {risk.risk_level.value} - {risk.risk_category}")
#                 print(f"   Confidence: {risk.confidence_score:.2f}")
#                 print(f"   Priority: {risk.negotiation_priority}/10")
#                 print(f"   Explanation: {risk.risk_explanation}")
#                 print()
            
#             print("=== DOCUMENT PROFILE ===")
#             print(f"Legal Complexity: {doc_profile.legal_complexity_score:.2f}")
#             print(f"Party Balance: {doc_profile.party_power_balance:.2f}")
#             print(f"Key Themes: {', '.join(doc_profile.key_themes)}")
            
#         except Exception as e:
#             print(f"Analysis failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Run test
#     asyncio.run(test_analyzer())

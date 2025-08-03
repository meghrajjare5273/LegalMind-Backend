import os
import logging
import re
from typing import Dict, List, Any, Optional
import json
import asyncio
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class AzureExplainabilityService:
    """Azure-powered explainability enhancements"""
    
    def __init__(self):
        self.text_analytics_client = self._initialize_text_analytics()
        self.legal_entities_cache = {}
        self.sentiment_cache = {}
        
    def _initialize_text_analytics(self) -> Optional[TextAnalyticsClient]:
        """Initialize Azure Text Analytics client"""
        try:
            endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
            key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
            
            if not endpoint or not key:
                logger.warning("Azure Text Analytics credentials not configured")
                return None
            
            credential = AzureKeyCredential(key)
            client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
            
            # Test the connection
            test_documents = [{"id": "1", "text": "Test connection"}]
            try:
                client.detect_language(test_documents)
                logger.info("Azure Text Analytics client initialized successfully")
            except Exception as test_error:
                logger.error(f"Azure Text Analytics client test failed: {test_error}")
                return None
                
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Text Analytics: {e}")
            return None
    
    async def enhance_explanation_with_azure(self, 
                                           risk_item: Dict[str, Any],
                                           base_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance explanation using Azure Cognitive Services"""
        
        if not self.text_analytics_client:
            logger.warning("Azure client not available, returning base explanation")
            return base_explanation
        
        sentence = risk_item.get("sentence", "")
        if not sentence or len(sentence) < 10:
            return base_explanation
        
        try:
            # Check cache first
            cache_key = hash(sentence)
            if cache_key in self.legal_entities_cache:
                logger.info("Using cached Azure analysis")
                azure_insights = self.legal_entities_cache[cache_key]
            else:
                # 1. Extract key phrases
                key_phrases = await self._extract_key_phrases(sentence)
                
                # 2. Sentiment analysis for risk tone
                sentiment_analysis = await self._analyze_sentiment(sentence)
                
                # 3. Named entity recognition
                entities = await self._extract_entities(sentence)
                
                # 4. Language detection and confidence
                language_info = await self._detect_language(sentence)
                
                # 5. Custom legal entity extraction
                legal_entities = self._extract_legal_entities(sentence)
                
                azure_insights = {
                    "key_phrases": key_phrases,
                    "sentiment_analysis": sentiment_analysis,
                    "entities": entities,
                    "language_info": language_info,
                    "legal_entities": legal_entities,
                    "risk_indicators": self._analyze_risk_indicators(
                        key_phrases, sentiment_analysis, entities
                    ),
                    "text_quality": self._assess_text_quality(sentence),
                    "complexity_metrics": self._calculate_azure_complexity(sentence)
                }
                
                # Cache the results
                self.legal_entities_cache[cache_key] = azure_insights
            
            # Enhance the explanation
            enhanced_explanation = base_explanation.copy()
            enhanced_explanation["azure_insights"] = azure_insights
            
            # Update human rationale with Azure insights
            azure_rationale = self._generate_azure_rationale(azure_insights)
            
            if azure_rationale:
                enhanced_explanation["human_rationale"] += f"\n\n**Azure AI Insights:**\n{azure_rationale}"
            
            # Update confidence based on Azure analysis
            azure_confidence_boost = self._calculate_azure_confidence_boost(azure_insights)
            if "confidence_breakdown" in enhanced_explanation:
                enhanced_explanation["confidence_breakdown"]["azure_enhancement"] = azure_confidence_boost
            
            return enhanced_explanation
            
        except Exception as e:
            logger.error(f"Azure enhancement failed: {e}")
            return base_explanation
    
    async def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using Azure Text Analytics"""
        try:
            documents = [{"id": "1", "text": text}]
            response = self.text_analytics_client.extract_key_phrases(documents)
            
            for doc in response:
                if not doc.is_error:
                    # Filter for legal-relevant phrases
                    legal_phrases = [
                        phrase for phrase in doc.key_phrases 
                        if self._is_legal_relevant_phrase(phrase)
                    ]
                    return legal_phrases[:10]  # Limit to top 10
            return []
            
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment to understand clause tone"""
        try:
            documents = [{"id": "1", "text": text}]
            response = self.text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
            
            for doc in response:
                if not doc.is_error:
                    sentiment_data = {
                        "overall_sentiment": doc.sentiment,
                        "confidence_scores": {
                            "positive": doc.confidence_scores.positive,
                            "neutral": doc.confidence_scores.neutral,
                            "negative": doc.confidence_scores.negative
                        },
                        "sentences": []
                    }
                    
                    # Analyze sentence-level sentiment
                    for sentence in doc.sentences:
                        sentiment_data["sentences"].append({
                            "text": sentence.text,
                            "sentiment": sentence.sentiment,
                            "confidence_scores": {
                                "positive": sentence.confidence_scores.positive,
                                "neutral": sentence.confidence_scores.neutral,
                                "negative": sentence.confidence_scores.negative
                            }
                        })
                    
                    # Extract opinions if available
                    if hasattr(doc, 'mined_opinions'):
                        sentiment_data["opinions"] = [
                            {
                                "aspect": opinion.aspect.text,
                                "sentiment": opinion.aspect.sentiment,
                                "confidence": opinion.aspect.confidence_scores.positive
                            }
                            for opinion in doc.mined_opinions
                        ]
                    
                    return sentiment_data
            return {}
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {}
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities"""
        try:
            documents = [{"id": "1", "text": text}]
            response = self.text_analytics_client.recognize_entities(documents)
            
            entities = []
            for doc in response:
                if not doc.is_error:
                    for entity in doc.entities:
                        # Focus on legally relevant entities
                        if self._is_legal_relevant_entity(entity):
                            entities.append({
                                "text": entity.text,
                                "category": entity.category,
                                "subcategory": getattr(entity, 'subcategory', None),
                                "confidence_score": entity.confidence_score,
                                "offset": entity.offset,
                                "length": entity.length
                            })
            
            # Sort by confidence score
            entities.sort(key=lambda x: x["confidence_score"], reverse=True)
            return entities[:15]  # Limit to top 15
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language and confidence"""
        try:
            documents = [{"id": "1", "text": text}]
            response = self.text_analytics_client.detect_language(documents)
            
            for doc in response:
                if not doc.is_error:
                    primary_language = doc.primary_language
                    return {
                        "language": primary_language.name,
                        "iso_code": primary_language.iso6391_name,
                        "confidence_score": primary_language.confidence_score,
                        "is_english": primary_language.iso6391_name == "en"
                    }
            return {}
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {}
    
    def _extract_legal_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract custom legal entities using pattern matching"""
        legal_entities = []
        text_lower = text.lower()
        
        # Legal roles and parties
        party_patterns = {
            "contracting_party": r'\b(party|parties|contractor|client|customer|vendor|supplier)\b',
            "legal_entity": r'\b(corporation|company|llc|inc|ltd|partnership)\b',
            "person_role": r'\b(employee|employer|agent|representative|assignee)\b'
        }
        
        for category, pattern in party_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                legal_entities.append({
                    "text": match,
                    "category": category,
                    "confidence": 0.8,
                    "type": "legal_role"
                })
        
        # Legal concepts
        concept_patterns = {
            "legal_obligation": r'\b(shall|must|required to|obligated to|duty to)\b',
            "legal_right": r'\b(may|entitled to|right to|privilege|authority)\b',
            "legal_consequence": r'\b(penalty|damages|forfeiture|termination|breach)\b'
        }
        
        for category, pattern in concept_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                legal_entities.append({
                    "text": match,
                    "category": category,
                    "confidence": 0.9,
                    "type": "legal_concept"
                })
        
        return legal_entities[:10]  # Limit results
    
    def _analyze_risk_indicators(self, 
                               key_phrases: List[str], 
                               sentiment: Dict[str, Any], 
                               entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk indicators from Azure insights"""
        
        risk_indicators = {
            "linguistic_complexity": min(1.0, len(key_phrases) / 10),  # Normalized
            "negative_sentiment_strength": sentiment.get("confidence_scores", {}).get("negative", 0),
            "entity_density": min(1.0, len(entities) / 20),  # Normalized
            "risk_keywords_detected": [],
            "concerning_entities": [],
            "phrase_risk_score": 0.0,
            "entity_risk_score": 0.0
        }
        
        # Check for concerning entities
        high_risk_categories = ["Money", "DateTime", "Quantity", "Person", "Organization"]
        concerning_entities = [
            entity for entity in entities 
            if entity["category"] in high_risk_categories
        ]
        risk_indicators["concerning_entities"] = concerning_entities
        risk_indicators["entity_risk_score"] = min(1.0, len(concerning_entities) / 5)
        
        # Check for risk keywords in key phrases
        risk_keywords = [
            "terminate", "penalty", "liable", "forfeit", "breach", "default", 
            "damages", "indemnify", "unlimited", "absolute", "immediate"
        ]
        
        risk_phrases = []
        for phrase in key_phrases:
            for keyword in risk_keywords:
                if keyword.lower() in phrase.lower():
                    risk_phrases.append(phrase)
                    break
        
        risk_indicators["risk_keywords_detected"] = risk_phrases
        risk_indicators["phrase_risk_score"] = min(1.0, len(risk_phrases) / 3)
        
        # Calculate overall Azure risk score
        risk_indicators["overall_azure_risk"] = (
            risk_indicators["negative_sentiment_strength"] * 0.3 +
            risk_indicators["phrase_risk_score"] * 0.4 +
            risk_indicators["entity_risk_score"] * 0.3
        )
        
        return risk_indicators
    
    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Assess the quality and characteristics of the text"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "readability_score": self._calculate_readability(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(words) / max(1, len(sentences)),
            "has_complex_punctuation": any(char in text for char in [';', ':', '(', ')']),
            "formality_level": self._assess_formality(text)
        }
    
    def _calculate_azure_complexity(self, text: str) -> Dict[str, float]:
        """Calculate complexity metrics using Azure insights"""
        return {
            "structural_complexity": min(1.0, text.count(',') / 10),
            "legal_density": min(1.0, len([w for w in text.split() if self._is_legal_term(w)]) / 20),
            "modal_verb_density": min(1.0, len(re.findall(r'\b(shall|must|may|will|should)\b', text.lower())) / 5)
        }
    
    def _generate_azure_rationale(self, azure_insights: Dict[str, Any]) -> str:
        """Generate rationale based on Azure insights"""
        
        rationale_parts = []
        
        # Key phrases analysis
        key_phrases = azure_insights.get("key_phrases", [])
        if key_phrases:
            rationale_parts.append(
                f"Azure identified {len(key_phrases)} key legal phrases: {', '.join(key_phrases[:3])}"
            )
        
        # Sentiment analysis
        sentiment = azure_insights.get("sentiment_analysis", {})
        if sentiment:
            overall_sentiment = sentiment.get("overall_sentiment", "")
            negative_confidence = sentiment.get("confidence_scores", {}).get("negative", 0)
            
            if negative_confidence > 0.7:
                rationale_parts.append(
                    f"High negative sentiment detected ({negative_confidence:.1%}) indicating potentially unfavorable terms"
                )
            elif overall_sentiment == "negative":
                rationale_parts.append("Negative sentiment detected in legal language")
        
        # Entity analysis
        entities = azure_insights.get("entities", [])
        concerning_entities = [e for e in entities if e["category"] in ["Money", "DateTime", "Quantity"]]
        if concerning_entities:
            rationale_parts.append(
                f"Found {len(concerning_entities)} financial/temporal entities that may indicate penalties or strict deadlines"
            )
        
        # Risk indicators
        risk_indicators = azure_insights.get("risk_indicators", {})
        overall_risk = risk_indicators.get("overall_azure_risk", 0)
        if overall_risk > 0.6:
            rationale_parts.append(
                f"Azure risk analysis score: {overall_risk:.1%} indicating elevated concern level"
            )
        
        # Text quality insights
        text_quality = azure_insights.get("text_quality", {})
        formality = text_quality.get("formality_level", "")
        if formality == "highly_formal":
            rationale_parts.append("Highly formal legal language detected, increasing interpretation complexity")
        
        # Legal entities
        legal_entities = azure_insights.get("legal_entities", [])
        obligation_entities = [e for e in legal_entities if e["category"] == "legal_obligation"]
        if len(obligation_entities) > 2:
            rationale_parts.append(
                f"Multiple legal obligations detected ({len(obligation_entities)}), indicating complex duty structure"
            )
        
        return "- " + "\n- ".join(rationale_parts) if rationale_parts else "No significant additional risk indicators detected"
    
    def _calculate_azure_confidence_boost(self, azure_insights: Dict[str, Any]) -> float:
        """Calculate confidence boost from Azure analysis"""
        base_boost = 0.0
        
        # Boost confidence if multiple indicators align
        risk_indicators = azure_insights.get("risk_indicators", {})
        
        if risk_indicators.get("overall_azure_risk", 0) > 0.7:
            base_boost += 0.15
        elif risk_indicators.get("overall_azure_risk", 0) > 0.5:
            base_boost += 0.1
        
        # Boost for entity detection accuracy
        entities = azure_insights.get("entities", [])
        high_confidence_entities = [e for e in entities if e["confidence_score"] > 0.8]
        if len(high_confidence_entities) > 3:
            base_boost += 0.05
        
        # Boost for language quality
        language_info = azure_insights.get("language_info", {})
        if language_info.get("confidence_score", 0) > 0.9:
            base_boost += 0.03
        
        return min(0.2, base_boost)  # Cap at 20% boost
    
    def _is_legal_relevant_phrase(self, phrase: str) -> bool:
        """Check if phrase is legally relevant"""
        legal_keywords = [
            "contract", "agreement", "party", "obligation", "liability", "damages",
            "termination", "breach", "default", "penalty", "indemnity", "warranty",
            "representation", "covenant", "condition", "provision", "clause",
            "intellectual property", "confidential", "proprietary", "assignment"
        ]
        return any(keyword in phrase.lower() for keyword in legal_keywords)
    
    def _is_legal_relevant_entity(self, entity) -> bool:
        """Check if entity is legally relevant"""
        relevant_categories = [
            "Person", "Organization", "Location", "Money", "DateTime", 
            "Quantity", "Percentage", "Ordinal", "Cardinal"
        ]
        return entity.category in relevant_categories and entity.confidence_score > 0.5
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score"""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability approximation (inverse relationship)
        readability = max(0.0, 1.0 - (avg_sentence_length / 30))
        return readability
    
    def _assess_formality(self, text: str) -> str:
        """Assess formality level of text"""
        formal_indicators = [
            "whereas", "herein", "thereof", "hereof", "aforementioned",
            "notwithstanding", "pursuant", "heretofore", "hereafter"
        ]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())
        
        if formal_count >= 3:
            return "highly_formal"
        elif formal_count >= 1:
            return "formal"
        else:
            return "standard"
    
    def _is_legal_term(self, word: str) -> bool:
        """Check if word is a legal term"""
        legal_terms = [
            "contract", "agreement", "party", "liability", "damages", "breach",
            "termination", "warranty", "indemnity", "covenant", "provision"
        ]
        return word.lower() in legal_terms

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about Azure service usage"""
        return {
            "cached_analyses": len(self.legal_entities_cache),
            "service_available": self.text_analytics_client is not None,
            "cache_hit_rate": len(self.sentiment_cache) / max(1, len(self.legal_entities_cache))
        }

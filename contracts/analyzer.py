import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import google.generativeai as genai
from contracts.patterns import ContractPatterns
from contracts.nlp_pipeline import LightweightNLPPipeline
from contracts.explainability import ExplainabilityEngine, ComplianceReference
import os
from azure_services.explainability_service import AzureExplainabilityService

logger = logging.getLogger(__name__)

@dataclass
class RiskAnalysisResult:
    """Structured result for risk analysis"""
    sentence: str
    risk_type: str
    risk_category: str
    risk_level: str
    confidence_score: float
    reasoning: str
    section: str = ""
    ai_response: str = ""
    explanation: Optional[Dict[str, Any]] = None

class HybridContractAnalyzer:
    """Enhanced contract analyzer with explainability and compliance reporting"""
    
    def __init__(self):
        self.patterns = ContractPatterns()
        self.nlp_pipeline = LightweightNLPPipeline()
        self.gemini_client = self._initialize_gemini()
        # Add explainability components
        self.explainability_engine = ExplainabilityEngine()
        self.azure_service = AzureExplainabilityService()
        
        # Analysis configuration
        self.config = {
            "enable_ai_enhancement": True,
            "enable_azure_insights": True,
            "confidence_threshold": 0.6,
            "max_ai_requests": 10,
            "enable_compliance_checking": True
        }
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "ai_enhanced_analyses": 0,
            "azure_enhanced_analyses": 0,
            "compliance_checks": 0,
            "average_confidence": 0.0
        }
        
    def _initialize_gemini(self) -> Optional[genai.GenerativeModel]:
        """Initialize Gemini AI client"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("Google API key not found")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Test the model
            test_response = model.generate_content("Test")
            logger.info("Gemini AI client initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None

    async def analyze_contract(self, text: str, options: Optional[Dict] = None) -> Dict:
        """Enhanced contract analysis with full explainability and compliance reporting"""
        start_time = datetime.now()
        
        try:
            # Update configuration if options provided
            if options:
                self.config.update(options)
            
            logger.info(f"Starting enhanced contract analysis (length: {len(text)} chars)")
            
            # Step 1: Section identification and preprocessing
            sections = self._identify_sections(text)
            processed_text = self._preprocess_text(text)
            
            # Step 2: Multi-layered risk analysis
            rule_based_risks = await self._analyze_with_rules(processed_text, sections)
            nlp_enhanced_risks = await self._enhance_with_nlp(rule_based_risks, processed_text)
            
            # Step 3: AI enhancement (if enabled)
            if self.config["enable_ai_enhancement"] and self.gemini_client:
                ai_enhanced_risks = await self._enhance_with_ai(nlp_enhanced_risks, processed_text)
            else:
                ai_enhanced_risks = nlp_enhanced_risks
            
            # Step 4: Filter by confidence threshold
            significant_risks = [
                risk for risk in ai_enhanced_risks 
                if risk.confidence_score >= self.config["confidence_threshold"]
            ]
            
            # Step 5: Generate explanations for each significant risk
            explained_risks = []
            for risk in significant_risks:
                try:
                    # Generate base explanation
                    explanation = self.explainability_engine.generate_explanation(
                        risk_item=asdict(risk),
                        analysis_method="hybrid",
                        ai_response=risk.ai_response
                    )
                    
                    # Enhance with Azure (if available and enabled)
                    if self.config["enable_azure_insights"] and self.azure_service.text_analytics_client:
                        explanation = await self.azure_service.enhance_explanation_with_azure(
                            asdict(risk), explanation
                        )
                        self.analysis_stats["azure_enhanced_analyses"] += 1
                    
                    # Add explanation to risk
                    risk.explanation = explanation
                    explained_risks.append(risk)
                    
                except Exception as e:
                    logger.error(f"Failed to generate explanation for risk: {e}")
                    explained_risks.append(risk)  # Add without explanation
            
            # Step 6: Generate compliance report
            compliance_report = await self._generate_compliance_report(explained_risks, text)
            
            # Step 7: Generate summary and recommendations
            analysis_summary = self._generate_analysis_summary(explained_risks)
            recommendations = self._generate_recommendations(explained_risks, compliance_report)
            
            # Step 8: Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(explained_risks, text)
            
            # Update statistics
            self._update_analysis_stats(explained_risks)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Contract analysis completed in {analysis_time:.2f} seconds")
            
            return {
                "analyses": [asdict(risk) for risk in explained_risks],
                "summary": analysis_summary,
                "sections": sections,
                "recommendations": recommendations,
                "compliance_report": compliance_report,
                "overall_metrics": overall_metrics,
                "explainability_metadata": {
                    "explanation_confidence": self._calculate_explanation_confidence(explained_risks),
                    "regulatory_coverage": len(set([
                        ref["regulation_name"] 
                        for risk in explained_risks 
                        for ref in risk.explanation.get("regulatory_compliance", []) if risk.explanation
                    ])),
                    "analysis_methods_used": self._get_analysis_methods_used(explained_risks),
                    "azure_enhanced": self.analysis_stats["azure_enhanced_analyses"] > 0,
                    "ai_enhanced": self.analysis_stats["ai_enhanced_analyses"] > 0
                },
                "performance_metrics": {
                    "analysis_time_seconds": analysis_time,
                    "risks_analyzed": len(explained_risks),
                    "total_sentences_processed": len(re.split(r'[.!?]+', text)),
                    "confidence_distribution": self._calculate_confidence_distribution(explained_risks)
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced contract analysis failed: {e}")
            raise
    
    async def _analyze_with_rules(self, text: str, sections: List[Dict]) -> List[RiskAnalysisResult]:
        """Analyze using rule-based patterns"""
        risks = []
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            # Check each pattern category
            for pattern_data in self.patterns.get_all_patterns():
                category = pattern_data["category"]
                patterns = pattern_data["patterns"]
                confidence = pattern_data.get("confidence", 0.8)
                
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Determine risk level based on pattern and context
                        risk_level = self._determine_risk_level(sentence, pattern, category)
                        
                        # Find which section this sentence belongs to
                        section_name = self._find_sentence_section(sentence, sections)
                        
                        risk = RiskAnalysisResult(
                            sentence=sentence.strip(),
                            risk_type=category,
                            risk_category=category,
                            risk_level=risk_level,
                            confidence_score=confidence,
                            reasoning=f"Matched pattern: {pattern}",
                            section=section_name
                        )
                        
                        # Avoid duplicates
                        if not self._is_duplicate_risk(risk, risks):
                            risks.append(risk)
                        break  # Don't match multiple patterns for same sentence
        
        logger.info(f"Rule-based analysis found {len(risks)} potential risks")
        return risks
    
    async def _enhance_with_nlp(self, base_risks: List[RiskAnalysisResult], text: str) -> List[RiskAnalysisResult]:
        """Enhance analysis using NLP pipeline"""
        enhanced_risks = base_risks.copy()
        
        # Analyze entire text for context
        doc_analysis = self.nlp_pipeline.analyze_document_sentiment(text)
        doc_sentiment = doc_analysis.get("overall_sentiment", "neutral")
        
        # Enhance each risk with NLP insights
        for risk in enhanced_risks:
            sentence_analysis = self.nlp_pipeline.analyze_sentence(risk.sentence)
            
            # Adjust confidence based on NLP analysis
            nlp_confidence = sentence_analysis.get("risk_confidence", 0.5)
            combined_confidence = (risk.confidence_score * 0.7) + (nlp_confidence * 0.3)
            risk.confidence_score = min(0.95, combined_confidence)
            
            # Enhance reasoning with NLP insights
            sentiment = sentence_analysis.get("sentiment", "neutral")
            if sentiment == "negative":
                risk.reasoning += " | NLP: Negative sentiment detected"
            
            # Check for complex linguistic structures
            complexity = sentence_analysis.get("complexity_score", 0)
            if complexity > 0.7:
                risk.reasoning += " | NLP: High linguistic complexity"
        
        # Look for additional risks using NLP
        additional_risks = await self._find_nlp_risks(text, doc_analysis)
        enhanced_risks.extend(additional_risks)
        
        logger.info(f"NLP enhancement added {len(additional_risks)} new risks")
        return enhanced_risks
    
    async def _enhance_with_ai(self, nlp_risks: List[RiskAnalysisResult], text: str) -> List[RiskAnalysisResult]:
        """Enhance analysis using AI (Gemini)"""
        if not self.gemini_client:
            return nlp_risks
        
        enhanced_risks = nlp_risks.copy()
        ai_request_count = 0
        max_requests = self.config["max_ai_requests"]
        
        # Sort risks by confidence to prioritize AI analysis
        sorted_risks = sorted(enhanced_risks, key=lambda x: x.confidence_score, reverse=True)
        
        for risk in sorted_risks:
            if ai_request_count >= max_requests:
                break
            
            try:
                # Generate AI prompt for this specific risk
                ai_prompt = self._create_ai_prompt(risk, text[:2000])  # Limit context
                
                # Get AI analysis
                ai_response = self.gemini_client.generate_content(ai_prompt)
                risk.ai_response = ai_response.text
                
                # Parse AI response to adjust risk assessment
                ai_insights = self._parse_ai_response(ai_response.text)
                
                # Adjust confidence based on AI insights
                if ai_insights.get("confirms_risk", False):
                    risk.confidence_score = min(0.95, risk.confidence_score + 0.1)
                    risk.reasoning += f" | AI: {ai_insights.get('reason', 'Risk confirmed')}"
                
                # Update risk level if AI suggests different level
                suggested_level = ai_insights.get("suggested_level")
                if suggested_level and suggested_level != risk.risk_level:
                    risk.risk_level = suggested_level
                    risk.reasoning += f" | AI: Risk level adjusted to {suggested_level}"
                
                ai_request_count += 1
                self.analysis_stats["ai_enhanced_analyses"] += 1
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"AI enhancement failed for risk: {e}")
                continue
        
        # Look for AI-identified risks that weren't caught by rules/NLP
        if ai_request_count < max_requests:
            try:
                additional_ai_risks = await self._find_ai_risks(text, ai_request_count, max_requests)
                enhanced_risks.extend(additional_ai_risks)
            except Exception as e:
                logger.warning(f"AI additional risk detection failed: {e}")
        
        logger.info(f"AI enhancement processed {ai_request_count} risks")
        return enhanced_risks
    
    async def _find_nlp_risks(self, text: str, doc_analysis: Dict) -> List[RiskAnalysisResult]:
        """Find additional risks using NLP analysis"""
        additional_risks = []
        
        # Look for imbalanced power language
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if len(sentence.strip()) < 15:
                continue
                
            sentence_analysis = self.nlp_pipeline.analyze_sentence(sentence)
            
            # Check for power imbalance indicators
            if self._has_power_imbalance(sentence_analysis):
                risk = RiskAnalysisResult(
                    sentence=sentence.strip(),
                    risk_type="Power Imbalance Risk",
                    risk_category="contract_imbalance",
                    risk_level="MEDIUM",
                    confidence_score=0.7,
                    reasoning="NLP detected linguistic power imbalance",
                    section="General"
                )
                additional_risks.append(risk)
        
        return additional_risks
    
    async def _find_ai_risks(self, text: str, current_requests: int, max_requests: int) -> List[RiskAnalysisResult]:
        """Find additional risks using AI analysis"""
        if current_requests >= max_requests or not self.gemini_client:
            return []
        
        additional_risks = []
        
        try:
            # General contract review prompt
            general_prompt = f"""
            Analyze this contract text for any legal risks not immediately obvious:
            
            {text[:3000]}
            
            Focus on:
            1. Hidden obligations
            2. Ambiguous terms
            3. Potential conflicts
            4. Unusual provisions
            
            Format response as: RISK: [brief description] | SENTENCE: [relevant sentence] | LEVEL: [HIGH/MEDIUM/LOW]
            """
            
            response = self.gemini_client.generate_content(general_prompt)
            ai_risks = self._parse_general_ai_risks(response.text, text)
            additional_risks.extend(ai_risks)
            
        except Exception as e:
            logger.warning(f"General AI risk detection failed: {e}")
        
        return additional_risks
    
    async def _generate_compliance_report(self, risks: List[RiskAnalysisResult], full_text: str) -> Dict:
        """Generate comprehensive regulatory compliance report"""
        
        if not self.config["enable_compliance_checking"]:
            return {"compliance_checking_disabled": True}
        
        all_compliance_refs = []
        risk_by_regulation = {}
        
        # Collect all compliance references from risk explanations
        for risk in risks:
            if not risk.explanation:
                continue
                
            compliance_refs = risk.explanation.get("regulatory_compliance", [])
            for ref in compliance_refs:
                all_compliance_refs.append(ref)
                
                # Group risks by regulation
                reg_name = ref["regulation_name"]
                if reg_name not in risk_by_regulation:
                    risk_by_regulation[reg_name] = []
                risk_by_regulation[reg_name].append({
                    "risk": risk,
                    "compliance_ref": ref
                })
        
        # Calculate compliance metrics
        compliance_levels = [ref["compliance_level"] for ref in all_compliance_refs]
        non_compliant_count = compliance_levels.count("non_compliant")
        requires_review_count = compliance_levels.count("requires_review")
        compliant_count = compliance_levels.count("compliant")
        total_refs = len(compliance_levels)
        
        # Calculate overall compliance score
        if total_refs == 0:
            compliance_score = 0.85  # Default if no specific regulations identified
            compliance_status = "no_specific_regulations"
        else:
            # Weighted scoring: non-compliant = -0.4, requires_review = -0.2, compliant = +0.1
            score_impact = (non_compliant_count * -0.4) + (requires_review_count * -0.2) + (compliant_count * 0.1)
            compliance_score = max(0.0, min(1.0, 0.8 + (score_impact / total_refs)))
            compliance_status = self._determine_compliance_status(compliance_score)
        
        # Generate detailed analysis by regulation
        detailed_analysis = {}
        for reg_name, reg_data in risk_by_regulation.items():
            reg_risks = [item["risk"] for item in reg_data]
            reg_refs = [item["compliance_ref"] for item in reg_data]
            
            detailed_analysis[reg_name] = {
                "affected_risks": len(reg_risks),
                "compliance_items": reg_refs,
                "risk_sentences": [risk.sentence[:100] + "..." for risk in reg_risks[:3]],
                "overall_impact": self._calculate_regulation_impact(reg_refs)
            }
        
        # Generate recommendations and next steps
        next_steps = self._generate_compliance_next_steps(all_compliance_refs, compliance_score)
        priority_actions = self._generate_priority_actions(all_compliance_refs)
        
        self.analysis_stats["compliance_checks"] += 1
        
        return {
            "overall_compliance_score": compliance_score,
            "compliance_status": compliance_status,
            "regulations_analyzed": list(risk_by_regulation.keys()),
            "compliance_summary": {
                "compliant_items": compliant_count,
                "non_compliant_items": non_compliant_count,
                "review_required_items": requires_review_count,
                "total_items_analyzed": total_refs
            },
            "detailed_compliance": detailed_analysis,
            "report_summary": self._generate_compliance_summary(
                compliance_score, compliance_status, non_compliant_count, requires_review_count
            ),
            "next_steps": next_steps,
            "priority_actions": priority_actions,
            "compliance_recommendations": self._generate_compliance_recommendations(all_compliance_refs),
            "regulatory_gaps": self._identify_regulatory_gaps(full_text, risk_by_regulation),
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "total_risks_with_compliance": len([r for r in risks if r.explanation and r.explanation.get("regulatory_compliance")]),
                "compliance_analysis_enabled": True
            }
        }
    
    def _generate_analysis_summary(self, risks: List[RiskAnalysisResult]) -> Dict:
        """Generate comprehensive analysis summary"""
        if not risks:
            return {
                "total_risks": 0,
                "risk_distribution": {},
                "overall_assessment": "No significant risks detected",
                "confidence": "N/A"
            }
        
        # Risk level distribution
        risk_levels = [risk.risk_level for risk in risks]
        risk_distribution = {
            "HIGH": risk_levels.count("HIGH"),
            "MEDIUM": risk_levels.count("MEDIUM"),
            "LOW": risk_levels.count("LOW")
        }
        
        # Category analysis
        categories = {}
        for risk in risks:
            category = risk.risk_category
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Overall assessment
        high_risk_count = risk_distribution["HIGH"]
        medium_risk_count = risk_distribution["MEDIUM"]
        
        if high_risk_count > 0:
            overall_assessment = f"Contract contains {high_risk_count} high-risk clauses requiring immediate attention"
        elif medium_risk_count > 2:
            overall_assessment = f"Contract has multiple medium-risk areas ({medium_risk_count}) that should be reviewed"
        else:
            overall_assessment = "Contract appears to have acceptable risk levels with standard precautions"
        
        # Calculate average confidence
        avg_confidence = sum(risk.confidence_score for risk in risks) / len(risks)
        
        return {
            "total_risks": len(risks),
            "risk_distribution": risk_distribution,
            "category_breakdown": categories,
            "overall_assessment": overall_assessment,
            "average_confidence": avg_confidence,
            "highest_risk_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _generate_recommendations(self, risks: List[RiskAnalysisResult], compliance_report: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High-priority recommendations based on high risks
        high_risks = [risk for risk in risks if risk.risk_level == "HIGH"]
        if high_risks:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "High Risk Mitigation",
                "title": f"Address {len(high_risks)} Critical Risk Issues",
                "description": "These clauses pose significant legal or financial risks and require immediate attention before contract execution.",
                "actions": [
                    "Schedule immediate legal review",
                    "Consider renegotiating problematic clauses",
                    "Obtain additional insurance if liability risks are unavoidable",
                    "Document risk acceptance rationale if proceeding"
                ],
                "affected_clauses": [risk.sentence[:100] + "..." for risk in high_risks[:3]]
            })
        
        # Compliance-based recommendations
        compliance_score = compliance_report.get("overall_compliance_score", 1.0)
        if compliance_score < 0.7:
            recommendations.append({
                "priority": "HIGH",
                "category": "Regulatory Compliance",
                "title": "Address Regulatory Compliance Issues",
                "description": f"Contract compliance score is {compliance_score:.1%}, indicating potential regulatory violations.",
                "actions": compliance_report.get("next_steps", []),
                "affected_regulations": compliance_report.get("regulations_analyzed", [])
            })
        
        # Category-specific recommendations
        category_counts = {}
        for risk in risks:
            category = risk.risk_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in category_counts.items():
            if count >= 3:  # Multiple risks in same category
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": f"{category} Management",
                    "title": f"Multiple {category} Issues Detected",
                    "description": f"Found {count} instances of {category.lower()}, suggesting systematic issues in this area.",
                    "actions": [
                        f"Review all {category.lower()} provisions systematically",
                        f"Consider standard protective language for {category.lower()}",
                        "Consult specialist attorney if needed"
                    ]
                })
        
        # General best practices
        if len(risks) > 0:
            recommendations.append({
                "priority": "LOW",
                "category": "Best Practices",
                "title": "General Contract Management",
                "description": "Implement standard risk management practices for ongoing contract administration.",
                "actions": [
                    "Establish regular contract review cycles",
                    "Maintain documentation of all contract modifications",
                    "Set up monitoring for contract performance obligations",
                    "Create escalation procedures for potential disputes"
                ]
            })
        
        return recommendations
    
    def _calculate_overall_metrics(self, risks: List[RiskAnalysisResult], text: str) -> Dict:
        """Calculate overall contract analysis metrics"""
        
        # Text complexity metrics
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        # Risk density
        risk_density = len(risks) / len(sentences) if sentences else 0
        
        # Confidence metrics
        confidence_scores = [risk.confidence_score for risk in risks]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        confidence_std = self._calculate_std_dev(confidence_scores) if len(confidence_scores) > 1 else 0
        
        # Risk severity score (weighted)
        severity_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        severity_score = sum(
            severity_weights.get(risk.risk_level, 1) for risk in risks
        ) / max(1, len(risks))
        
        # Contract complexity
        complexity_score = self._calculate_contract_complexity(text)
        
        # Power balance assessment
        power_balance = self._assess_power_balance(text)
        
        return {
            "risk_density": risk_density,
            "average_confidence": avg_confidence,
            "confidence_consistency": 1.0 - min(1.0, confidence_std),
            "severity_score": severity_score,
            "complexity_score": complexity_score,
            "power_balance_score": power_balance,
            "readability_score": self._calculate_readability_score(text),
            "legal_language_density": self._calculate_legal_density(text),
            "overall_risk_score": min(1.0, (risk_density * 0.3) + (severity_score / 3 * 0.7))
        }

    # Helper methods for analysis
    def _determine_risk_level(self, sentence: str, pattern: str, category: str) -> str:
        """Determine risk level based on context and pattern"""
        sentence_lower = sentence.lower()
        
        # High risk indicators
        high_risk_terms = [
            'unlimited', 'absolute', 'entire', 'forfeit', 'immediately', 
            'without notice', 'at will', 'sole discretion'
        ]
        
        # Check for high risk terms
        if any(term in sentence_lower for term in high_risk_terms):
            return "HIGH"
        
        # Category-specific risk assessment
        if category.lower() == "liability risk":
            if any(term in sentence_lower for term in ['damages', 'liable', 'responsible']):
                return "HIGH" if 'unlimited' in sentence_lower else "MEDIUM"
        
        return "MEDIUM"  # Default
    
    def _find_sentence_section(self, sentence: str, sections: List[Dict]) -> str:
        """Find which section a sentence belongs to"""
        for section in sections:
            if sentence.strip() in section.get("content", ""):
                return section.get("title", "General")
        return "General"
    
    def _is_duplicate_risk(self, new_risk: RiskAnalysisResult, existing_risks: List[RiskAnalysisResult]) -> bool:
        """Check if risk is already identified"""
        for existing in existing_risks:
            if (existing.sentence.strip() == new_risk.sentence.strip() and 
                existing.risk_category == new_risk.risk_category):
                return True
        return False
    
    def _create_ai_prompt(self, risk: RiskAnalysisResult, context: str) -> str:
        """Create AI prompt for risk analysis"""
        return f"""
        Analyze this legal clause for potential risks:
        
        CLAUSE: "{risk.sentence}"
        IDENTIFIED RISK TYPE: {risk.risk_type}
        CURRENT RISK LEVEL: {risk.risk_level}
        
        CONTEXT: {context}
        
        Please:
        1. Confirm if this is actually a risk (yes/no)
        2. Suggest appropriate risk level (HIGH/MEDIUM/LOW)
        3. Explain the specific legal concern in one sentence
        4. Suggest if risk level should be adjusted
        
        Respond in format:
        CONFIRMS_RISK: [yes/no]
        SUGGESTED_LEVEL: [HIGH/MEDIUM/LOW]
        REASON: [brief explanation]
        """
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response for insights"""
        insights = {
            "confirms_risk": False,
            "suggested_level": None,
            "reason": ""
        }
        
        response_lower = response.lower()
        
        # Parse confirmation
        if "confirms_risk: yes" in response_lower or "yes" in response_lower[:50]:
            insights["confirms_risk"] = True
        
        # Parse suggested level
        if "high" in response_lower:
            insights["suggested_level"] = "HIGH"
        elif "medium" in response_lower:
            insights["suggested_level"] = "MEDIUM"
        elif "low" in response_lower:
            insights["suggested_level"] = "LOW"
        
        # Extract reason
        if "reason:" in response_lower:
            reason_start = response_lower.find("reason:") + 7
            reason = response[reason_start:reason_start+200].strip()
            insights["reason"] = reason
        
        return insights
    
    def _parse_general_ai_risks(self, response: str, text: str) -> List[RiskAnalysisResult]:
        """Parse general AI risk detection response"""
        risks = []
        lines = response.split('\n')
        
        for line in lines:
            if 'RISK:' in line.upper():
                try:
                    # Extract risk info from formatted response
                    risk_desc = self._extract_between(line, 'RISK:', 'SENTENCE:')
                    sentence = self._extract_between(line, 'SENTENCE:', 'LEVEL:')
                    level = self._extract_after(line, 'LEVEL:')
                    
                    if risk_desc and sentence:
                        risk = RiskAnalysisResult(
                            sentence=sentence.strip()[:500],
                            risk_type=risk_desc.strip(),
                            risk_category="AI Identified Risk",
                            risk_level=level.strip() if level else "MEDIUM",
                            confidence_score=0.75,
                            reasoning=f"AI identified: {risk_desc.strip()}",
                            section="AI Analysis"
                        )
                        risks.append(risk)
                except Exception as e:
                    logger.warning(f"Failed to parse AI risk: {e}")
                    continue
        
        return risks
    
    def _has_power_imbalance(self, sentence_analysis: Dict) -> bool:
        """Check if sentence shows power imbalance"""
        # This would be implemented based on NLP analysis results
        return sentence_analysis.get("power_imbalance_score", 0) > 0.7
    
    # Additional helper methods...
    def _identify_sections(self, text: str) -> List[Dict]:
        """Identify contract sections"""
        sections = []
        
        # Common section headers
        section_patterns = [
            r'\b(?:ARTICLE|SECTION)\s+\d+[.\s]+([A-Z\s]+)',
            r'\b\d+\.\s+([A-Z][A-Z\s]+)',
            r'\b([A-Z][A-Z\s]{5,})\b(?=\s*\n|\s*\.)'
        ]
        
        lines = text.split('\n')
        current_section = {"title": "Preamble", "content": "", "start_line": 0}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.search(pattern, line_stripped)
                if match:
                    # Save current section
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": match.group(1).strip().title(),
                        "content": "",
                        "start_line": i
                    }
                    is_header = True
                    break
            
            if not is_header:
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections if sections else [{"title": "Full Document", "content": text, "start_line": 0}]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess contract text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotation marks
        text = re.sub(r'["""]', '"', text)
        
        # Fix common contract formatting issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before caps
        
        return text.strip()
    
    # Statistics and utility methods
    def _calculate_explanation_confidence(self, risks: List[RiskAnalysisResult]) -> float:
        """Calculate overall confidence in explanations"""
        if not risks:
            return 0.0
        
        confidence_scores = []
        for risk in risks:
            if risk.explanation and risk.explanation.get("evidence_chain"):
                evidence_chain = risk.explanation["evidence_chain"]
                avg_confidence = sum(e.confidence for e in evidence_chain) / len(evidence_chain)
                confidence_scores.append(avg_confidence)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    def _get_analysis_methods_used(self, risks: List[RiskAnalysisResult]) -> List[str]:
        """Get list of analysis methods used"""
        methods = set()
        for risk in risks:
            if risk.explanation and risk.explanation.get("evidence_chain"):
                for evidence in risk.explanation["evidence_chain"]:
                    methods.add(evidence.evidence_type.value)
        return list(methods)
    
    def _calculate_confidence_distribution(self, risks: List[RiskAnalysisResult]) -> Dict[str, int]:
        """Calculate distribution of confidence scores"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for risk in risks:
            if risk.confidence_score >= 0.8:
                distribution["high"] += 1
            elif risk.confidence_score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _update_analysis_stats(self, risks: List[RiskAnalysisResult]) -> None:
        """Update analysis statistics"""
        self.analysis_stats["total_analyses"] += 1
        
        confidence_scores = [risk.confidence_score for risk in risks]
        if confidence_scores:
            self.analysis_stats["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    # Additional compliance helper methods
    def _determine_compliance_status(self, score: float) -> str:
        """Determine overall compliance status"""
        if score >= 0.9:
            return "compliant"
        elif score >= 0.7:
            return "mostly_compliant"
        elif score >= 0.5:
            return "requires_review"
        else:
            return "non_compliant"
    
    def _calculate_regulation_impact(self, refs: List[Dict]) -> str:
        """Calculate impact level for a regulation"""
        non_compliant = sum(1 for ref in refs if ref["compliance_level"] == "non_compliant")
        if non_compliant > 0:
            return "HIGH"
        elif len(refs) > 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_compliance_next_steps(self, refs: List[Dict], score: float) -> List[str]:
        """Generate next steps for compliance"""
        next_steps = []
        
        non_compliant_refs = [ref for ref in refs if ref["compliance_level"] == "non_compliant"]
        if non_compliant_refs:
            next_steps.append("Immediately address non-compliant clauses before contract execution")
            next_steps.append("Consult legal counsel for compliance violation remediation")
        
        review_refs = [ref for ref in refs if ref["compliance_level"] == "requires_review"]
        if review_refs:
            next_steps.append("Schedule legal review for clauses requiring regulatory assessment")
        
        if score < 0.7:
            next_steps.append("Conduct comprehensive regulatory compliance audit")
        
        if not next_steps:
            next_steps.append("Proceed with standard contract review process")
        
        return next_steps
    
    def _generate_priority_actions(self, refs: List[Dict]) -> List[Dict]:
        """Generate priority actions for compliance"""
        actions = []
        
        # Group by regulation
        by_regulation = {}
        for ref in refs:
            reg = ref["regulation_name"]
            if reg not in by_regulation:
                by_regulation[reg] = []
            by_regulation[reg].append(ref)
        
        # Create actions for each regulation with issues
        for reg_name, reg_refs in by_regulation.items():
            non_compliant = [r for r in reg_refs if r["compliance_level"] == "non_compliant"]
            if non_compliant:
                actions.append({
                    "priority": "CRITICAL",
                    "regulation": reg_name,
                    "action": f"Address {len(non_compliant)} non-compliant provisions",
                    "details": [ref["recommendation"] for ref in non_compliant[:3]]
                })
        
        return actions
    
    def _generate_compliance_recommendations(self, refs: List[Dict]) -> List[str]:
        """Generate compliance-specific recommendations"""
        recommendations = []
        
        # Get unique recommendations
        unique_recs = list(set(ref["recommendation"] for ref in refs))
        
        for rec in unique_recs[:5]:  # Limit to top 5
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_compliance_summary(self, score: float, status: str, non_compliant: int, review_required: int) -> str:
        """Generate compliance summary"""
        if status == "compliant":
            return f"Contract demonstrates strong regulatory compliance (score: {score:.1%})"
        elif status == "mostly_compliant":
            return f"Contract is generally compliant with {review_required} items requiring review (score: {score:.1%})"
        else:
            return f"Contract has significant compliance concerns: {non_compliant} non-compliant items, {review_required} requiring review (score: {score:.1%})"
    
    def _identify_regulatory_gaps(self, text: str, analyzed_regs: Dict) -> List[str]:
        """Identify potential regulatory gaps"""
        gaps = []
        text_lower = text.lower()
        
        # Check for common regulatory areas not covered
        if "data" in text_lower and "privacy" in text_lower and "GDPR" not in analyzed_regs:
            gaps.append("GDPR compliance for data processing provisions")
        
        if "employment" in text_lower and "labor" not in analyzed_regs:
            gaps.append("Labor law compliance for employment terms")
        
        if "intellectual property" in text_lower and "copyright" not in analyzed_regs:
            gaps.append("Intellectual property law compliance")
        
        return gaps
    
    # Utility helper methods
    def _extract_between(self, text: str, start: str, end: str) -> Optional[str]:
        """Extract text between two markers"""
        try:
            start_idx = text.upper().find(start.upper()) + len(start)
            end_idx = text.upper().find(end.upper())
            if start_idx > len(start) - 1 and end_idx > start_idx:
                return text[start_idx:end_idx].strip()
        except:
            pass
        return None
    
    def _extract_after(self, text: str, marker: str) -> Optional[str]:
        """Extract text after a marker"""
        try:
            start_idx = text.upper().find(marker.upper()) + len(marker)
            if start_idx > len(marker) - 1:
                return text[start_idx:].split('|')[0].strip()
        except:
            pass
        return None
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_contract_complexity(self, text: str) -> float:
        """Calculate contract complexity score"""
        # Simplified complexity calculation
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        avg_sentence_length = len(words) / max(1, len(sentences))
        legal_term_density = self._calculate_legal_density(text)
        
        complexity = min(1.0, (avg_sentence_length / 20) + legal_term_density)
        return complexity
    
    # def _assess_power_balance(self, text: str) -> float:
    #     """Assess power balance in contract (0=heavily favors one party, 1=balanced)"""
    #     text_lower = text.lower()
        
    #     # Count obligation words for each party
    #     party_a_obligations = len(re.findall(r'party\s+a\

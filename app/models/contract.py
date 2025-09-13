from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RiskAnalysis(BaseModel):
    sentence: str = Field(..., description="The problematic sentence")
    risk_category: str = Field(..., description="Category of risk")
    risk_level: str = Field(..., description="Risk severity level")
    risk_type: str = Field(..., description="Type of legal risk")
    description: str = Field(..., description="Detailed risk explanation")
    specific_concerns: List[str] = Field(default_factory=list)
    negotiation_strategies: List[str] = Field(default_factory=list)
    priority_score: int = Field(..., ge=1, le=10, description="Priority from 1-10")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    legal_concepts: List[str] = Field(default_factory=list)
    entities: List[Dict] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    alternative_language: str = Field(default="")
    cost_implications: str = Field(default="")

class ContractSection(BaseModel):
    title: str
    content: str
    risk_count: int
    section_type: str = Field(default="general")

class RiskSummary(BaseModel):
    total_risks: int
    critical_risk_count: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    overall_risk_level: str
    risk_distribution: Dict[str, int] = Field(default_factory=dict)

class AnalysisResponse(BaseModel):
    filename: str
    extracted_text: str
    analysis: List[RiskAnalysis]
    summary: RiskSummary
    sections: List[ContractSection]
    recommendations: List[str]
    overall_summary: str
    document_complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    party_power_balance: float = Field(default=0.5, ge=0.0, le=1.0)
    processing_time: Optional[float] = None

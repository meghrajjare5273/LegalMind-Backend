"""
Lightweight rule-based contract analyser with optional Gemini enrichment.
Only the interface touched by main.py is included; existing NLP helpers remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from cachetools import TTLCache

# Optional Gemini
try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore

# --------------------------------------------------------------------------- #
# Enum definitions
# --------------------------------------------------------------------------- #
class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ClauseType(Enum):
    TERMINATION = "termination"
    LIABILITY = "liability"
    PAYMENT = "payment"
    INTEREST_RATES = "interest_rates"
    DEFAULT_CONSEQUENCES = "default_consequences"
    COLLATERAL_SECURITY = "collateral_security"
    REPOSSESSION = "repossession"
    ARBITRATION = "arbitration"
    INSURANCE = "insurance"
    MOVEMENT_RESTRICTIONS = "movement_restrictions"
    DEATH_INCAPACITY = "death_incapacity"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    CONFIDENTIALITY = "confidentiality"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    GENERAL = "general"


# --------------------------------------------------------------------------- #
# Dataclass for risk items
# --------------------------------------------------------------------------- #
@dataclass
class RiskAnalysis:
    sentence: str
    risk_category: str
    risk_level: RiskLevel
    clause_type: ClauseType
    risk_explanation: str
    mitigation_strategies: List[str]
    negotiation_priority: int
    confidence_score: float
    legal_concepts: List[str]
    entities: List[Dict]
    specific_concerns: List[str] | None = None
    negotiation_tactics: List[str] | None = None
    alternative_language: str = ""
    cost_implications: str = ""


# --------------------------------------------------------------------------- #
# Core analyser (simplified; original NLP functions omitted for brevity)
# --------------------------------------------------------------------------- #
class LightweightContractAnalyzer:
    """Stub demonstrating the attributes used by main.py."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.suggestion_cache: TTLCache = TTLCache(maxsize=500, ttl=3_600)

        # if Gemini present, try to init
        api_key = "AIzaSyCWqH4CpR1EfWcmF-yiq26xrwxyooPcrDs"
        if genai and api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Gemini init failed: %s", exc)
                self.gemini_client = None
        else:
            self.gemini_client = None

    # --------------------------------------------------------------------- #
    # Public method used by main.py
    # --------------------------------------------------------------------- #
    def analyze_contract_advanced(self, text: str):
        """
        Extremely stripped-down demo:
        returns (list[RiskAnalysis], doc_profile)
        """
        # Very naive split; production code would have real NLP here
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        risks: List[RiskAnalysis] = []
        for sent in sentences[:10]:  # limit for demo
            risks.append(
                RiskAnalysis(
                    sentence=sent,
                    risk_category="General",
                    risk_level=RiskLevel.MEDIUM,
                    clause_type=ClauseType.GENERAL,
                    risk_explanation="Potential ambiguity detected",
                    mitigation_strategies=["Clarify language", "Consult counsel"],
                    negotiation_priority=3,
                    confidence_score=0.75,
                    legal_concepts=["Ambiguity"],
                    entities=[],
                )
            )

        # Dummy doc profile object with attributes referenced in main.py
        class _DocProfile:  # noqa: D401
            legal_complexity_score = 0.4
            party_power_balance = 0.5
            compliance_coverage: Dict[str, float] = {}

        return risks, _DocProfile()

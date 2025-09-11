class ContractPatterns:
    """Comprehensive contract risk patterns with fixed regex"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.risk_keywords = self._initialize_risk_keywords()
    
    def _initialize_patterns(self):
        return [
            {
                "category": "Termination Risk",
                "clause_type": "TERMINATION", 
                "risk_level": "HIGH",
                "patterns": [
                    r"terminate.*without.*notice",
                    r"terminate.*at.*will",
                    r"terminate.*sole.*discretion",
                    r"immediate.*termination",
                    r"terminate.*without.*cause",
                    r"end.*agreement.*immediately"
                ],
                "description": "Contract allows termination with minimal notice or justification",
                "concerns": [
                    "Sudden contract termination risk",
                    "Lack of adequate notice period", 
                    "No protection against arbitrary termination"
                ],
                "strategies": [
                    "Negotiate minimum notice period",
                    "Add termination for cause requirements",
                    "Include cure period provisions"
                ],
                "priority": 8,
                "confidence": 0.9
            },
            {
                "category": "Liability Risk",
                "clause_type": "LIABILITY",
                "risk_level": "CRITICAL",
                "patterns": [
                    r"unlimited.*liability",
                    r"liable.*for.*all.*damages",
                    r"responsible.*for.*any.*loss",
                    r"indemnify.*against.*all.*claims",
                    r"hold.*harmless.*from.*any",
                    r"total.*liability.*exceeds"
                ],
                "description": "Excessive liability exposure with unlimited financial risk",
                "concerns": [
                    "Unlimited financial exposure",
                    "Disproportionate risk allocation", 
                    "Potential bankruptcy risk"
                ],
                "strategies": [
                    "Cap liability at contract value",
                    "Exclude consequential damages",
                    "Negotiate mutual indemnification"
                ],
                "priority": 10,
                "confidence": 0.95
            },
            {
                "category": "Payment Risk",
                "clause_type": "PAYMENT",
                "risk_level": "HIGH", 
                "patterns": [
                    r"payment.*due.*immediately",
                    r"no.*refund",
                    r"non-refundable",
                    r"penalty.*for.*late.*payment",
                    r"interest.*rate.*of.*\d+%",
                    r"late.*fee.*of.*\$\d+"
                ],
                "description": "Harsh payment terms with penalties and no refund provisions",
                "concerns": [
                    "Immediate payment requirements",
                    "High penalty fees",
                    "No refund protection"
                ],
                "strategies": [
                    "Negotiate payment schedule",
                    "Reduce penalty rates", 
                    "Add refund provisions for cause"
                ],
                "priority": 7,
                "confidence": 0.85
            },
            {
                "category": "Intellectual Property Risk",
                "clause_type": "INTELLECTUAL_PROPERTY",
                "risk_level": "HIGH",
                "patterns": [
                    r"all.*rights.*belong.*to",
                    r"work.*for.*hire",
                    r"assign.*all.*intellectual.*property",
                    r"waive.*moral.*rights",
                    r"transfer.*all.*rights",
                    r"exclusive.*license.*to"
                ],
                "description": "Broad IP assignment with potential overreach",
                "concerns": [
                    "Loss of IP ownership",
                    "Overly broad assignment",
                    "No retained rights"
                ],
                "strategies": [
                    "Limit assignment to project-specific IP",
                    "Retain pre-existing IP rights",
                    "Negotiate shared ownership"
                ],
                "priority": 8,
                "confidence": 0.9
            },
            {
                "category": "Confidentiality Risk", 
                "clause_type": "CONFIDENTIALITY",
                "risk_level": "MEDIUM",
                "patterns": [
                    r"perpetual.*confidentiality",
                    r"confidential.*information.*includes.*all",
                    r"non-disclosure.*indefinitely",
                    r"confidentiality.*period.*of.*\d+.*years"
                ],
                "description": "Overly broad or perpetual confidentiality requirements",
                "concerns": [
                    "Indefinite confidentiality period",
                    "Overly broad definition",
                    "Unfair information restrictions"
                ],
                "strategies": [
                    "Limit confidentiality period",
                    "Define confidential information clearly",
                    "Add standard exceptions"
                ],
                "priority": 5,
                "confidence": 0.8
            },
            {
                "category": "Governing Law Risk",
                "clause_type": "GOVERNING_LAW",
                "risk_level": "MEDIUM",
                "patterns": [
                    r"governed.*by.*laws.*of.*(?:delaware|new york|california|texas)",
                    r"jurisdiction.*of.*courts.*in.*(?:delaware|new york|california)",
                    r"arbitration.*in.*(?:delaware|new york|singapore|london)",
                    r"exclusive.*jurisdiction.*of"
                ],
                "description": "Potentially unfavorable jurisdiction or governing law provisions",
                "concerns": [
                    "Inconvenient legal jurisdiction",
                    "Unfamiliar legal system",
                    "Higher dispute resolution costs"
                ],
                "strategies": [
                    "Negotiate mutual jurisdiction",
                    "Choose neutral arbitration location",
                    "Select familiar governing law"
                ],
                "priority": 4,
                "confidence": 0.75
            }
        ]
    
    def _initialize_risk_keywords(self):
        """Keywords that indicate high-risk content"""
        return {
            "critical": ["unlimited", "liable", "indemnify", "penalty", "terminate", "forfeit"],
            "high": ["exclusively", "perpetual", "irrevocable", "waive", "assign", "transfer"],
            "medium": ["confidential", "proprietary", "jurisdiction", "governing", "arbitration"]
        }
    
    def get_all_patterns(self):
        """Return all contract risk patterns"""
        return self.patterns
    
    def get_risk_keywords(self):
        """Return risk keywords for NLP filtering"""
        return self.risk_keywords
    
    def get_patterns_by_category(self, category: str):
        """Get patterns for specific category"""
        return [p for p in self.patterns if p["category"] == category]
    
    def get_patterns_by_risk_level(self, risk_level: str):
        """Get patterns by risk level"""
        return [p for p in self.patterns if p["risk_level"] == risk_level]

import re
from typing import Dict, List, Any

class ContractPatterns:
    """Enhanced contract risk patterns with confidence scores and categories"""
    
    def __init__(self):
        self.patterns = self._initialize_enhanced_patterns()
    
    def _initialize_enhanced_patterns(self) -> List[Dict[str, Any]]:
        """Initialize comprehensive risk patterns with metadata"""
        return [
            {
                "category": "Liability Risk",
                "confidence": 0.85,
                "severity": "HIGH",
                "patterns": [
                    r"\bunlimited\s+liability\b",
                    r"\bliable\s+for\s+all\s+damages\b",
                    r"\bindemnify.*against\s+all\b",
                    r"\bhold\s+harmless.*all\s+claims\b",
                    r"\bresponsible\s+for\s+any\s+and\s+all\b",
                    r"\bassume\s+all\s+risks?\b",
                    r"\bwithout\s+limitation.*damages\b",
                    r"\bliability.*shall\s+not\s+be\s+limited\b",
                    r"\bpunitive\s+damages\b",
                    r"\bconsequential\s+damages.*unlimited\b"
                ],
                "description": "Clauses that impose unlimited or excessive liability exposure",
                "legal_concepts": ["unlimited liability", "indemnification", "hold harmless", "damages"]
            },
            {
                "category": "Termination Risk", 
                "confidence": 0.80,
                "severity": "MEDIUM",
                "patterns": [
                    r"\bterminate.*at\s+will\b",
                    r"\bterminate.*without\s+cause\b",
                    r"\bterminate.*immediately\b",
                    r"\bterminate.*without\s+notice\b",
                    r"\bsole\s+discretion.*terminate\b",
                    r"\bno\s+cure\s+period\b",
                    r"\bwithout\s+opportunity\s+to\s+cure\b",
                    r"\bterminate.*upon.*occurrence\b",
                    r"\bautomatically\s+terminate\b",
                    r"\bimmediately\s+effective.*termination\b"
                ],
                "description": "Clauses allowing termination without adequate protection",
                "legal_concepts": ["at-will termination", "cure period", "notice requirements", "material breach"]
            },
            {
                "category": "Payment Risk",
                "confidence": 0.75,
                "severity": "MEDIUM", 
                "patterns": [
                    r"\bpayable\s+immediately\b",
                    r"\bno\s+right\s+to\s+offset\b",
                    r"\bpayment.*without\s+deduction\b",
                    r"\binterest.*default\b",
                    r"\bcompound\s+interest\b",
                    r"\bpenalty.*late\s+payment\b",
                    r"\bliquidated\s+damages\b",
                    r"\bpayment.*demand\b",
                    r"\baccelerating\s+payment\b",
                    r"\bforfeiture.*payment\b"
                ],
                "description": "Payment terms that may create financial hardship or unfair burden",
                "legal_concepts": ["liquidated damages", "penalty clauses", "acceleration", "compound interest"]
            },
            {
                "category": "Intellectual Property Risk",
                "confidence": 0.82,
                "severity": "HIGH",
                "patterns": [
                    r"\bassign.*all\s+rights\b",
                    r"\bwork\s+for\s+hire\b",
                    r"\bintellectual\s+property.*belongs\b",
                    r"\bwaive.*moral\s+rights\b",
                    r"\bno\s+right.*intellectual\s+property\b",
                    r"\bassignment.*inventions\b",
                    r"\bexclusive\s+ownership\b",
                    r"\bderivative\s+works.*owned\b",
                    r"\bpatent.*assigned\b",
                    r"\bcopyright.*transfer\b"
                ],
                "description": "Clauses that may result in loss of intellectual property rights",
                "legal_concepts": ["work for hire", "assignment", "moral rights", "derivative works"]
            },
            {
                "category": "Confidentiality Risk",
                "confidence": 0.70,
                "severity": "MEDIUM",
                "patterns": [
                    r"\bconfidential.*perpetuity\b",
                    r"\bnon-disclosure.*unlimited\b",
                    r"\bproprietary.*forever\b",
                    r"\bconfidential.*survive\s+termination\b",
                    r"\btrade\s+secrets.*indefinite\b",
                    r"\bnon-use.*all\s+purposes\b",
                    r"\breturn.*destroy.*materials\b",
                    r"\bconfidential.*broadly\s+defined\b",
                    r"\bresidual\s+knowledge.*restricted\b",
                    r"\bpublicly\s+available.*exception\b"
                ],
                "description": "Confidentiality obligations that may be overly broad or restrictive",
                "legal_concepts": ["trade secrets", "confidential information", "residual knowledge", "survival clauses"]
            },
            {
                "category": "Performance Risk",
                "confidence": 0.78,
                "severity": "MEDIUM",
                "patterns": [
                    r"\bguarantee.*performance\b",
                    r"\bwarranty.*absolute\b", 
                    r"\bstrict\s+liability.*performance\b",
                    r"\bno\s+excuses.*non-performance\b",
                    r"\bperformance.*regardless\s+of\b",
                    r"\bforce\s+majeure.*excluded\b",
                    r"\btime\s+is\s+of\s+the\s+essence\b",
                    r"\bpenalty.*delay\b",
                    r"\bmaterial\s+breach.*any\s+delay\b",
                    r"\bliquidated\s+damages.*delay\b"
                ],
                "description": "Performance obligations that may be unrealistic or create excessive liability",
                "legal_concepts": ["performance guarantees", "strict liability", "force majeure", "time essence"]
            },
            {
                "category": "Dispute Resolution Risk",
                "confidence": 0.73,
                "severity": "MEDIUM",
                "patterns": [
                    r"\barbitration.*binding\b",
                    r"\bexclusive\s+jurisdiction\b",
                    r"\bwaive.*jury\s+trial\b",
                    r"\bno\s+class\s+action\b",
                    r"\blimited.*discovery\b",
                    r"\barbitration.*expedited\b",
                    r"\bfinal.*non-appealable\b",
                    r"\barbitrator.*sole\s+discretion\b",
                    r"\bvenue.*inconvenient\b",
                    r"\bgoverning\s+law.*foreign\b"
                ],
                "description": "Dispute resolution terms that may limit legal remedies or forum access",
                "legal_concepts": ["mandatory arbitration", "jury waiver", "class action waiver", "forum selection"]
            },
            {
                "category": "Modification Risk",
                "confidence": 0.68,
                "severity": "LOW",
                "patterns": [
                    r"\bmodify.*sole\s+discretion\b",
                    r"\bamend.*without\s+notice\b",
                    r"\bchange.*unilateral\b",
                    r"\bmodification.*binding\b",
                    r"\bterms.*subject\s+to\s+change\b",
                    r"\bamendment.*written\s+notice\b",
                    r"\bmodify.*any\s+time\b",
                    r"\bchange.*effective\s+immediately\b",
                    r"\bno\s+consent.*modification\b",
                    r"\bunilateral.*amendment\b"
                ],
                "description": "Provisions allowing unilateral contract modifications",
                "legal_concepts": ["unilateral modification", "contract amendment", "notice requirements"]
            },
            {
                "category": "Compliance Risk",
                "confidence": 0.85,
                "severity": "HIGH",
                "patterns": [
                    r"\bviolation.*law.*material\s+breach\b",
                    r"\bcomply.*all\s+laws\b",
                    r"\bregulatory.*strict\s+compliance\b",
                    r"\blegal.*requirements.*absolute\b",
                    r"\bstatutory.*violation.*termination\b",
                    r"\bpermits.*maintain\s+all\b",
                    r"\blicense.*suspension.*breach\b",
                    r"\bregulatory.*approval.*required\b",
                    r"\bcompliance.*ongoing\s+obligation\b",
                    r"\bviolation.*immediate\s+default\b"
                ],
                "description": "Compliance obligations that may create strict liability or termination triggers",
                "legal_concepts": ["regulatory compliance", "material breach", "statutory violation", "permit maintenance"]
            },
            {
                "category": "Financial Risk",
                "confidence": 0.80,
                "severity": "HIGH",
                "patterns": [
                    r"\bpersonal\s+guarantee\b",
                    r"\bfinancial.*covenant\b",
                    r"\bdebt.*equity\s+ratio\b",
                    r"\bworking\s+capital.*minimum\b",
                    r"\bbankruptcy.*event\s+of\s+default\b",
                    r"\binsolvency.*material\s+breach\b",
                    r"\bcredit\s+rating.*maintain\b",
                    r"\bfinancial\s+statements.*quarterly\b",
                    r"\baudited.*financial\s+reports\b",
                    r"\bcash\s+flow.*restrictions\b"
                ],
                "description": "Financial covenants and guarantees that may create business operation risks",
                "legal_concepts": ["personal guarantee", "financial covenants", "insolvency", "credit rating"]
            }
        ]
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all contract risk patterns"""
        return self.patterns
    
    def get_patterns_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get patterns for specific category"""
        return [p for p in self.patterns if p["category"].lower() == category.lower()]
    
    def get_high_severity_patterns(self) -> List[Dict[str, Any]]:
        """Get only high severity patterns"""
        return [p for p in self.patterns if p["severity"] == "HIGH"]
    
    def search_patterns(self, text: str, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Search for patterns in text with minimum confidence threshold"""
        found_patterns = []
        
        for pattern_group in self.patterns:
            if pattern_group["confidence"] < min_confidence:
                continue
            
            for pattern in pattern_group["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    found_patterns.append({
                        "category": pattern_group["category"],
                        "pattern": pattern,
                        "confidence": pattern_group["confidence"],
                        "severity": pattern_group["severity"],
                        "description": pattern_group["description"],
                        "legal_concepts": pattern_group["legal_concepts"]
                    })
                    break  # Only count one match per category
        
        return found_patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern database"""
        total_patterns = sum(len(p["patterns"]) for p in self.patterns)
        categories = [p["category"] for p in self.patterns]
        severity_dist = {}
        confidence_dist = {"high": 0, "medium": 0, "low": 0}
        
        for pattern in self.patterns:
            severity = pattern["severity"]
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
            
            confidence = pattern["confidence"]
            if confidence >= 0.8:
                confidence_dist["high"] += 1
            elif confidence >= 0.7:
                confidence_dist["medium"] += 1
            else:
                confidence_dist["low"] += 1
        
        return {
            "total_pattern_groups": len(self.patterns),
            "total_individual_patterns": total_patterns,
            "categories": categories,
            "severity_distribution": severity_dist,
            "confidence_distribution": confidence_dist,
            "average_patterns_per_category": total_patterns / len(self.patterns)
        }

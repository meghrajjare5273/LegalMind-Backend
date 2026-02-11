// models/types.ts

/**
 * Risk level enumeration
 */
export enum RiskLevel {
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
  CRITICAL = "CRITICAL",
}

/**
 * Clause type enumeration
 */
export enum ClauseType {
  TERMINATION = "Termination",
  LIABILITY = "Liability",
  PAYMENT = "Payment",
  PENALTY = "Penalty",
  CONFIDENTIALITY = "Confidentiality",
  INTELLECTUAL_PROPERTY = "Intellectual Property",
  INDEMNIFICATION = "Indemnification",
  GOVERNING_LAW = "Governing Law",
  ARBITRATION = "Arbitration",
  FORCE_MAJEURE = "Force Majeure",
  GENERAL = "General",
}

/**
 * User roles for authentication
 */
export type UserRole = "admin" | "user";

/**
 * Authentication data extracted from JWT
 */
export interface AuthData {
  userID: string;
  email: string;
  role: UserRole;
}

/**
 * Enhanced risk item with detailed analysis
 */
export interface RiskAnalysis {
  sentence: string;
  riskCategory: string;
  riskLevel: RiskLevel;
  riskType: string;
  description: string;
  specificConcerns: string[];
  negotiationStrategies: string[];
  priorityScore: number;
  confidenceScore: number;
  legalConcepts: string[];
  entities: Record<string, string[]>;
  mitigationStrategies: string[];
  alternativeLanguage: string;
  costImplications: string;
}

/**
 * Contract section breakdown
 */
export interface ContractSection {
  title: string;
  content: string;
  riskCount: number;
  sectionType: string;
}

/**
 * Risk summary statistics
 */
export interface RiskSummary {
  totalRisks: number;
  criticalRiskCount: number;
  highRiskCount: number;
  mediumRiskCount: number;
  lowRiskCount: number;
  overallRiskLevel: string;
  riskDistribution: Record<string, number>;
}

/**
 * Complete analysis response
 */
export interface AnalysisResponse {
  filename: string;
  extractedText: string;
  analysis: RiskAnalysis[];
  summary: RiskSummary;
  sections: ContractSection[];
  recommendations: string[];
  overallSummary: string;
  documentComplexityScore: number;
  partyPowerBalance: number;
  processingTime: number;
  detectionStats?: {
    ruleBasedCount: number;
    aiEnhancedCount: number;
    nlpFlaggedCount: number;
  };
}

/**
 * Internal enhanced risk item (before AI enhancement)
 */
export interface EnhancedRiskItem {
  sentence: string;
  riskCategory: string;
  riskLevel: RiskLevel;
  clauseType: ClauseType;
  description: string;
  concerns: string[];
  strategies: string[];
  priority: number;
  confidence: number;
  detectionMethod: string;
  specificRisks?: string[];
  entities?: Record<string, string[]>;
}

/**
 * NLP portion data
 */
export interface NLPPortionData {
  sentence: string;
  score: number;
  riskIndicators: string[];
  entities: Record<string, string[]>;
  keyPhrases: string[];
}

/**
 * Contract pattern definition
 */
export interface ContractPattern {
  category: string;
  riskLevel: RiskLevel;
  clauseType: ClauseType;
  patterns: string[];
  description: string;
  concerns: string[];
  strategies: string[];
  priority: number;
  confidence: number;
}

/**
 * AI Enhancement result from Gemini
 */
export interface AIEnhancement {
  enhancedDescription: string;
  specificConcerns: string[];
  negotiationStrategies: string[];
  alternativeLanguage: string;
  legalPrecedent: string;
  urgencyAssessment: "HIGH" | "MEDIUM" | "LOW";
  financialImpact: string;
  mitigationPriority: number;
}

/**
 * Rate limit error details
 */
export interface RateLimitError {
  error: string;
  message: string;
  retryAfter: number; // seconds
  nextAvailableAt: string; // ISO timestamp
}

/**
 * Analyze contract request
 */
export interface AnalyzeContractRequest {
  userRole: string;
  file: Buffer;
  filename: string;
}

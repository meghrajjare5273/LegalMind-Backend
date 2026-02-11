// services/contract/patterns.ts
import { RiskLevel, ClauseType, type ContractPattern } from "@/models/types";

/**
 * Comprehensive contract risk patterns
 * Based on legal analysis and common contract pitfalls
 */
export class ContractPatterns {
  /**
   * Get all contract patterns
   */
  static getAllPatterns(): ContractPattern[] {
    return [
      // CRITICAL RISKS
      {
        category: "Unlimited Liability Risk",
        riskLevel: RiskLevel.CRITICAL,
        clauseType: ClauseType.LIABILITY,
        patterns: [
          "\\bunlimited\\s+liability\\b",
          "\\bliable\\s+for\\s+all\\b",
          "\\bentire\\s+liability\\b",
          "\\bfully\\s+liable\\b",
          "\\bliability\\s+without\\s+limit\\b",
        ],
        description: "Clause exposes party to unlimited financial liability",
        concerns: [
          "Potentially catastrophic financial exposure",
          "No cap on damages",
          "Could exceed contract value significantly",
        ],
        strategies: [
          "Negotiate liability cap (e.g., 2x contract value)",
          "Exclude consequential damages",
          "Obtain liability insurance",
        ],
        priority: 10,
        confidence: 0.95,
      },
      {
        category: "Immediate Termination Risk",
        riskLevel: RiskLevel.CRITICAL,
        clauseType: ClauseType.TERMINATION,
        patterns: [
          "\\bterminate\\s+immediately\\b",
          "\\bterminate\\s+without\\s+notice\\b",
          "\\binstant\\s+termination\\b",
          "\\bterminate\\s+at\\s+will\\b",
          "\\btermination\\s+effective\\s+immediately\\b",
        ],
        description: "Contract can be terminated without notice or cause",
        concerns: [
          "No time to prepare for contract end",
          "Potential business disruption",
          "Loss of expected revenue",
        ],
        strategies: [
          "Require 30-60 days termination notice",
          "Add 'for cause' termination clause",
          "Include transition assistance period",
        ],
        priority: 9,
        confidence: 0.9,
      },

      // HIGH RISKS
      {
        category: "Severe Payment Terms",
        riskLevel: RiskLevel.HIGH,
        clauseType: ClauseType.PAYMENT,
        patterns: [
          "\\bpayment\\s+due\\s+immediately\\b",
          "\\bpayable\\s+upon\\s+demand\\b",
          "\\badvance\\s+payment\\b",
          "\\bpayment\\s+in\\s+full\\s+upfront\\b",
          "\\bnon-refundable\\s+payment\\b",
        ],
        description: "Harsh payment conditions that favor one party",
        concerns: [
          "Cash flow constraints",
          "Payment before services rendered",
          "No recourse if service not delivered",
        ],
        strategies: [
          "Negotiate milestone-based payments",
          "Request performance guarantees",
          "Add service level agreements",
        ],
        priority: 8,
        confidence: 0.85,
      },
      {
        category: "Automatic Renewal Risk",
        riskLevel: RiskLevel.HIGH,
        clauseType: ClauseType.TERMINATION,
        patterns: [
          "\\bautomatically\\s+renew\\b",
          "\\bauto-renew\\b",
          "\\bperpetual\\s+renewal\\b",
          "\\brenews\\s+unless\\s+cancelled\\b",
          "\\btacit\\s+renewal\\b",
        ],
        description: "Contract automatically renews without explicit consent",
        concerns: [
          "Easy to miss cancellation window",
          "Locked into unfavorable terms",
          "Unexpected cost continuation",
        ],
        strategies: [
          "Add calendar reminders before renewal",
          "Negotiate explicit renewal terms",
          "Require mutual written consent for renewal",
        ],
        priority: 7,
        confidence: 0.8,
      },
      {
        category: "Excessive Penalty Clauses",
        riskLevel: RiskLevel.HIGH,
        clauseType: ClauseType.PENALTY,
        patterns: [
          "\\bpenalty\\s+of\\s+\\d+%",
          "\\bliquidated\\s+damages\\b",
          "\\bforfeiture\\s+of\\b",
          "\\bfine\\s+of\\b",
          "\\bpenalty\\s+fee\\b",
        ],
        description: "Disproportionate penalties for minor breaches",
        concerns: [
          "Penalties may exceed actual damages",
          "Financial burden for minor violations",
          "May be unenforceable in some jurisdictions",
        ],
        strategies: [
          "Negotiate reasonable penalty caps",
          "Tie penalties to actual damages",
          "Add cure periods before penalties apply",
        ],
        priority: 8,
        confidence: 0.85,
      },

      // MEDIUM RISKS
      {
        category: "Broad Confidentiality",
        riskLevel: RiskLevel.MEDIUM,
        clauseType: ClauseType.CONFIDENTIALITY,
        patterns: [
          "\\ball\\s+information\\s+confidential\\b",
          "\\bperpetual\\s+confidentiality\\b",
          "\\bindefinite\\s+confidentiality\\b",
          "\\bconfidentiality\\s+survives\\s+termination\\b",
        ],
        description: "Overly broad confidentiality obligations",
        concerns: [
          "May restrict normal business operations",
          "Long-term disclosure restrictions",
          "Unclear scope of confidential information",
        ],
        strategies: [
          "Define specific confidential information",
          "Limit confidentiality period (e.g., 3-5 years)",
          "Add standard exceptions (public domain, etc.)",
        ],
        priority: 6,
        confidence: 0.75,
      },
      {
        category: "IP Ownership Concerns",
        riskLevel: RiskLevel.MEDIUM,
        clauseType: ClauseType.INTELLECTUAL_PROPERTY,
        patterns: [
          "\\ball\\s+intellectual\\s+property\\b",
          "\\bownership\\s+of\\s+all\\s+work\\b",
          "\\bwork\\s+for\\s+hire\\b",
          "\\bassign\\s+all\\s+rights\\b",
          "\\btransfer\\s+of\\s+ip\\b",
        ],
        description: "Broad intellectual property transfer or assignment",
        concerns: [
          "Loss of IP rights to created work",
          "Cannot reuse developed solutions",
          "May include pre-existing IP",
        ],
        strategies: [
          "Clarify pre-existing IP exclusions",
          "Negotiate limited license vs. assignment",
          "Retain rights to reusable components",
        ],
        priority: 7,
        confidence: 0.8,
      },
      {
        category: "Indemnification Risk",
        riskLevel: RiskLevel.MEDIUM,
        clauseType: ClauseType.INDEMNIFICATION,
        patterns: [
          "\\bindemnify\\s+and\\s+hold\\s+harmless\\b",
          "\\bindemnification\\s+for\\s+any\\s+claims\\b",
          "\\bdefend,\\s+indemnify\\b",
          "\\bindemnify\\s+against\\s+all\\b",
        ],
        description: "One-sided indemnification obligations",
        concerns: [
          "Assuming liability for third-party claims",
          "Legal defense costs",
          "Unbalanced risk allocation",
        ],
        strategies: [
          "Make indemnification mutual",
          "Limit to claims arising from own actions",
          "Add exceptions for gross negligence",
        ],
        priority: 6,
        confidence: 0.75,
      },
      {
        category: "Unfavorable Jurisdiction",
        riskLevel: RiskLevel.MEDIUM,
        clauseType: ClauseType.GOVERNING_LAW,
        patterns: [
          "\\bgoverned\\s+by\\s+laws\\s+of\\b",
          "\\bjurisdiction\\s+of\\s+courts\\s+in\\b",
          "\\bexclusive\\s+jurisdiction\\b",
          "\\bvenue\\s+shall\\s+be\\b",
        ],
        description: "Dispute resolution in inconvenient location",
        concerns: [
          "Higher costs for legal proceedings",
          "Unfamiliar legal system",
          "Travel requirements for disputes",
        ],
        strategies: [
          "Negotiate neutral jurisdiction",
          "Consider arbitration instead",
          "Choose mutually convenient location",
        ],
        priority: 5,
        confidence: 0.7,
      },

      // LOW RISKS
      {
        category: "Force Majeure Clause",
        riskLevel: RiskLevel.LOW,
        clauseType: ClauseType.FORCE_MAJEURE,
        patterns: [
          "\\bforce\\s+majeure\\b",
          "\\bact\\s+of\\s+god\\b",
          "\\bunforeseen\\s+circumstances\\b",
          "\\bbeyond\\s+reasonable\\s+control\\b",
        ],
        description:
          "Force majeure provisions that may be too restrictive or broad",
        concerns: [
          "May not cover relevant disruptions",
          "Unclear triggering events",
          "Notice requirements may be strict",
        ],
        strategies: [
          "Ensure pandemic/epidemic covered",
          "Add clear notification procedures",
          "Define specific qualifying events",
        ],
        priority: 4,
        confidence: 0.65,
      },
      {
        category: "Assignment Restrictions",
        riskLevel: RiskLevel.LOW,
        clauseType: ClauseType.GENERAL,
        patterns: [
          "\\bcannot\\s+assign\\b",
          "\\bno\\s+assignment\\b",
          "\\bassignment\\s+prohibited\\b",
          "\\bnot\\s+assignable\\b",
        ],
        description: "Restrictions on contract assignment",
        concerns: [
          "Limits business flexibility",
          "May affect corporate restructuring",
          "Could impact asset sales",
        ],
        strategies: [
          "Allow assignment with consent",
          "Permit assignment to affiliates",
          "Add exceptions for business transfers",
        ],
        priority: 3,
        confidence: 0.6,
      },
    ];
  }

  /**
   * Get patterns by category
   */
  static getPatternsByCategory(category: string): ContractPattern[] {
    return this.getAllPatterns().filter((p) => p.category === category);
  }

  /**
   * Get patterns by risk level
   */
  static getPatternsByRiskLevel(riskLevel: RiskLevel): ContractPattern[] {
    return this.getAllPatterns().filter((p) => p.riskLevel === riskLevel);
  }

  /**
   * Get high-priority patterns (priority >= 7)
   */
  static getHighPriorityPatterns(): ContractPattern[] {
    return this.getAllPatterns().filter((p) => p.priority >= 7);
  }
}

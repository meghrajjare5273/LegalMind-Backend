// services/contract/analyzer.ts
import {
  AnalysisResponse,
  RiskAnalysis,
  RiskSummary,
  ContractSection,
  EnhancedRiskItem,
  AIEnhancement,
  RiskLevel,
  ClauseType,
} from "@/models/types";
import { ContractPatterns } from "@/services/contracts.patterns";
import { NLPPipeline } from "@/services/contracts.nlp-pipeline";
import { geminiPool } from "@/services/ai.geminipool";
import { truncate, deduplicateByKey } from "@/utils/helpers";

interface AnalyzeOptions {
  userRole: string;
  filename: string;
}

/**
 * Main contract analyzer:
 * - PDF text is already extracted when this is called
 * - Combines rule-based + NLP + (optional) Gemini enhancement
 */
export class ContractAnalyzer {
  private readonly nlp: NLPPipeline;

  constructor() {
    this.nlp = new NLPPipeline();
  }

  /**
   * Public entrypoint: analyze raw contract text
   */
  async analyzeContract(
    text: string,
    options: AnalyzeOptions,
  ): Promise<AnalysisResponse> {
    const start = Date.now();

    // Basic sanity
    const cleaned = text.trim();
    if (!cleaned) {
      throw new Error("No readable text found in contract");
    }

    // Step 1: NLP risky portions
    const riskyPortions = this.nlp.extractRiskyPortions(cleaned, 20);

    // Step 2: Rule-based pattern scan
    const ruleBasedRisks = this.runRuleBasedAnalysis(cleaned, riskyPortions);

    // Step 3: Optional AI enhancement via Gemini
    const enhancedRisks = await this.enhanceRisksWithAI(
      ruleBasedRisks,
      cleaned,
      options.userRole,
    );

    // Step 4: Aggregate and summarize
    const {
      summary,
      sections,
      recommendations,
      overallSummary,
      complexityScore,
      powerBalance,
      detectionStats,
    } = this.buildAggregateResults(enhancedRisks, cleaned, riskyPortions);

    const processingTime = (Date.now() - start) / 1000;

    const response: AnalysisResponse = {
      filename: options.filename,
      extractedText: cleaned.length > 2000 ? truncate(cleaned, 2000) : cleaned,
      analysis: enhancedRisks.map((r) => this.toRiskAnalysis(r)),
      summary,
      sections,
      recommendations,
      overallSummary,
      documentComplexityScore: complexityScore,
      partyPowerBalance: powerBalance,
      processingTime,
      detectionStats,
    };

    return response;
  }

  /**
   * Rule-based analysis over full text + risky portions
   */
  private runRuleBasedAnalysis(
    fullText: string,
    portions: ReturnType<NLPPipeline["extractRiskyPortions"]>,
  ): EnhancedRiskItem[] {
    const patterns = ContractPatterns.getAllPatterns();
    const allSentences = this.nlp.smartSentenceSplit(fullText);
    const risks: EnhancedRiskItem[] = [];

    // 1) Pattern-based scan over all sentences
    for (const sentence of allSentences) {
      const lower = sentence.toLowerCase();

      for (const pat of patterns) {
        const matched = pat.patterns.some((re) =>
          new RegExp(re, "i").test(lower),
        );
        if (!matched) continue;

        risks.push({
          sentence,
          riskCategory: pat.category,
          riskLevel: pat.riskLevel,
          clauseType: pat.clauseType,
          description: pat.description,
          concerns: pat.concerns,
          strategies: pat.strategies,
          priority: pat.priority,
          confidence: pat.confidence,
          detectionMethod: "rule-based",
        });

        // One risk per pattern per sentence is enough
        break;
      }
    }

    // 2) If a portion is flagged by NLP but no pattern matched, add generic risk
    for (const portion of portions) {
      const already = risks.some(
        (r) => r.sentence.trim() === portion.sentence.trim(),
      );
      if (!already && portion.score >= 0.4) {
        risks.push({
          sentence: portion.sentence,
          riskCategory: "Potential Risk",
          riskLevel: RiskLevel.MEDIUM,
          clauseType: ClauseType.GENERAL,
          description:
            "Clause flagged by NLP analysis as potentially risky or important.",
          concerns: [
            "May contain unfavorable terms.",
            "Requires detailed legal review.",
          ],
          strategies: [
            "Review with legal counsel.",
            "Compare with standard market practices.",
          ],
          priority: 5,
          confidence: portion.score,
          detectionMethod: "nlp-flagged",
          specificRisks: portion.riskIndicators,
          entities: portion.entities,
        });
      }
    }

    // Deduplicate by (sentence + category)
    const unique = deduplicateByKey(
      risks,
      (r) => `${r.sentence.slice(0, 100)}|${r.riskCategory}`,
    );

    return unique;
  }

  /**
   * Enhance a subset of risks with Gemini (if available)
   */
  private async enhanceRisksWithAI(
    risks: EnhancedRiskItem[],
    fullText: string,
    userRole: string,
  ): Promise<EnhancedRiskItem[]> {
    if (!geminiPool.isAvailable()) {
      // No AI available, just return rule-based
      return risks;
    }

    // Only enhance top N by priority
    const toEnhance = [...risks]
      .sort((a, b) => b.priority - a.priority)
      .slice(0, 8);

    const enhanced: EnhancedRiskItem[] = [...risks];

    for (const risk of toEnhance) {
      try {
        const prompt = this.buildPrompt(risk, fullText, userRole);
        const text = await geminiPool.generateText(prompt, {
          temperature: 0.2,
          maxTokens: 600,
          model: "gemini-1.5-flash",
        });

        const parsed = this.parseAIResponse(text);
        const merged = this.mergeAIEnhancement(risk, parsed);

        // Replace in list
        const idx = enhanced.findIndex(
          (r) =>
            r.sentence === risk.sentence &&
            r.riskCategory === risk.riskCategory,
        );
        if (idx >= 0) enhanced[idx] = merged;
      } catch {
        // Swallow AI errors; keep original risk
      }
    }

    return enhanced;
  }

  /**
   * Build Gemini prompt incorporating user role and context
   */
  private buildPrompt(
    risk: EnhancedRiskItem,
    fullText: string,
    userRole: string,
  ): string {
    const contextSnippet = truncate(fullText, 3000);

    return `
You are an expert contract lawyer analyzing a contract for a ${userRole}.

A clause has been identified as risky:

Clause:
"""
${risk.sentence}
"""

Risk Category: ${risk.riskCategory}
Risk Level: ${risk.riskLevel}
Clause Type: ${risk.clauseType}
Existing Description: ${risk.description}

Full Contract Context (truncated):
"""
${contextSnippet}
"""

TASK:
Respond ONLY with a JSON object with the following shape:

{
  "enhancedDescription": "string - improved explanation of the risk tailored to the ${userRole}",
  "specificConcerns": ["string", "..."],
  "negotiationStrategies": ["string", "..."],
  "alternativeLanguage": "string - suggested replacement clause from the ${userRole}'s perspective",
  "legalPrecedent": "string - brief mention of typical market practice or principle",
  "urgencyAssessment": "HIGH | MEDIUM | LOW",
  "financialImpact": "string - brief explanation of monetary impact if risk materializes",
  "mitigationPriority": 1-10
}

Important:
- Return ONLY valid JSON.
- Do not wrap JSON in backticks.
- Keep text concise but specific.
`;
  }

  /**
   * Parse AI JSON response robustly
   */
  private parseAIResponse(text: string): AIEnhancement {
    try {
      const trimmed = text.trim();

      // In case model added code fences or explanation
      const jsonString = trimmed
        .replace(/```json/i, "")
        .replace(/```/g, "")
        .trim();

      const parsed = JSON.parse(jsonString);

      const result: AIEnhancement = {
        enhancedDescription: parsed.enhancedDescription ?? "",
        specificConcerns: Array.isArray(parsed.specificConcerns)
          ? parsed.specificConcerns
          : [],
        negotiationStrategies: Array.isArray(parsed.negotiationStrategies)
          ? parsed.negotiationStrategies
          : [],
        alternativeLanguage: parsed.alternativeLanguage ?? "",
        legalPrecedent: parsed.legalPrecedent ?? "",
        urgencyAssessment:
          parsed.urgencyAssessment === "HIGH" ||
          parsed.urgencyAssessment === "LOW"
            ? parsed.urgencyAssessment
            : "MEDIUM",
        financialImpact: parsed.financialImpact ?? "",
        mitigationPriority:
          typeof parsed.mitigationPriority === "number"
            ? parsed.mitigationPriority
            : 5,
      };

      return result;
    } catch {
      // Fallback if parsing fails
      return {
        enhancedDescription: "",
        specificConcerns: [],
        negotiationStrategies: [],
        alternativeLanguage: "",
        legalPrecedent: "",
        urgencyAssessment: "MEDIUM",
        financialImpact: "",
        mitigationPriority: 5,
      };
    }
  }

  /**
   * Merge AI enhancement into base risk item
   */
  private mergeAIEnhancement(
    base: EnhancedRiskItem,
    ai: AIEnhancement,
  ): EnhancedRiskItem {
    const description =
      ai.enhancedDescription || base.description || base.riskCategory;

    const concerns =
      ai.specificConcerns.length > 0 ? ai.specificConcerns : base.concerns;

    const strategies =
      ai.negotiationStrategies.length > 0
        ? ai.negotiationStrategies
        : base.strategies;

    const priority =
      typeof ai.mitigationPriority === "number"
        ? ai.mitigationPriority
        : base.priority;

    return {
      ...base,
      description,
      concerns,
      strategies,
      priority,
      detectionMethod: base.detectionMethod.includes("ai")
        ? base.detectionMethod
        : `${base.detectionMethod}+ai`,
    };
  }

  /**
   * Build summary, sections, recommendations, etc.
   */
  private buildAggregateResults(
    risks: EnhancedRiskItem[],
    fullText: string,
    portions: ReturnType<NLPPipeline["extractRiskyPortions"]>,
  ): {
    summary: RiskSummary;
    sections: ContractSection[];
    recommendations: string[];
    overallSummary: string;
    complexityScore: number;
    powerBalance: number;
    detectionStats: {
      ruleBasedCount: number;
      aiEnhancedCount: number;
      nlpFlaggedCount: number;
    };
  } {
    const total = risks.length;
    const critical = risks.filter(
      (r) => r.riskLevel === RiskLevel.CRITICAL,
    ).length;
    const high = risks.filter((r) => r.riskLevel === RiskLevel.HIGH).length;
    const medium = risks.filter((r) => r.riskLevel === RiskLevel.MEDIUM).length;
    const low = risks.filter((r) => r.riskLevel === RiskLevel.LOW).length;

    const summary: RiskSummary = {
      totalRisks: total,
      criticalRiskCount: critical,
      highRiskCount: high,
      mediumRiskCount: medium,
      lowRiskCount: low,
      overallRiskLevel: this.computeOverallRiskLevel(
        critical,
        high,
        medium,
        low,
      ),
      riskDistribution: {
        CRITICAL: critical,
        HIGH: high,
        MEDIUM: medium,
        LOW: low,
      },
    };

    const sections = this.buildSections(risks, fullText);

    const recommendations = this.buildRecommendations(summary, risks);

    const overallSummary = this.buildOverallSummary(summary);

    const stats = this.nlp.getDocumentStats(fullText);

    const powerBalance = this.estimatePowerBalance(risks);

    const detectionStats = {
      ruleBasedCount: risks.filter((r) => r.detectionMethod.includes("rule"))
        .length,
      aiEnhancedCount: risks.filter((r) => r.detectionMethod.includes("ai"))
        .length,
      nlpFlaggedCount: portions.length,
    };

    return {
      summary,
      sections,
      recommendations,
      overallSummary,
      complexityScore: stats.complexityScore,
      powerBalance,
      detectionStats,
    };
  }

  private computeOverallRiskLevel(
    critical: number,
    high: number,
    medium: number,
    low: number,
  ): string {
    if (critical > 0) return "CRITICAL";
    if (high > 2 || (high > 0 && medium > 3)) return "HIGH";
    if (medium > 0) return "MEDIUM";
    return "LOW";
  }

  private buildSections(
    risks: EnhancedRiskItem[],
    fullText: string,
  ): ContractSection[] {
    const sections: ContractSection[] = [];

    const headings = this.nlp
      .smartSentenceSplit(fullText)
      .filter((s) => s.length <= 80 && /^[A-Z0-9 .-]{10,}$/.test(s));

    if (headings.length === 0) {
      sections.push({
        title: "Entire Contract",
        content: truncate(fullText, 2000),
        riskCount: risks.length,
        sectionType: "general",
      });
      return sections;
    }

    for (const heading of headings) {
      const regex = new RegExp(
        `${this.escapeRegex(heading)}[\\s\\S]*?(?=${headings
          .filter((h) => h !== heading)
          .map((h) => this.escapeRegex(h))
          .join("|")}|$)`,
        "i",
      );

      const match = fullText.match(regex);
      if (!match) continue;

      const content = match[0];
      const sectionRisks = risks.filter((r) =>
        content.includes(r.sentence.slice(0, 20)),
      );

      sections.push({
        title: heading,
        content: truncate(content, 1500),
        riskCount: sectionRisks.length,
        sectionType: "heading",
      });
    }

    return sections;
  }

  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  private buildRecommendations(
    summary: RiskSummary,
    risks: EnhancedRiskItem[],
  ): string[] {
    const recs: string[] = [];

    if (summary.criticalRiskCount > 0) {
      recs.push(
        "Address all CRITICAL risks before signing. These may materially impact your legal and financial position.",
      );
    }
    if (summary.highRiskCount > 0) {
      recs.push(
        "Focus negotiation efforts on HIGH risk clauses, especially around liability, termination, and penalties.",
      );
    }

    if (risks.some((r) => r.clauseType === ClauseType.LIABILITY)) {
      recs.push(
        "Consider adding liability caps aligned with industry standards and excluding indirect or consequential damages.",
      );
    }
    if (risks.some((r) => r.clauseType === ClauseType.TERMINATION)) {
      recs.push(
        "Ensure termination clauses include reasonable notice periods and clear 'for cause' conditions.",
      );
    }

    if (recs.length === 0) {
      recs.push(
        "No major issues detected, but have a qualified lawyer review the contract before signing.",
      );
    }

    return recs;
  }

  private buildOverallSummary(summary: RiskSummary): string {
    return `Overall, the contract is assessed as ${summary.overallRiskLevel} risk with ${summary.totalRisks} notable risk item(s), including ${summary.criticalRiskCount} critical and ${summary.highRiskCount} high-risk clauses.`;
  }

  private estimatePowerBalance(risks: EnhancedRiskItem[]): number {
    // Simple heuristic: more high/critical risks => more imbalanced
    const score =
      risks.filter((r) => r.riskLevel === RiskLevel.CRITICAL).length * 0.15 +
      risks.filter((r) => r.riskLevel === RiskLevel.HIGH).length * 0.1;

    // Clamp 0..1, where 0.5 is balanced, <0.5 user disadvantaged, >0.5 user favored
    const base = 0.5 - Math.min(score, 0.4);
    return Math.max(0, Math.min(1, base));
  }

  private toRiskAnalysis(r: EnhancedRiskItem): RiskAnalysis {
    return {
      sentence: r.sentence,
      riskCategory: r.riskCategory,
      riskLevel: r.riskLevel,
      riskType: r.clauseType,
      description: r.description,
      specificConcerns: r.concerns,
      negotiationStrategies: r.strategies,
      priorityScore: r.priority,
      confidenceScore: r.confidence,
      legalConcepts: [],
      entities: r.entities ?? {},
      mitigationStrategies: r.strategies,
      alternativeLanguage: "",
      costImplications: "",
    };
  }
}

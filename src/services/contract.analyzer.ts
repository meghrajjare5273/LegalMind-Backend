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
import { cache } from "@/utils/cache";

interface AnalyzeOptions {
  userRole: string;
  filename: string;
  userId?: string; // Optional user ID for tracking/logging
}

/**
 * Result of AI enhancement for a single risk
 */
interface AIEnhancementResult {
  risk: EnhancedRiskItem;
  enhancement: AIEnhancement | null;
  error: Error | null;
}

/**
 * Main contract analyzer:
 * - PDF text is already extracted when this is called
 * - Combines rule-based + NLP + (optional) Gemini enhancement
 * - Uses caching for repeated analyses
 */
export class ContractAnalyzer {
  private readonly nlp: NLPPipeline;
  private static nlpInstance: NLPPipeline | null = null;

  constructor() {
    // Use singleton NLP instance to avoid reloading model
    if (!ContractAnalyzer.nlpInstance) {
      ContractAnalyzer.nlpInstance = new NLPPipeline();
    }
    this.nlp = ContractAnalyzer.nlpInstance;
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

    // Check cache for identical document
    const documentHash = cache.generateKey(cleaned);
    const cacheKey = `analysis:${documentHash}:${options.userRole}`;
    const cached = cache.get<AnalysisResponse>(cacheKey);

    if (cached) {
      console.log(
        `[Analyzer] Cache hit for document hash: ${documentHash.slice(0, 8)}...`,
      );
      return {
        ...cached,
        filename: options.filename, // Always use current filename
        processingTime: (Date.now() - start) / 1000,
      };
    }

    // Step 1: NLP risky portions
    const riskyPortions = this.nlp.extractRiskyPortions(cleaned, 20);

    // Step 2: Rule-based pattern scan (uses pre-compiled patterns)
    const ruleBasedRisks = this.runRuleBasedAnalysis(cleaned, riskyPortions);

    // Step 3: Optional AI enhancement via Gemini (parallel processing)
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

    // Cache the result (without filename-specific data)
    cache.set(cacheKey, response, 30 * 60 * 1000); // 30 minutes TTL

    return response;
  }

  /**
   * Rule-based analysis over full text + risky portions
   * Uses pre-compiled regex patterns for better performance
   */
  private runRuleBasedAnalysis(
    fullText: string,
    portions: ReturnType<NLPPipeline["extractRiskyPortions"]>,
  ): EnhancedRiskItem[] {
    // Get pre-compiled patterns
    const patterns = ContractPatterns.getAllPatterns();
    const allSentences = this.nlp.smartSentenceSplit(fullText);
    const risks: EnhancedRiskItem[] = [];

    // 1) Pattern-based scan over all sentences using pre-compiled regex
    for (const sentence of allSentences) {
      const lower = sentence.toLowerCase();

      for (const pat of patterns) {
        // Use pre-compiled patterns instead of creating new RegExp each time
        const matched = pat.compiledPatterns.some((regex) => regex.test(lower));
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
   * Uses parallel processing with Promise.allSettled for better performance
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

    // Process AI enhancements in parallel using Promise.allSettled
    const enhancementPromises = toEnhance.map(async (risk) => {
      try {
        const prompt = this.buildPrompt(risk, fullText, userRole);
        const text = await geminiPool.generateText(prompt, {
          temperature: 0.2,
          maxTokens: 600,
          model: "gemini-1.5-flash",
        });

        const parsed = this.parseAIResponse(text);
        return {
          risk,
          enhancement: parsed,
          error: null,
        } as AIEnhancementResult;
      } catch (error) {
        // Log error but don't throw - return null enhancement
        console.warn(
          `[Analyzer] AI enhancement failed for risk "${risk.riskCategory}":`,
          (error as Error).message,
        );
        return {
          risk,
          enhancement: null,
          error: error as Error,
        } as AIEnhancementResult;
      }
    });

    // Wait for all enhancements to complete (parallel)
    const enhancementResults = await Promise.allSettled(enhancementPromises);

    // Create a map of enhancements for quick lookup
    const enhancementMap = new Map<string, AIEnhancement>();

    for (const result of enhancementResults) {
      if (result.status === "fulfilled" && result.value.enhancement) {
        const key = `${result.value.risk.sentence}|${result.value.risk.riskCategory}`;
        enhancementMap.set(key, result.value.enhancement);
      }
    }

    // Apply enhancements to risks
    const enhanced: EnhancedRiskItem[] = risks.map((risk) => {
      const key = `${risk.sentence}|${risk.riskCategory}`;
      const enhancement = enhancementMap.get(key);

      if (enhancement) {
        return this.mergeAIEnhancement(risk, enhancement);
      }
      return risk;
    });

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
    _low: number,
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
    const recommendations: string[] = [];

    // Critical risks
    if (summary.criticalRiskCount > 0) {
      recommendations.push(
        `Address ${summary.criticalRiskCount} critical risk(s) before signing this contract.`,
      );
    }

    // High risks
    if (summary.highRiskCount > 0) {
      recommendations.push(
        `Review and negotiate the ${summary.highRiskCount} high-risk clause(s) identified.`,
      );
    }

    // Top priority items
    const topPriority = risks.filter((r) => r.priority >= 8).slice(0, 3);

    for (const risk of topPriority) {
      if (risk.strategies.length > 0) {
        recommendations.push(`${risk.riskCategory}: ${risk.strategies[0]}`);
      }
    }

    // General recommendation
    if (summary.totalRisks > 5) {
      recommendations.push(
        "Consider having a legal professional review this contract due to the number of identified risks.",
      );
    }

    return recommendations.slice(0, 6);
  }

  private buildOverallSummary(summary: RiskSummary): string {
    const parts: string[] = [];

    parts.push(
      `Found ${summary.totalRisks} potential risk(s) in this contract.`,
    );

    if (summary.criticalRiskCount > 0) {
      parts.push(
        `${summary.criticalRiskCount} critical risk(s) require immediate attention.`,
      );
    }

    if (summary.highRiskCount > 0) {
      parts.push(`${summary.highRiskCount} high-risk clause(s) identified.`);
    }

    if (summary.mediumRiskCount > 0) {
      parts.push(`${summary.mediumRiskCount} moderate concern(s) noted.`);
    }

    parts.push(`Overall risk level: ${summary.overallRiskLevel}.`);

    return parts.join(" ");
  }

  private estimatePowerBalance(risks: EnhancedRiskItem[]): number {
    // Simple heuristic: count one-sided vs balanced clauses
    // Returns a score from -10 (heavily favors other party) to +10 (heavily favors user)
    let score = 0;

    for (const risk of risks) {
      // Critical and high risks typically indicate one-sided terms
      if (risk.riskLevel === RiskLevel.CRITICAL) {
        score -= 3;
      } else if (risk.riskLevel === RiskLevel.HIGH) {
        score -= 2;
      } else if (risk.riskLevel === RiskLevel.MEDIUM) {
        score -= 1;
      }
    }

    // Normalize to -10 to +10 range
    return Math.max(-10, Math.min(10, score));
  }

  private toRiskAnalysis(risk: EnhancedRiskItem): RiskAnalysis {
    return {
      sentence: risk.sentence,
      riskCategory: risk.riskCategory,
      riskLevel: risk.riskLevel,
      riskType: risk.clauseType,
      description: risk.description,
      specificConcerns: risk.concerns,
      negotiationStrategies: risk.strategies,
      priorityScore: risk.priority,
      confidenceScore: risk.confidence,
      legalConcepts: [],
      entities: risk.entities || {},
      mitigationStrategies: risk.strategies,
      alternativeLanguage: "",
      costImplications: "",
    };
  }
}

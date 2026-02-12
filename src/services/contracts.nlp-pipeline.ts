// services/contract/nlp-pipeline.ts
import winkNLP from "wink-nlp";
import model from "wink-eng-lite-web-model";
import type { NLPPortionData } from "@/models/types";

/**
 * NLP Pipeline using wink-nlp for contract text analysis
 * Provides sentence splitting, entity extraction, and risk scoring
 */
export class NLPPipeline {
  private nlp: any;
  private readonly riskKeywords = {
    critical: [
      "unlimited",
      "liable",
      "indemnify",
      "penalty",
      "terminate",
      "forfeit",
      "irrevocable",
    ],
    high: [
      "exclusively",
      "perpetual",
      "waive",
      "assign",
      "transfer",
      "immediate",
      "forthwith",
    ],
    medium: [
      "confidential",
      "proprietary",
      "jurisdiction",
      "governing",
      "arbitration",
      "dispute",
    ],
  };

  constructor() {
    this.nlp = winkNLP(model);
  }

  /**
   * Smart sentence splitting with legal text awareness
   */
  smartSentenceSplit(text: string): string[] {
    // Clean text first
    const cleanedText = text.replace(/\s+/g, " ").trim();

    // Use wink-nlp for sentence tokenization
    const doc = this.nlp.readDoc(cleanedText);
    const sentences = doc.sentences().out();

    // Filter sentences by length (remove very short or very long)
    return sentences.filter((sentence: string) => {
      const wordCount = sentence.split(/\s+/).length;
      return wordCount >= 5 && wordCount <= 120;
    });
  }

  /**
   * Extract risky portions from text with scoring
   */
  extractRiskyPortions(
    text: string,
    maxPortions: number = 20,
  ): NLPPortionData[] {
    const sentences = this.smartSentenceSplit(text);
    const scoredSentences: NLPPortionData[] = [];

    for (const sentence of sentences) {
      const score = this.calculateRiskScore(sentence);

      if (score >= 0.3) {
        // Only include sentences with moderate+ risk
        scoredSentences.push({
          sentence,
          score,
          riskIndicators: this.extractRiskIndicators(sentence),
          entities: this.extractEntities(sentence),
          keyPhrases: this.extractKeyPhrases(sentence),
        });
      }
    }

    // Sort by score descending and take top N
    return scoredSentences
      .sort((a, b) => b.score - a.score)
      .slice(0, maxPortions);
  }

  /**
   * Calculate risk score for a sentence
   */
  private calculateRiskScore(sentence: string): number {
    const lower = sentence.toLowerCase();
    let score = 0;

    // Check for risk keywords
    for (const [level, keywords] of Object.entries(this.riskKeywords)) {
      for (const keyword of keywords) {
        if (lower.includes(keyword)) {
          if (level === "critical") score += 0.6;
          else if (level === "high") score += 0.4;
          else if (level === "medium") score += 0.2;
        }
      }
    }

    // Check for absolute language
    if (/\b(all|any|every|entire|complete|total|absolute)\b/.test(lower)) {
      score += 0.3;
    }

    // Check for time constraints
    if (/\b(immediately|forthwith|without delay|instant)\b/.test(lower)) {
      score += 0.3;
    }

    // Check for monetary amounts (higher risk)
    if (/\$\d{1,3}(,\d{3})*(\.\d{2})?/.test(sentence)) {
      score += 0.2;
    }

    // Check for percentages
    if (/\d+(\.\d+)?%/.test(sentence)) {
      score += 0.2;
    }

    // Check for negations (often risky)
    if (/\b(not|no|never|without|except)\b/.test(lower)) {
      score += 0.2;
    }

    return Math.min(score, 1.0); // Cap at 1.0
  }

  /**
   * Extract risk indicators from sentence
   */
  private extractRiskIndicators(sentence: string): string[] {
    const indicators: string[] = [];
    const lower = sentence.toLowerCase();

    for (const [level, keywords] of Object.entries(this.riskKeywords)) {
      for (const keyword of keywords) {
        if (lower.includes(keyword)) {
          indicators.push(`${level}:${keyword}`);
        }
      }
    }

    return indicators;
  }

  /**
   * Extract entities from sentence using wink-nlp
   */
  private extractEntities(sentence: string): Record<string, string[]> {
    const doc = this.nlp.readDoc(sentence);
    const entities: Record<string, string[]> = {
      amounts: [],
      percentages: [],
      timeframes: [],
      parties: [],
      dates: [],
    };

    // Extract monetary amounts
    const amountMatches = sentence.match(/\$\d{1,3}(,\d{3})*(\.\d{2})?/g);
    if (amountMatches) {
      entities.amounts = amountMatches;
    }

    // Extract percentages
    const percentMatches = sentence.match(/\d+(\.\d+)?%/g);
    if (percentMatches) {
      entities.percentages = percentMatches;
    }

    // Extract timeframes
    const timeMatches = sentence.match(
      /\d+\s+(days?|months?|years?|hours?|weeks?)/gi,
    );
    if (timeMatches) {
      entities.timeframes = timeMatches;
    }

    // Extract potential party names (proper nouns)
    const properNouns = doc
      .tokens()
      .filter((t: any) => t.out(this.nlp.its.pos) === "PROPN")
      .out();

    if (properNouns.length > 0) {
      entities.parties = properNouns;
    }

    // Extract dates
    const dateMatches = sentence.match(/\d{1,2}[-/]\d{1,2}[-/]\d{2,4}/g);
    if (dateMatches) {
      entities.dates = dateMatches;
    }

    return entities;
  }

  /**
   * Extract key phrases from sentence
   */
  private extractKeyPhrases(sentence: string): string[] {
    const doc = this.nlp.readDoc(sentence);

    // Extract noun phrases and important terms
    const tokens = doc.tokens().out();
    const phrases: string[] = [];

    // Simple bigram extraction for key phrases
    for (let i = 0; i < tokens.length - 1; i++) {
      const bigram = `${tokens[i]} ${tokens[i + 1]}`;
      const lower = bigram.toLowerCase();

      // Only include meaningful bigrams
      if (
        lower.includes("payment") ||
        lower.includes("liability") ||
        lower.includes("termination") ||
        lower.includes("penalty") ||
        lower.includes("indemnif") ||
        lower.includes("intellectual") ||
        lower.includes("property")
      ) {
        phrases.push(bigram);
      }
    }

    return phrases;
  }

  /**
   * Get document statistics
   */
  getDocumentStats(text: string): {
    sentenceCount: number;
    wordCount: number;
    averageWordsPerSentence: number;
    complexityScore: number;
  } {
    const doc = this.nlp.readDoc(text);
    const sentences = doc.sentences().out();
    const words = doc.tokens().out();

    const sentenceCount = sentences.length;
    const wordCount = words.length;
    const averageWordsPerSentence = wordCount / sentenceCount;

    // Simple complexity score based on sentence length and vocabulary
    const complexityScore = Math.min(
      averageWordsPerSentence / 30 + wordCount / 5000,
      1.0,
    );

    return {
      sentenceCount,
      wordCount,
      averageWordsPerSentence: Math.round(averageWordsPerSentence * 10) / 10,
      complexityScore: Math.round(complexityScore * 100) / 100,
    };
  }
}

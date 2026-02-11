// services/ai/gemini-pool.ts
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { generateText } from "ai";
import { secret } from "encore.dev/config";
import { retryWithBackoff } from "@/utils/helpers";

// Gemini API keys from environment (comma-separated)
const geminiKeysSecret = secret("GeminiAPIKeys");

/**
 * Gemini client pool with automatic rotation
 * Uses Vercel AI SDK for streaming and advanced features
 */
export class GeminiClientPool {
  private clients: any[] = [];
  private currentIndex = 0;
  private apiKeys: string[] = [];

  constructor() {
    this.initializePool();
  }

  /**
   * Initialize client pool from API keys
   */
  private initializePool(): void {
    const keysString = geminiKeysSecret();

    if (!keysString) {
      console.warn("GeminiAPIKeys not configured. AI enhancement disabled.");
      return;
    }

    // Split comma-separated keys and trim whitespace
    this.apiKeys = keysString
      .split(",")
      .map((key) => key.trim())
      .filter((key) => key.length > 0);

    if (this.apiKeys.length === 0) {
      console.warn("No valid Gemini API keys found. AI enhancement disabled.");
      return;
    }

    // Create clients for each API key
    this.clients = this.apiKeys.map((apiKey) =>
      createGoogleGenerativeAI({ apiKey }),
    );

    console.log(`Initialized ${this.clients.length} Gemini clients in pool`);
  }

  /**
   * Get next client in rotation
   */
  private getNextClient(): any | null {
    if (this.clients.length === 0) {
      return null;
    }

    const client = this.clients[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.clients.length;
    return client;
  }

  /**
   * Check if pool is available
   */
  isAvailable(): boolean {
    return this.clients.length > 0;
  }

  /**
   * Generate text using Gemini with rotation and retry
   */
  async generateText(
    prompt: string,
    options: {
      temperature?: number;
      maxTokens?: number;
      model?: string;
    } = {},
  ): Promise<string> {
    if (!this.isAvailable()) {
      throw new Error("Gemini client pool not available");
    }

    const {
      temperature = 0.2,
      maxTokens = 600,
      model = "gemini-1.5-flash",
    } = options;

    // Retry with exponential backoff
    return retryWithBackoff(async () => {
      const client = this.getNextClient();

      if (!client) {
        throw new Error("No Gemini client available");
      }

      const result = await generateText({
        model: client(model),
        prompt,
        temperature,
        maxOutputTokens: maxTokens,
      });

      return result.text;
    }, 3);
  }

  /**
   * Get pool statistics
   */
  getStats(): {
    totalClients: number;
    currentIndex: number;
    isAvailable: boolean;
  } {
    return {
      totalClients: this.clients.length,
      currentIndex: this.currentIndex,
      isAvailable: this.isAvailable(),
    };
  }
}

// Export singleton instance
export const geminiPool = new GeminiClientPool();

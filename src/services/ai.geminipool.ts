// services/ai.geminipool.ts
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { generateText } from "ai";
// import { secret } from "encore.dev/config";
import { retryWithBackoff } from "@/utils/helpers";
import { GeminiAPIKey } from "@/contract-analysis/secrets";

// const geminiKeysSecret = secret("GeminiAPIKeys");

export class GeminiClientPool {
  private clients: any[] = [];
  private currentIndex = 0;
  private initialized = false;

  private initializePool(): void {
    const keysString = GeminiAPIKey;

    if (!keysString) {
      console.warn("GeminiAPIKeys not configured. AI enhancement disabled.");
      return;
    }

    const apiKeys = keysString
      .split(",")
      .map((key) => key.trim())
      .filter((key) => key.length > 0);

    if (apiKeys.length === 0) {
      console.warn("No valid Gemini API keys found. AI enhancement disabled.");
      return;
    }

    this.clients = apiKeys.map((apiKey) =>
      createGoogleGenerativeAI({ apiKey }),
    );
    this.initialized = true;

    console.log(`Initialized ${this.clients.length} Gemini clients in pool`);
  }

  private ensureInitialized(): void {
    if (!this.initialized) {
      this.initializePool();
    }
  }

  private getNextClient(): any | null {
    if (this.clients.length === 0) return null;
    const client = this.clients[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.clients.length;
    return client;
  }

  isAvailable(): boolean {
    this.ensureInitialized();
    return this.clients.length > 0;
  }

  async generateText(
    prompt: string,
    options: {
      temperature?: number;
      maxTokens?: number;
      model?: string;
    } = {},
  ): Promise<string> {
    this.ensureInitialized();

    if (!this.isAvailable()) {
      throw new Error("Gemini client pool not available");
    }

    const {
      temperature = 0.2,
      maxTokens = 600,
      model = "gemini-1.5-flash",
    } = options;

    return retryWithBackoff(async () => {
      const client = this.getNextClient();
      if (!client) throw new Error("No Gemini client available");

      const result = await generateText({
        model: client(model),
        prompt,
        temperature,
        maxOutputTokens: maxTokens,
      });

      return result.text;
    }, 3);
  }

  getStats() {
    this.ensureInitialized();
    return {
      totalClients: this.clients.length,
      currentIndex: this.currentIndex,
      isAvailable: this.isAvailable(),
    };
  }
}

export const geminiPool = new GeminiClientPool();

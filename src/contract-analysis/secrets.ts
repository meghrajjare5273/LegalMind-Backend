// src/contract-analysis/secrets.ts
import { secret } from "encore.dev/config";

/**
 * Secret declarations for the contract-analysis service.
 * These are loaded from Encore's secret management.
 *
 * Encore secrets are accessed via the secret() function which returns
 * a function that when called returns the secret value.
 */

// Secret accessors
const jwtSecretLoader = secret("JWTSecret");
const geminiApiKeyLoader = secret("GeminiAPIKey");

// Cached values
let _jwtSecret: string | null = null;
let _geminiApiKey: string | null = null;
let _secretsValidated = false;

/**
 * Load and validate all secrets at startup
 * Call this once during service initialization
 */
export function validateSecrets(): {
  jwtSecret: boolean;
  geminiApiKey: boolean;
} {
  const result = {
    jwtSecret: false,
    geminiApiKey: false,
  };

  try {
    const secretValue = jwtSecretLoader();
    if (secretValue && secretValue.length >= 32) {
      _jwtSecret = secretValue;
      result.jwtSecret = true;
    } else if (secretValue) {
      console.error("JWTSecret is too short (minimum 32 characters required)");
    } else {
      console.error("JWTSecret is not configured");
    }
  } catch (error) {
    console.error("Failed to load JWTSecret:", error);
  }

  try {
    const secretValue = geminiApiKeyLoader();
    if (secretValue && secretValue.length > 0) {
      _geminiApiKey = secretValue;
      result.geminiApiKey = true;
    } else {
      console.warn(
        "GeminiAPIKey is not configured. AI enhancement will be disabled.",
      );
    }
  } catch (error) {
    console.warn(
      "Failed to load GeminiAPIKey. AI enhancement will be disabled:",
      error,
    );
  }

  _secretsValidated = true;
  return result;
}

/**
 * Get JWT secret (throws if not available)
 */
export function getJWTSecret(): string {
  if (!_secretsValidated) {
    validateSecrets();
  }
  if (!_jwtSecret) {
    throw new Error(
      "JWTSecret is not configured. Please set it in Encore secrets.",
    );
  }
  return _jwtSecret;
}

/**
 * Get Gemini API key (returns null if not available)
 */
export function getGeminiAPIKey(): string | null {
  if (!_secretsValidated) {
    validateSecrets();
  }
  return _geminiApiKey;
}

/**
 * Check if AI enhancement is available
 */
export function isAIEnhancementAvailable(): boolean {
  if (!_secretsValidated) {
    validateSecrets();
  }
  return _geminiApiKey !== null && _geminiApiKey.length > 0;
}

// Legacy exports for backward compatibility
// These match the original pattern used in auth.ts and ai.geminipool.ts
export const JWTSecret = jwtSecretLoader;
export const GeminiAPIKey = geminiApiKeyLoader() ?? "";

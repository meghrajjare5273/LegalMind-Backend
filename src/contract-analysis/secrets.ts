// src/contract-analysis/secrets.ts
import { secret } from "encore.dev/config";

// All secrets for the contract-analysis service are declared here.
// This file is inside the service root (same folder as encore.service.ts),
// so Encore is happy to load them from here.

export const JWTSecret = secret("JWTSecret") as unknown as string;
export const GeminiAPIKey = secret("GeminiAPIKey") as unknown as string;
// export const getGeminiAPIKeys = (): string => secrets.GeminiAPIKeys;

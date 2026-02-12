// src/contract-analysis/auth.ts
import { authHandler } from "encore.dev/auth";
import { Header, APIError, Gateway } from "encore.dev/api";
import { secret } from "encore.dev/config";
import jwt, { JwtPayload } from "jsonwebtoken";
import { AuthData } from "@/models/types";

interface AuthParams {
  authorization: Header<"Authorization">;
}

// Load JWT secret via Encore's secret management
const jwtSecretLoader = secret("JWTSecret");

/**
 * JWT payload interface
 */
interface TokenPayload extends JwtPayload {
  userId: string;
  email: string;
  role?: string;
}

/**
 * Auth handler for validating JWT tokens
 *
 * Security features:
 * - HS256 algorithm enforcement
 * - Token expiration validation
 * - Required claims validation (userId, email)
 */
export const auth = authHandler<AuthParams, AuthData>(
  async (params): Promise<AuthData> => {
    const authHeader = params.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      throw APIError.unauthenticated("Missing or invalid Authorization header");
    }

    const token = authHeader.replace("Bearer ", "");

    // Get secret value
    const secretValue = jwtSecretLoader();
    if (!secretValue) {
      console.error("JWTSecret is not configured");
      throw APIError.internal("Authentication service unavailable");
    }

    try {
      // Verify token with explicit algorithm restriction
      const payload = jwt.verify(token, secretValue, {
        algorithms: ["HS256"], // Restrict to HS256 algorithm only
      }) as TokenPayload;

      // Validate required claims
      if (!payload.userId || !payload.email) {
        throw APIError.unauthenticated(
          "Invalid token payload: missing required claims",
        );
      }

      // Validate userId format (should be a non-empty string)
      if (
        typeof payload.userId !== "string" ||
        payload.userId.trim().length === 0
      ) {
        throw APIError.unauthenticated("Invalid userId in token");
      }

      // Validate email format (basic check)
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(payload.email)) {
        throw APIError.unauthenticated("Invalid email format in token");
      }

      return {
        userID: payload.userId,
        email: payload.email,
        role: (payload.role === "admin" ? "admin" : "user") as "admin" | "user",
      };
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }

      if (error instanceof jwt.TokenExpiredError) {
        throw APIError.unauthenticated("Token has expired");
      }

      if (error instanceof jwt.JsonWebTokenError) {
        // Don't expose specific JWT errors to client
        throw APIError.unauthenticated("Invalid token");
      }

      if (error instanceof jwt.NotBeforeError) {
        throw APIError.unauthenticated("Token not yet valid");
      }

      // Log unexpected errors for debugging
      console.error("Unexpected token verification error:", error);
      throw APIError.unauthenticated("Token verification failed");
    }
  },
);

/**
 * Gateway configuration for the auth handler
 */
export const gateway = new Gateway({
  authHandler: auth,
});

// auth.ts
import { authHandler } from "encore.dev/auth";
import { Header, APIError, Gateway } from "encore.dev/api";
import { secret } from "encore.dev/config";
import jwt from "jsonwebtoken";
import { AuthData } from "@/models/types";
// import type { AuthData } from "/models/types";

interface AuthParams {
  authorization: Header<"Authorization">;
}

// JWT secret shared with Next.js frontend
const jwtSecret = secret("JWTSecret");

/**
 * Auth handler that validates JWT tokens from Next.js
 * Extracts userId, email, and role from the token
 */
export const auth = authHandler<AuthParams, AuthData>(
  async (params): Promise<AuthData> => {
    const authHeader = params.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      throw APIError.unauthenticated("Missing or invalid Authorization header");
    }

    const token = authHeader.replace("Bearer ", "");

    try {
      const payload = jwt.verify(token, jwtSecret()) as any;

      // Validate required fields
      if (!payload.userId || !payload.email) {
        throw APIError.unauthenticated("Invalid token payload");
      }

      return {
        userID: payload.userId,
        email: payload.email,
        role: payload.role || "user", // Default to 'user' if role not specified
      };
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        throw APIError.unauthenticated("Token has expired");
      }
      if (error instanceof jwt.JsonWebTokenError) {
        throw APIError.unauthenticated("Invalid token signature");
      }
      throw APIError.unauthenticated("Token verification failed");
    }
  },
);

/**
 * Gateway with CORS configuration for Next.js frontend
 */
export const gateway = new Gateway({
  authHandler: auth,
});

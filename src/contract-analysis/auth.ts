// src/contract-analysis/auth.ts
import { authHandler } from "encore.dev/auth";
import { Header, APIError, Gateway } from "encore.dev/api";
import { secret } from "encore.dev/config";
import jwt from "jsonwebtoken";
import { AuthData } from "@/models/types";

interface AuthParams {
  authorization: Header<"Authorization">;
}

const jwtSecret = secret("JWTSecret");

export const auth = authHandler<AuthParams, AuthData>(
  async (params): Promise<AuthData> => {
    const authHeader = params.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      throw APIError.unauthenticated("Missing or invalid Authorization header");
    }

    const token = authHeader.replace("Bearer ", "");

    try {
      const payload = jwt.verify(token, jwtSecret()) as any;

      if (!payload.userId || !payload.email) {
        throw APIError.unauthenticated("Invalid token payload");
      }

      return {
        userID: payload.userId,
        email: payload.email,
        role: payload.role || "user",
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

export const gateway = new Gateway({
  authHandler: auth,
});

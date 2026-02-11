// rate-limiter.ts
import { APIError } from "encore.dev/api";
import type { AuthData, RateLimitError } from "@/models/types";

interface RateLimitEntry {
  userId: string;
  lastRequestAt: Date;
  requestCount: number;
}

/**
 * In-memory rate limiter with sliding window
 * - Regular users: 1 request per 10 minutes
 * - Admin users: Unlimited requests
 */
class RateLimiter {
  private limits = new Map<string, RateLimitEntry>();
  private readonly windowMs = 10 * 60 * 1000; // 10 minutes in milliseconds

  /**
   * Check if user can proceed with request
   * @param auth - User authentication data
   * @returns true if allowed, throws APIError if rate limited
   */
  checkRateLimit(auth: AuthData): boolean {
    // Admins bypass rate limiting
    if (auth.role === "admin") {
      return true;
    }

    const now = new Date();
    const entry = this.limits.get(auth.userID);

    // First request from this user
    if (!entry) {
      this.limits.set(auth.userID, {
        userId: auth.userID,
        lastRequestAt: now,
        requestCount: 1,
      });
      return true;
    }

    const timeSinceLastRequest = now.getTime() - entry.lastRequestAt.getTime();

    // Still within rate limit window
    if (timeSinceLastRequest < this.windowMs) {
      const retryAfterSeconds = Math.ceil(
        (this.windowMs - timeSinceLastRequest) / 1000,
      );
      const nextAvailableAt = new Date(
        entry.lastRequestAt.getTime() + this.windowMs,
      );

      const errorDetails: RateLimitError = {
        error: "rate_limit_exceeded",
        message: `Rate limit exceeded. You can analyze 1 contract every 10 minutes. Please try again in ${Math.ceil(retryAfterSeconds / 60)} minutes.`,
        retryAfter: retryAfterSeconds,
        nextAvailableAt: nextAvailableAt.toISOString(),
      };

      throw APIError.resourceExhausted(
        errorDetails.message,
        errorDetails as any,
      );
    }

    // Outside window, reset and allow
    entry.lastRequestAt = now;
    entry.requestCount = 1;
    return true;
  }

  /**
   * Clear rate limit for a specific user (useful for testing)
   */
  clearUserLimit(userId: string): void {
    this.limits.delete(userId);
  }

  /**
   * Get current rate limit status for a user
   */
  getUserStatus(userId: string): {
    hasLimit: boolean;
    nextAvailableAt: Date | null;
    remainingTime: number | null;
  } {
    const entry = this.limits.get(userId);
    if (!entry) {
      return { hasLimit: false, nextAvailableAt: null, remainingTime: null };
    }

    const now = new Date();
    const timeSinceLastRequest = now.getTime() - entry.lastRequestAt.getTime();

    if (timeSinceLastRequest >= this.windowMs) {
      return { hasLimit: false, nextAvailableAt: null, remainingTime: null };
    }

    const nextAvailableAt = new Date(
      entry.lastRequestAt.getTime() + this.windowMs,
    );
    const remainingTime = Math.ceil(
      (this.windowMs - timeSinceLastRequest) / 1000,
    );

    return { hasLimit: true, nextAvailableAt, remainingTime };
  }
}

// Export singleton instance
export const rateLimiter = new RateLimiter();

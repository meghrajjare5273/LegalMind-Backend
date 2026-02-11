// utils/cache.ts
import crypto from "crypto";

interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

/**
 * In-memory cache with TTL support
 * Used for caching analysis results
 */
class CacheManager<T = any> {
  private cache = new Map<string, CacheEntry<T>>();
  private readonly defaultTTL = 3600 * 1000; // 1 hour in milliseconds

  /**
   * Generate cache key from input data
   */
  generateKey(data: string | Buffer): string {
    const hash = crypto.createHash("sha256");
    hash.update(typeof data === "string" ? data : data.toString());
    return hash.digest("hex");
  }

  /**
   * Get item from cache
   */
  get(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  /**
   * Set item in cache with TTL
   */
  set(key: string, data: T, ttlMs?: number): void {
    const expiresAt = Date.now() + (ttlMs || this.defaultTTL);
    this.cache.set(key, { data, expiresAt });
  }

  /**
   * Check if key exists and is not expired
   */
  has(key: string): boolean {
    return this.get(key) !== null;
  }

  /**
   * Delete item from cache
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Clean up expired entries
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    activeEntries: number;
    expiredEntries: number;
  } {
    const now = Date.now();
    let activeEntries = 0;
    let expiredEntries = 0;

    for (const entry of this.cache.values()) {
      if (now > entry.expiresAt) {
        expiredEntries++;
      } else {
        activeEntries++;
      }
    }

    return {
      size: this.cache.size,
      activeEntries,
      expiredEntries,
    };
  }
}

// Export singleton instance
export const cache = new CacheManager();

// Schedule periodic cleanup every 10 minutes
if (typeof setInterval !== "undefined") {
  setInterval(
    () => {
      cache.cleanup();
    },
    10 * 60 * 1000,
  );
}

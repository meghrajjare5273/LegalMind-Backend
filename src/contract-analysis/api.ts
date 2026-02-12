// api.ts
import { api, APIError, ErrCode } from "encore.dev/api";
import { getAuthData } from "~encore/auth";
import { rateLimiter } from "./rate-limiter";
import { PDFProcessor } from "@/services/pdf.processor";
import { ContractAnalyzer } from "@/services/contract.analyzer";
import type { AnalysisResponse } from "@/models/types";
import {
  validateUserRole,
  validateFilename,
  validateFileSize,
} from "@/utils/validation";

// Initialize analyzer singleton
const analyzer = new ContractAnalyzer();

// Configuration
const REQUEST_TIMEOUT_MS = 60000; // 60 seconds timeout

// Map ErrCode to HTTP status codes
const errCodeToStatus = (code: ErrCode): number => {
  const mapping: Record<ErrCode, number> = {
    [ErrCode.OK]: 200,
    [ErrCode.Canceled]: 499,
    [ErrCode.Unknown]: 500,
    [ErrCode.InvalidArgument]: 400,
    [ErrCode.DeadlineExceeded]: 504,
    [ErrCode.NotFound]: 404,
    [ErrCode.AlreadyExists]: 409,
    [ErrCode.PermissionDenied]: 403,
    [ErrCode.ResourceExhausted]: 429,
    [ErrCode.FailedPrecondition]: 400,
    [ErrCode.Aborted]: 409,
    [ErrCode.OutOfRange]: 400,
    [ErrCode.Unimplemented]: 501,
    [ErrCode.Internal]: 500,
    [ErrCode.Unavailable]: 503,
    [ErrCode.DataLoss]: 500,
    [ErrCode.Unauthenticated]: 401,
  };
  return mapping[code] ?? 500;
};

/**
 * Health check endpoint with dependency status
 */
export const health = api(
  { method: "GET", path: "/health", expose: true, auth: false },
  async (): Promise<{
    status: string;
    version: string;
    timestamp: string;
    uptime: number;
    services: {
      analyzer: boolean;
      ai: boolean;
      cache: boolean;
    };
  }> => {
    // Import dynamically to avoid circular dependencies
    const { geminiPool } = await import("@/services/ai.geminipool");
    const { cache } = await import("@/utils/cache");

    return {
      status: "healthy",
      version: "1.0.0",
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      services: {
        analyzer: true,
        ai: geminiPool.isAvailable(),
        cache: cache.getStats().activeEntries >= 0,
      },
    };
  },
);

/**
 * Contract analysis endpoint.
 * Implemented as a raw endpoint to support multipart/form-data file upload.
 *
 * Security: Requires authentication and enforces rate limiting.
 */
export const analyzeContract = api.raw(
  {
    method: "POST",
    path: "/contract-analysis/analyze",
    expose: true,
    auth: true, // Authentication enabled
  },
  async (req, resp): Promise<void> => {
    // Create timeout promise
    const timeoutId = setTimeout(() => {
      throw APIError.deadlineExceeded(
        "Request processing timed out. Please try with a smaller document.",
      );
    }, REQUEST_TIMEOUT_MS);

    try {
      // Authentication check
      const auth = getAuthData();
      if (!auth) {
        throw APIError.unauthenticated("Authentication required");
      }

      // Rate limiting check
      rateLimiter.checkRateLimit(auth);

      const contentType = req.headers["content-type"] || "";
      if (!contentType.startsWith("multipart/form-data")) {
        throw APIError.invalidArgument(
          "Content-Type must be multipart/form-data",
        );
      }

      // Collect raw body with size limit
      const chunks: Uint8Array[] = [];
      let totalSize = 0;
      const maxBodySize = 15 * 1024 * 1024; // 15MB max body size

      for await (const chunk of req) {
        totalSize += chunk.length;
        if (totalSize > maxBodySize) {
          throw APIError.invalidArgument(
            "Request body too large. Maximum size is 15MB.",
          );
        }
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);

      // Parse multipart manually (simple, assumes single file + userRole field)
      const boundaryMatch = contentType.match(/boundary=(.+)$/);
      if (!boundaryMatch) {
        throw APIError.invalidArgument("Invalid multipart boundary");
      }
      const boundary = boundaryMatch[1];

      const parts = body.toString("binary").split(`--${boundary}`);
      let fileBuffer: Buffer | null = null;
      let rawFilename = "contract.pdf";
      let rawUserRole = "Neutral Observer";

      for (const part of parts) {
        if (part === "--\r\n" || !part.trim()) continue;

        const [rawHeaders, rawContent] = part.split("\r\n\r\n");
        if (!rawHeaders || !rawContent) continue;

        const headers = rawHeaders.split("\r\n").filter(Boolean);
        const contentDisposition = headers.find((h) =>
          h.toLowerCase().startsWith("content-disposition:"),
        );
        if (!contentDisposition) continue;

        const nameMatch = contentDisposition.match(/name="([^"]+)"/);
        const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);

        const content = rawContent.slice(0, rawContent.lastIndexOf("\r\n"));

        if (filenameMatch) {
          // File part
          rawFilename = filenameMatch[1];
          const startIndex = body.indexOf(content, 0, "binary");
          const endIndex = startIndex + Buffer.byteLength(content, "binary");
          fileBuffer = body.subarray(startIndex, endIndex);
        } else if (nameMatch && nameMatch[1] === "user_role") {
          rawUserRole = content.trim() || "Neutral Observer";
        }
      }

      if (!fileBuffer) {
        throw APIError.invalidArgument("Missing PDF file in 'file' field");
      }

      // Validate and sanitize inputs
      const userRole = validateUserRole(rawUserRole);
      const filename = validateFilename(rawFilename);

      // Validate file size
      try {
        validateFileSize(fileBuffer);
      } catch (error) {
        throw APIError.invalidArgument((error as Error).message);
      }

      // Validate PDF and extract text
      if (!PDFProcessor.isValidPDF(fileBuffer)) {
        throw APIError.invalidArgument("Invalid PDF file format");
      }

      const text = await PDFProcessor.extractTextFromPDF(fileBuffer);
      if (!text.trim()) {
        throw APIError.invalidArgument("No readable text found in PDF");
      }

      // Analyze contract
      const result: AnalysisResponse = await analyzer.analyzeContract(text, {
        userRole,
        filename,
        userId: auth.userID,
      });

      // Clear timeout on success
      clearTimeout(timeoutId);

      // Respond
      resp.setHeader("content-type", "application/json");
      resp.writeHead(200);
      resp.end(JSON.stringify(result));
    } catch (err) {
      clearTimeout(timeoutId);
      const e = err as any;

      if (e instanceof APIError) {
        resp.setHeader("content-type", "application/json");
        resp.writeHead(errCodeToStatus(e.code));
        resp.end(
          JSON.stringify({
            code: e.code,
            message: e.message,
            details: e.details ?? null,
          }),
        );
        return;
      }

      // Log unexpected errors (in production, use proper logging)
      console.error("Unexpected error in analyzeContract:", e);

      resp.setHeader("content-type", "application/json");
      resp.writeHead(500);
      resp.end(
        JSON.stringify({
          code: ErrCode.Internal,
          message: "Internal server error",
        }),
      );
    }
  },
);

/**
 * Development-only endpoint for testing without authentication.
 * This should be disabled in production.
 */
export const analyzeContractDev = api.raw(
  {
    method: "POST",
    path: "/contract-analysis/analyze-dev",
    expose: true,
    auth: false, // No auth for development testing
  },
  async (req, resp): Promise<void> => {
    // Check if running in development
    if (process.env.NODE_ENV === "production") {
      resp.setHeader("content-type", "application/json");
      resp.writeHead(404);
      resp.end(JSON.stringify({ message: "Not found" }));
      return;
    }

    const timeoutId = setTimeout(() => {
      throw APIError.deadlineExceeded(
        "Request processing timed out. Please try with a smaller document.",
      );
    }, REQUEST_TIMEOUT_MS);

    try {
      const contentType = req.headers["content-type"] || "";
      if (!contentType.startsWith("multipart/form-data")) {
        throw APIError.invalidArgument(
          "Content-Type must be multipart/form-data",
        );
      }

      // Collect raw body with size limit
      const chunks: Uint8Array[] = [];
      let totalSize = 0;
      const maxBodySize = 15 * 1024 * 1024;

      for await (const chunk of req) {
        totalSize += chunk.length;
        if (totalSize > maxBodySize) {
          throw APIError.invalidArgument(
            "Request body too large. Maximum size is 15MB.",
          );
        }
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);

      const boundaryMatch = contentType.match(/boundary=(.+)$/);
      if (!boundaryMatch) {
        throw APIError.invalidArgument("Invalid multipart boundary");
      }
      const boundary = boundaryMatch[1];

      const parts = body.toString("binary").split(`--${boundary}`);
      let fileBuffer: Buffer | null = null;
      let rawFilename = "contract.pdf";
      let rawUserRole = "Neutral Observer";

      for (const part of parts) {
        if (part === "--\r\n" || !part.trim()) continue;

        const [rawHeaders, rawContent] = part.split("\r\n\r\n");
        if (!rawHeaders || !rawContent) continue;

        const headers = rawHeaders.split("\r\n").filter(Boolean);
        const contentDisposition = headers.find((h) =>
          h.toLowerCase().startsWith("content-disposition:"),
        );
        if (!contentDisposition) continue;

        const nameMatch = contentDisposition.match(/name="([^"]+)"/);
        const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);

        const content = rawContent.slice(0, rawContent.lastIndexOf("\r\n"));

        if (filenameMatch) {
          rawFilename = filenameMatch[1];
          const startIndex = body.indexOf(content, 0, "binary");
          const endIndex = startIndex + Buffer.byteLength(content, "binary");
          fileBuffer = body.subarray(startIndex, endIndex);
        } else if (nameMatch && nameMatch[1] === "user_role") {
          rawUserRole = content.trim() || "Neutral Observer";
        }
      }

      if (!fileBuffer) {
        throw APIError.invalidArgument("Missing PDF file in 'file' field");
      }

      // Validate and sanitize inputs
      const userRole = validateUserRole(rawUserRole);
      const filename = validateFilename(rawFilename);

      try {
        validateFileSize(fileBuffer);
      } catch (error) {
        throw APIError.invalidArgument((error as Error).message);
      }

      if (!PDFProcessor.isValidPDF(fileBuffer)) {
        throw APIError.invalidArgument("Invalid PDF file format");
      }

      const text = await PDFProcessor.extractTextFromPDF(fileBuffer);
      if (!text.trim()) {
        throw APIError.invalidArgument("No readable text found in PDF");
      }

      // Analyze contract with dev user
      const result: AnalysisResponse = await analyzer.analyzeContract(text, {
        userRole,
        filename,
        userId: "dev-user",
      });

      clearTimeout(timeoutId);

      resp.setHeader("content-type", "application/json");
      resp.writeHead(200);
      resp.end(JSON.stringify(result));
    } catch (err) {
      clearTimeout(timeoutId);
      const e = err as any;

      if (e instanceof APIError) {
        resp.setHeader("content-type", "application/json");
        resp.writeHead(errCodeToStatus(e.code));
        resp.end(
          JSON.stringify({
            code: e.code,
            message: e.message,
            details: e.details ?? null,
          }),
        );
        return;
      }

      console.error("Unexpected error in analyzeContractDev:", e);

      resp.setHeader("content-type", "application/json");
      resp.writeHead(500);
      resp.end(
        JSON.stringify({
          code: ErrCode.Internal,
          message: "Internal server error",
        }),
      );
    }
  },
);

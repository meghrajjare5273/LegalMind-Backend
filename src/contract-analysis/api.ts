// api.ts
import {
  api,
  APIError,
  ErrCode,
  RawRequest,
  RawResponse,
} from "encore.dev/api";
import { getAuthData } from "~encore/auth";
import { rateLimiter } from "./rate-limiter";
import { PDFProcessor } from "@/services/pdf.processor";
import { ContractAnalyzer } from "@/services/contract.analyzer";
import type { AnalysisResponse } from "@/models/types";

// Initialize analyzer singleton
const analyzer = new ContractAnalyzer();

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
 * Health check (simple JSON)
 */
export const health = api(
  { method: "GET", path: "/health", expose: true, auth: false },
  async (): Promise<{ status: string; version: string }> => {
    return {
      status: "healthy",
      version: "1.0.0",
    };
  },
);

/**
 * Contract analysis endpoint.
 * Implemented as a raw endpoint to support multipart/form-data file upload.
 */
export const analyzeContract = api.raw(
  {
    method: "POST",
    path: "/contract-analysis/analyze",
    expose: true,
    auth: true,
  },
  async (req, resp): Promise<void> => {
    try {
      const auth = getAuthData();
      if (!auth) {
        throw APIError.unauthenticated("Authentication required");
      }

      // Rate limiting
      rateLimiter.checkRateLimit(auth);

      const contentType = req.headers["content-type"] || "";
      if (!contentType.startsWith("multipart/form-data")) {
        throw APIError.invalidArgument(
          "Content-Type must be multipart/form-data",
        );
      }

      // Collect raw body - use any[] to bypass strict type checking
      const chunks: any[] = [];
      for await (const chunk of req) {
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
      let filename = "contract.pdf";
      let userRole = "Neutral Observer";

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
          filename = filenameMatch[1];
          const startIndex = body.indexOf(content, 0, "binary");
          const endIndex = startIndex + Buffer.byteLength(content, "binary");
          fileBuffer = body.subarray(startIndex, endIndex);
        } else if (nameMatch && nameMatch[1] === "user_role") {
          userRole = content.trim() || "Neutral Observer";
        }
      }

      if (!fileBuffer) {
        throw APIError.invalidArgument("Missing PDF file in 'file' field");
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
      });

      // Respond
      resp.setHeader("content-type", "application/json");
      resp.writeHead(200);
      resp.end(JSON.stringify(result));
    } catch (err) {
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

// utils/validation.ts

/**
 * Allowed user roles for contract analysis
 * These are whitelisted to prevent prompt injection attacks
 */
export const ALLOWED_USER_ROLES = [
  "Neutral Observer",
  "Business Owner",
  "Employee",
  "Contractor",
  "Service Provider",
  "Client",
  "Landlord",
  "Tenant",
  "Employer",
  "Job Seeker",
  "Vendor",
  "Supplier",
  "Partner",
  "Investor",
  "Lender",
  "Borrower",
] as const;

export type AllowedUserRole = (typeof ALLOWED_USER_ROLES)[number];

/**
 * Validate and sanitize user role input
 * Returns a valid role or default if invalid
 */
export function validateUserRole(role: string | undefined): AllowedUserRole {
  if (!role) {
    return "Neutral Observer";
  }

  // Trim and check against whitelist
  const trimmed = role.trim();

  // Check if it's an exact match (case-sensitive)
  const exactMatch = ALLOWED_USER_ROLES.find((r) => r === trimmed);
  if (exactMatch) {
    return exactMatch;
  }

  // Check case-insensitive match
  const caseInsensitiveMatch = ALLOWED_USER_ROLES.find(
    (r) => r.toLowerCase() === trimmed.toLowerCase(),
  );
  if (caseInsensitiveMatch) {
    return caseInsensitiveMatch;
  }

  // Return default if not found
  console.warn(`Invalid user role "${trimmed}" provided, using default`);
  return "Neutral Observer";
}

/**
 * Validate filename to prevent path traversal and other attacks
 */
export function validateFilename(filename: string | undefined): string {
  if (!filename) {
    return "contract.pdf";
  }

  // Remove any path separators
  const sanitized = filename.replace(/[\/\\]/g, "_");

  // Remove null bytes
  const cleaned = sanitized.replace(/\0/g, "");

  // Limit length
  const truncated = cleaned.slice(0, 255);

  // Ensure it has a PDF extension
  if (!truncated.toLowerCase().endsWith(".pdf")) {
    return `${truncated}.pdf`;
  }

  return truncated;
}

/**
 * Validate PDF file size (max 10MB)
 */
export function validateFileSize(buffer: Buffer): void {
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB

  if (!buffer || buffer.length === 0) {
    throw new Error("Empty file provided");
  }

  if (buffer.length > MAX_SIZE) {
    throw new Error(`File size exceeds maximum allowed size of 10MB`);
  }
}

/**
 * Sanitize text for safe logging (remove sensitive patterns)
 */
export function sanitizeForLogging(text: string, maxLength: number = 200): string {
  if (!text) return "";

  // Remove potential sensitive patterns
  const sanitized = text
    .replace(/\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g, "[CARD]") // Credit card numbers
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[SSN]") // SSN
    .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, "[EMAIL]") // Email
    .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[IP]"); // IP addresses

  return sanitized.slice(0, maxLength) + (sanitized.length > maxLength ? "..." : "");
}

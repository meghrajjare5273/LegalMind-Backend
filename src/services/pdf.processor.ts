// services/pdf/processor.ts
import * as pdfjsLib from "pdfjs-dist";
import { APIError } from "encore.dev/api";

/**
 * PDF processor using pdfjs-dist for low-level text extraction
 * Provides detailed extraction capabilities for contract analysis
 */
export class PDFProcessor {
  private static readonly MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  /**
   * Validate PDF file format
   */
  static isValidPDF(buffer: Buffer): boolean {
    if (!buffer || buffer.length === 0) {
      return false;
    }

    // Check PDF magic number (PDF files start with %PDF-)
    const pdfHeader = buffer.slice(0, 5).toString("binary");
    return pdfHeader === "%PDF-";
  }

  /**
   * Extract text from PDF buffer
   */
  static async extractTextFromPDF(buffer: Buffer): Promise<string> {
    try {
      // Validate file
      if (!this.isValidPDF(buffer)) {
        throw APIError.invalidArgument("Invalid PDF file format");
      }

      if (buffer.length > this.MAX_FILE_SIZE) {
        throw APIError.invalidArgument("PDF file size exceeds 10MB limit");
      }

      // Load PDF document
      const loadingTask = pdfjsLib.getDocument({
        data: new Uint8Array(buffer),
        useSystemFonts: true,
        standardFontDataUrl: undefined,
      });

      const pdf = await loadingTask.promise;
      const textParts: string[] = [];

      // Extract text from each page
      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const textContent = await page.getTextContent();

        // Combine text items with proper spacing
        const pageText = textContent.items
          .map((item: any) => {
            if ("str" in item) {
              return item.str;
            }
            return "";
          })
          .join(" ");

        textParts.push(pageText);
      }

      const fullText = textParts.join("\n\n");

      // Clean up extracted text
      return this.cleanExtractedText(fullText);
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      throw APIError.internal(
        `PDF extraction failed: ${(error as Error).message}`,
      );
    }
  }

  /**
   * Clean and normalize extracted text
   */
  private static cleanExtractedText(text: string): string {
    return (
      text
        // Remove excessive whitespace
        .replace(/\s+/g, " ")
        // Remove multiple newlines
        .replace(/\n{3,}/g, "\n\n")
        // Trim each line
        .split("\n")
        .map((line) => line.trim())
        .join("\n")
        // Final trim
        .trim()
    );
  }

  /**
   * Extract metadata from PDF
   */
  static async extractMetadata(buffer: Buffer): Promise<{
    numPages: number;
    title?: string;
    author?: string;
    creationDate?: string;
  }> {
    try {
      const loadingTask = pdfjsLib.getDocument({
        data: new Uint8Array(buffer),
      });

      const pdf = await loadingTask.promise;
      const metadata = await pdf.getMetadata();

      // Safely access metadata properties using type assertion
      const info = metadata.info as Record<string, any> | undefined;

      return {
        numPages: pdf.numPages,
        title: info?.Title,
        author: info?.Author,
        creationDate: info?.CreationDate,
      };
    } catch (error) {
      throw APIError.internal(
        `Metadata extraction failed: ${(error as Error).message}`,
      );
    }
  }

  /**
   * Check if PDF is readable (contains extractable text)
   */
  static async hasReadableText(buffer: Buffer): Promise<boolean> {
    try {
      const text = await this.extractTextFromPDF(buffer);
      return text.trim().length > 0;
    } catch {
      return false;
    }
  }
}

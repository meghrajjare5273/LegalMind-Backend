import fitz  # PyMuPDF
import logging
from typing import List
import gc

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Optimized PDF processing service for contract analysis"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file content with memory optimization"""
        doc = None
        try:
            doc = fitz.open("pdf", file_content)
            text_parts = []
            
            # Process pages in batches to reduce memory usage
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text and isinstance(page_text, str):
                    # Clean text immediately to reduce memory
                    cleaned_text = PDFProcessor._clean_extracted_text(page_text)
                    if cleaned_text.strip():
                        text_parts.append(cleaned_text)
                
                # Clear page from memory
                page = None
                
                # Force garbage collection every 10 pages for large documents
                if page_num % 10 == 9:
                    gc.collect()
            
            final_text = ' '.join(text_parts)
            logger.info(f"Successfully extracted text from PDF ({len(final_text)} characters)")
            return final_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
        finally:
            if doc:
                doc.close()
            # Final cleanup
            gc.collect()
    
    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Optimized text cleaning"""
        if not text:
            return ""
        
        # Single pass cleaning with compiled regex for better performance
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Replace common Unicode characters
        replacements = {'\uf0b7': '•', '\uf0a7': '§', '\n': ' ', '\r': ' '}
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    @staticmethod
    def is_valid_pdf(file_content: bytes) -> bool:
        """Lightweight PDF validation"""
        try:
            # Quick validation - check PDF header
            if not file_content.startswith(b'%PDF-'):
                return False
                
            # Quick structure check
            doc = fitz.open("pdf", file_content)
            is_valid = doc.page_count > 0
            doc.close()
            return is_valid
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

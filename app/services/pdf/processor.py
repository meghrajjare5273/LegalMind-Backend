import fitz  # PyMuPDF
import logging
from typing import List

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processing service for contract analysis"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file content"""
        try:
            doc = fitz.open("pdf", file_content)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if isinstance(page_text, list):
                    page_text = ' '.join(str(item) for item in page_text)
                elif not isinstance(page_text, str):
                    page_text = str(page_text)
                
                text += page_text + "\n"
                
            doc.close()
            text = PDFProcessor._clean_extracted_text(text)
            
            logger.info(f"Successfully extracted text from PDF ({len(text)} characters)")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """Clean and normalize extracted text"""
        if not isinstance(text, str):
            text = str(text)
        
        text = ' '.join(text.split())
        text = text.replace('\uf0b7', '•')
        text = text.replace('\uf0a7', '§')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        return text.strip()
    
    @staticmethod
    def is_valid_pdf(file_content: bytes) -> bool:
        """Check if the file content is a valid PDF"""
        try:
            doc = fitz.open("pdf", file_content)
            if doc.page_count > 0:
                doc[0].get_text()
            doc.close()
            return True
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

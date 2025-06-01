import fitz  # PyMuPDF
import logging
# from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extracts text from a PDF file using Fitz (PyMuPDF).
    
    Args:
        file_content (bytes): The content of the PDF file.
    
    Returns:
        str: The extracted text from the PDF.
        
    Raises:
        Exception: If there's an error processing the PDF.
    """
    try:
        # Open PDF from bytes
        doc = fitz.open("pdf", file_content)
        text = ""
        
        # Extract text from each page
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text
            
        doc.close()
        
        # Clean up the text
        text = clean_extracted_text(text)
        
        logger.info(f"Successfully extracted text from PDF ({len(text)} characters)")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text (str): Raw extracted text.
        
    Returns:
        str: Cleaned text.
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace('\uf0b7', '•')  # Replace bullet character
    text = text.replace('\uf0a7', '§')  # Replace section character
    
    return text

def extract_metadata_from_pdf(file_content: bytes) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        file_content (bytes): The content of the PDF file.
        
    Returns:
        dict: PDF metadata including title, author, subject, etc.
    """
    try:
        doc = fitz.open("pdf", file_content)
        metadata = doc.metadata
        
        # Add additional information
        metadata['page_count'] = doc.page_count
        
        doc.close()
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF: {str(e)}")
        return {}

def extract_text_by_page(file_content: bytes) -> list[str]:
    """
    Extract text from PDF page by page.
    
    Args:
        file_content (bytes): The content of the PDF file.
        
    Returns:
        list[str]: List of text content for each page.
    """
    try:
        doc = fitz.open("pdf", file_content)
        pages_text = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = clean_extracted_text(page.get_text())
            pages_text.append(page_text)
            
        doc.close()
        
        return pages_text
        
    except Exception as e:
        logger.error(f"Error extracting text by page from PDF: {str(e)}")
        raise Exception(f"Failed to extract text by page from PDF: {str(e)}")

def is_valid_pdf(file_content: bytes) -> bool:
    """
    Check if the file content is a valid PDF.
    
    Args:
        file_content (bytes): The content to validate.
        
    Returns:
        bool: True if valid PDF, False otherwise.
    """
    try:
        doc = fitz.open("pdf", file_content)
        doc.close()
        return True
    except:
        return False


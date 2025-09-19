# app/core/startup.py
from contextlib import asynccontextmanager
from services.contract.analyzer import HybridContractAnalyzer
from utils.cache import CacheManager
import logging
import os
import nltk 
from pathlib import Path

logger = logging.getLogger(__name__)

def initialize_nltk():
    """Initialize NLTK data for production deployment"""
    
    # Set environment variable for NLTK data path
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.environ['NLTK_DATA'] = nltk_data_path
    
    # Ensure directory exists
    Path(nltk_data_path).mkdir(exist_ok=True)
    
    # Add to NLTK's data path
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    
    # Download required data
    required_packages = ['punkt_tab', 'stopwords']
    
    for package in required_packages:
        try:
            # Try to find the package first
            if package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
            logger.info(f"NLTK package '{package}' found")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK package: {package}")
                result = nltk.download(package, download_dir=nltk_data_path, quiet=False)
                if result:
                    logger.info(f"Successfully downloaded: {package}")
                else:
                    logger.warning(f"Download may have failed for: {package}")
            except Exception as e:
                logger.error(f"Error downloading {package}: {str(e)}")
                # Try alternative download without custom directory
                try:
                    nltk.download(package, quiet=False)
                    logger.info(f"Downloaded {package} to default location")
                except Exception as alt_error:
                    logger.error(f"Complete download failure for {package}: {str(alt_error)}")


analyzer: HybridContractAnalyzer | None = None
cache_manager = CacheManager()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    global analyzer
    logger.info("Initializing Hybrid Contract Analyzer…")
    analyzer = HybridContractAnalyzer()
    logger.info("Initailizing NLTK Libraries.....")
    initialize_nltk()
    yield

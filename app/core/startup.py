# app/core/startup.py
from contextlib import asynccontextmanager
from services.contract.analyzer import HybridContractAnalyzer
from utils.cache import CacheManager
import logging

analyzer: HybridContractAnalyzer | None = None
cache_manager = CacheManager()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    global analyzer
    logger.info("Initializing Hybrid Contract Analyzer…")
    analyzer = HybridContractAnalyzer()
    yield

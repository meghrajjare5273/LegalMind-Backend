import functools
import hashlib
import json
import logging
from typing import Any, Dict, Optional
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class CacheManager:
    """Centralized cache management"""
    
    def __init__(self):
        self.analysis_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour
        self.ai_cache = TTLCache(maxsize=500, ttl=1800)       # 30 minutes
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        return {
            "analysis_cache_size": len(self.analysis_cache),
            "ai_cache_size": len(self.ai_cache),
            "analysis_cache_hits": getattr(self.analysis_cache, 'hits', 0),
            "ai_cache_hits": getattr(self.ai_cache, 'hits', 0)
        }

# Global cache manager instance
cache_manager = CacheManager()

def cache_result(ttl: int = 3600, cache_type: str = "analysis"):
    """Decorator for caching function results"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Select cache
            cache = (cache_manager.ai_cache if cache_type == "ai" 
                    else cache_manager.analysis_cache)
            
            # Check cache
            if cache_key in cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[cache_key]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = result
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

# src/utils/cache_setup.py
"""
LangChain InMemoryCache — Global Setup
=======================================
Call setup_cache() once at app startup (imported by workflow.py).
After this runs, ALL LangChain LLM calls in the entire project
automatically cache their responses. No other file needs to change.

Why InMemoryCache?
- Zero dependencies — no Redis, no database, no config
- Built into LangChain core
- Survives for the lifetime of the Python process
- Perfect for MVP — upgrade to Redis in production if needed

Limitation: Cache resets when the app restarts (in-memory only).
That is acceptable for this portfolio MVP.
"""

import logging
try:
    # LangChain >= 0.1.0
    from langchain.globals import set_llm_cache
except ImportError:
    # LangChain 1.x
    from langchain_core.globals import set_llm_cache

try:
    from langchain_community.cache import InMemoryCache
except ImportError:
    from langchain.cache import InMemoryCache
from src.utils.config import CACHE_ENABLED

logger = logging.getLogger(__name__)

# This holds the cache instance so we can inspect it in tests
_cache_instance: InMemoryCache | None = None


def setup_cache() -> InMemoryCache | None:
    """
    Enable LangChain's InMemoryCache globally.

    Call this ONCE at app startup. After this call, every LangChain
    LLM call across the entire project is automatically cached.

    Returns the cache instance (useful for tests), or None if caching
    is disabled via CACHE_ENABLED=false in .env
    """
    global _cache_instance

    if not CACHE_ENABLED:
        logger.info("[cache] Caching disabled via CACHE_ENABLED=false")
        return None

    _cache_instance = InMemoryCache()
    set_llm_cache(_cache_instance)
    logger.info("[cache] LangChain InMemoryCache enabled globally")
    return _cache_instance


def get_cache_instance() -> InMemoryCache | None:
    """Return the active cache instance. Used by tests to inspect state."""
    return _cache_instance
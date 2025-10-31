"""
✅ Retry utilities with tenacity for error recovery
"""
import logging
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

# 🔧 FIXED: Enhanced retry configuration for vector store operations
# Added more exception types for better error recovery
vector_store_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        ConnectionError,
        TimeoutError,
        OSError,  # Network issues
        IOError   # I/O errors
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True  # Re-raise the last exception after all retries fail
)

# Retry for database operations
database_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Retry for LLM operations (more attempts, longer waits)
llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Retry for embedding operations
embedding_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

def with_fallback(fallback_value):
    """
    Decorator that returns a fallback value if all retries fail

    Usage:
        @with_fallback([])
        @vector_store_retry
        def search_similar(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"All retries failed for {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator

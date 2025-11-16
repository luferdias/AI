"""Retry utilities with exponential backoff."""

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

logger = logging.getLogger("ingestion_agent.retry")

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff delay
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    delay = backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator


class RetryManager:
    """Manages retry logic for operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """Initialize retry manager.
        
        Args:
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"All retry attempts failed for {func.__name__}")
                    raise
                
                delay = self.backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
        
        raise last_exception

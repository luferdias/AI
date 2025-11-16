"""Unit tests for retry logic."""

import asyncio
import pytest

from ingestion_agent.retry import retry_with_backoff, RetryManager


@pytest.mark.asyncio
async def test_retry_with_backoff_success():
    """Test retry decorator with successful operation."""
    call_count = 0
    
    @retry_with_backoff(max_retries=3, backoff_factor=0.1)
    async def successful_operation():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = await successful_operation()
    
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_eventual_success():
    """Test retry decorator with eventual success."""
    call_count = 0
    
    @retry_with_backoff(max_retries=3, backoff_factor=0.1)
    async def eventually_successful():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Not ready yet")
        return "success"
    
    result = await eventually_successful()
    
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_max_retries():
    """Test retry decorator reaches max retries."""
    call_count = 0
    
    @retry_with_backoff(max_retries=2, backoff_factor=0.1)
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")
    
    with pytest.raises(ValueError):
        await always_fails()
    
    assert call_count == 3  # Initial attempt + 2 retries


@pytest.mark.asyncio
async def test_retry_manager_success():
    """Test RetryManager with successful operation."""
    manager = RetryManager(max_retries=3, backoff_factor=0.1)
    
    async def successful_func():
        return "success"
    
    result = await manager.execute_with_retry(successful_func)
    
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_manager_with_args():
    """Test RetryManager with function arguments."""
    manager = RetryManager(max_retries=3, backoff_factor=0.1)
    
    async def func_with_args(x, y):
        return x + y
    
    result = await manager.execute_with_retry(func_with_args, 5, 10)
    
    assert result == 15


@pytest.mark.asyncio
async def test_retry_manager_eventual_success():
    """Test RetryManager with eventual success."""
    manager = RetryManager(max_retries=3, backoff_factor=0.1)
    call_count = 0
    
    async def eventually_works():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Connection failed")
        return "connected"
    
    result = await manager.execute_with_retry(eventually_works)
    
    assert result == "connected"
    assert call_count == 2

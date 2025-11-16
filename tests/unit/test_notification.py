"""Unit tests for notification service."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import aiohttp

from ingestion_agent.notification import NotificationService, notify_pipeline_ready


@pytest.fixture
def notification_service():
    """Create notification service instance."""
    return NotificationService(notification_url="http://localhost:9000/webhook")


@pytest.mark.asyncio
async def test_notify_pipeline_ready_success(notification_service):
    """Test successful pipeline ready notification."""
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        metrics = {'num_images': 10, 'num_tiles': 100}
        result = await notification_service.notify_pipeline_ready(
            dvc_ref="data/tiles.dvc",
            metrics=metrics
        )
        
        assert result is True


@pytest.mark.asyncio
async def test_notify_pipeline_ready_failure(notification_service):
    """Test failed pipeline ready notification."""
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        metrics = {'num_images': 10}
        result = await notification_service.notify_pipeline_ready(
            dvc_ref="data/tiles.dvc",
            metrics=metrics
        )
        
        assert result is False


@pytest.mark.asyncio
async def test_notify_pipeline_ready_no_url():
    """Test notification with no URL configured."""
    service = NotificationService(notification_url=None)
    
    result = await service.notify_pipeline_ready(
        dvc_ref="data/tiles.dvc",
        metrics={}
    )
    
    assert result is False


@pytest.mark.asyncio
async def test_notify_processing_failed(notification_service):
    """Test processing failed notification."""
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        result = await notification_service.notify_processing_failed(
            error="Test error",
            event_info={'bucket': 'test', 'key': 'image.jpg'}
        )
        
        assert result is True


@pytest.mark.asyncio
async def test_notify_pipeline_ready_helper():
    """Test helper function for notification."""
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        result = await notify_pipeline_ready(
            dvc_ref="data/tiles.dvc",
            notification_url="http://localhost:9000/webhook",
            num_images=5
        )
        
        assert result is True

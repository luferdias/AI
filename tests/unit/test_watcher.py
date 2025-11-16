"""Unit tests for watcher module."""

import asyncio
from unittest.mock import AsyncMock, patch, Mock

import pytest

from ingestion_agent.watcher import S3EventWatcher, parse_event
from ingestion_agent.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        s3_bucket="test-bucket",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_region="us-east-1"
    )


@pytest.fixture
def event_handler():
    """Create mock event handler."""
    return AsyncMock()


@pytest.fixture
def watcher(config, event_handler):
    """Create S3 event watcher instance."""
    return S3EventWatcher(config, event_handler)


def test_create_event(watcher):
    """Test event creation from S3 object metadata."""
    obj = {
        'Key': 'images/test.jpg',
        'Size': 1024,
        'ETag': '"abc123"',
        'LastModified': None
    }
    
    event = watcher._create_event('images/test.jpg', obj)
    
    assert event['bucket'] == 'test-bucket'
    assert event['key'] == 'images/test.jpg'
    assert event['size'] == 1024
    assert event['etag'] == 'abc123'


@pytest.mark.asyncio
async def test_process_sqs_message(watcher, event_handler):
    """Test processing SQS message with S3 event."""
    message = {
        'Records': [
            {
                'eventSource': 'aws:s3',
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {
                        'key': 'images/test.jpg',
                        'size': 2048,
                        'eTag': '"def456"'
                    }
                }
            }
        ]
    }
    
    await watcher.process_sqs_message(message)
    
    event_handler.assert_called_once()
    call_args = event_handler.call_args[0][0]
    assert call_args['key'] == 'images/test.jpg'
    assert call_args['size'] == 2048


def test_parse_event():
    """Test parsing event to extract bucket and key."""
    event = {
        'bucket': 'my-bucket',
        'key': 'images/photo.jpg'
    }
    
    bucket, key = parse_event(event)
    
    assert bucket == 'my-bucket'
    assert key == 'images/photo.jpg'


def test_watcher_stop(watcher):
    """Test stopping the watcher."""
    watcher._running = True
    watcher.stop()
    assert watcher._running is False

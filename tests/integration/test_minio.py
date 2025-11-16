"""Integration test with MinIO."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_image():
    """Create a test image file."""
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "test_drone_image.jpg")
    
    # Create a 1024x1024 RGB image
    img = Image.new('RGB', (1024, 1024), color='green')
    img.save(image_path)
    
    yield image_path
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)


@pytest.mark.skip(reason="Requires MinIO setup")
@pytest.mark.asyncio
async def test_end_to_end_with_minio(test_image):
    """End-to-end test with MinIO.
    
    This test requires MinIO to be running locally.
    Run: docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
    """
    from ingestion_agent.config import Config
    from ingestion_agent.main import IngestionAgent
    
    # Configure to use local MinIO
    config = Config(
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        s3_endpoint_url="http://localhost:9000",
        s3_bucket="test-bucket",
        tile_size=256,
        temp_dir=tempfile.mkdtemp(),
        output_dir=tempfile.mkdtemp()
    )
    
    agent = IngestionAgent(config)
    
    # Create test event
    event = {
        'bucket': 'test-bucket',
        'key': 'images/test_drone_image.jpg'
    }
    
    # Process event
    await agent.process_event(event)
    
    # Verify tiles were created
    tiles_dir = os.path.join(config.output_dir, "test_drone_image")
    assert os.path.exists(tiles_dir)
    
    # Check metadata file
    metadata_path = os.path.join(tiles_dir, "metadata.json")
    assert os.path.exists(metadata_path)

"""Unit tests for data ingestion."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from src.data.ingestor import ImageIngestor


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img_path = temp_dir / "test_image.jpg"
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


class TestImageIngestor:
    """Test cases for ImageIngestor."""
    
    def test_initialization(self, temp_dir):
        """Test ingestor initialization."""
        data_dir = temp_dir / "data"
        ingestor = ImageIngestor(data_dir)
        assert ingestor.data_dir == data_dir
        assert data_dir.exists()
    
    def test_supported_formats(self):
        """Test supported image formats."""
        ingestor = ImageIngestor(Path("/tmp"))
        assert '.jpg' in ingestor.SUPPORTED_FORMATS
        assert '.png' in ingestor.SUPPORTED_FORMATS
        assert '.tiff' in ingestor.SUPPORTED_FORMATS
    
    def test_validate_image(self, temp_dir, sample_image):
        """Test image validation."""
        ingestor = ImageIngestor(temp_dir)
        assert ingestor._validate_image(sample_image)
    
    def test_get_image_stats(self, temp_dir, sample_image):
        """Test image statistics."""
        # Ingest image first
        ingestor = ImageIngestor(temp_dir / "data")
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        
        # Copy sample image to source
        import shutil
        shutil.copy(sample_image, source_dir / "test.jpg")
        
        ingestor.ingest_from_directory(source_dir)
        
        stats = ingestor.get_image_stats()
        assert stats['total_images'] > 0
        assert stats['total_size_mb'] > 0

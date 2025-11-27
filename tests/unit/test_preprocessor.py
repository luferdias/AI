"""Unit tests for image preprocessing."""

import pytest
import numpy as np
import torch

from src.data.preprocessor import ImagePreprocessor


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        assert preprocessor.target_size == (256, 256)
        assert preprocessor.normalize is True
    
    def test_training_transforms(self):
        """Test training augmentation pipeline."""
        preprocessor = ImagePreprocessor()
        transforms = preprocessor.get_training_transforms()
        assert transforms is not None
    
    def test_validation_transforms(self):
        """Test validation transforms."""
        preprocessor = ImagePreprocessor()
        transforms = preprocessor.get_validation_transforms()
        assert transforms is not None
    
    def test_preprocess_single(self, sample_image):
        """Test single image preprocessing."""
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        result = preprocessor.preprocess_single(sample_image, for_training=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 512, 512)
    
    def test_enhance_contrast(self, sample_image):
        """Test contrast enhancement."""
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.enhance_contrast(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
    
    def test_detect_edges(self, sample_image):
        """Test edge detection."""
        preprocessor = ImagePreprocessor()
        edges = preprocessor.detect_edges(sample_image)
        
        assert edges.shape == sample_image.shape[:2]
        assert len(edges.shape) == 2  # Should be grayscale

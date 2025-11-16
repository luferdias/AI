"""Unit tests for models."""

import pytest
import torch

from src.models.unet import UNet
from src.models.deeplab import DeepLabV3Plus


class TestUNet:
    """Test cases for U-Net model."""
    
    def test_initialization(self):
        """Test U-Net initialization."""
        model = UNet(n_channels=3, n_classes=1)
        assert model.n_channels == 3
        assert model.n_classes == 1
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(1, 3, 512, 512)
        
        output = model(x)
        assert output.shape == (1, 1, 512, 512)
    
    def test_predict(self):
        """Test prediction method."""
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(1, 3, 512, 512)
        
        pred = model.predict(x)
        assert pred.shape == (1, 1, 512, 512)
        assert pred.min() >= 0 and pred.max() <= 1  # Sigmoid output


class TestDeepLabV3Plus:
    """Test cases for DeepLabV3+ model."""
    
    def test_initialization(self):
        """Test DeepLabV3+ initialization."""
        model = DeepLabV3Plus(n_channels=3, n_classes=1, pretrained=False)
        assert model.n_channels == 3
        assert model.n_classes == 1
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DeepLabV3Plus(n_channels=3, n_classes=1, pretrained=False)
        x = torch.randn(1, 3, 512, 512)
        
        output = model(x)
        assert output.shape == (1, 1, 512, 512)
    
    def test_predict(self):
        """Test prediction method."""
        model = DeepLabV3Plus(n_channels=3, n_classes=1, pretrained=False)
        x = torch.randn(1, 3, 512, 512)
        
        pred = model.predict(x)
        assert pred.shape == (1, 1, 512, 512)
        assert pred.min() >= 0 and pred.max() <= 1  # Sigmoid output

"""Unit tests for image processor."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from ingestion_agent.config import Config
from ingestion_agent.processor import ImageProcessor


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        tile_size=128,
        tile_overlap=0,
        normalize=True,
        apply_augmentation=False,
        temp_dir=tempfile.mkdtemp(),
        output_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def processor(config):
    """Create image processor instance."""
    return ImageProcessor(config)


@pytest.fixture
def test_image_path():
    """Create a test image."""
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "test_image.jpg")

    # Create a 512x512 RGB image
    img = Image.new("RGB", (512, 512), color="red")
    img.save(image_path)

    yield image_path

    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)


@pytest.mark.asyncio
async def test_validate_image_valid(processor, test_image_path):
    """Test image validation with valid image."""
    is_valid = await processor.validate_image(test_image_path)
    assert is_valid is True


@pytest.mark.asyncio
async def test_validate_image_invalid():
    """Test image validation with invalid file."""
    processor = ImageProcessor(Config())
    is_valid = await processor.validate_image("/nonexistent/file.jpg")
    assert is_valid is False


@pytest.mark.asyncio
async def test_validate_image_too_small(processor):
    """Test image validation with too small image."""
    temp_dir = tempfile.mkdtemp()
    small_image_path = os.path.join(temp_dir, "small.jpg")

    # Create a 32x32 image (too small)
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(small_image_path)

    is_valid = await processor.validate_image(small_image_path)
    assert is_valid is False

    os.remove(small_image_path)


def test_calculate_checksum(processor, test_image_path):
    """Test checksum calculation."""
    checksum = processor.calculate_checksum(test_image_path)
    assert isinstance(checksum, str)
    assert len(checksum) == 64  # SHA256 produces 64 hex characters

    # Verify consistency
    checksum2 = processor.calculate_checksum(test_image_path)
    assert checksum == checksum2


@pytest.mark.asyncio
async def test_tile_image(processor, test_image_path):
    """Test image tiling."""
    output_dir = tempfile.mkdtemp()

    tiles_metadata = await processor.tile_image(test_image_path, output_dir)

    # Check that tiles were generated
    assert len(tiles_metadata) > 0

    # For a 512x512 image with 128x128 tiles, we should get 16 tiles (4x4)
    assert len(tiles_metadata) == 16

    # Check metadata structure
    tile = tiles_metadata[0]
    assert "tile_id" in tile
    assert "filename" in tile
    assert "source_image" in tile
    assert "bbox" in tile
    assert "size" in tile
    assert "checksum" in tile

    # Verify tile files exist
    for tile in tiles_metadata:
        tile_path = os.path.join(output_dir, tile["filename"])
        assert os.path.exists(tile_path)


@pytest.mark.asyncio
async def test_tile_image_with_overlap(config, test_image_path):
    """Test image tiling with overlap."""
    config.tile_overlap = 32
    processor = ImageProcessor(config)
    output_dir = tempfile.mkdtemp()

    tiles_metadata = await processor.tile_image(test_image_path, output_dir)

    # With overlap, we should get more tiles
    assert len(tiles_metadata) > 16


@pytest.mark.asyncio
async def test_save_tile_metadata(processor):
    """Test saving tile metadata to JSON."""
    output_dir = tempfile.mkdtemp()
    metadata_path = os.path.join(output_dir, "metadata.json")

    test_metadata = [
        {"tile_id": 0, "filename": "tile_0.png"},
        {"tile_id": 1, "filename": "tile_1.png"},
    ]

    await processor.save_tile_metadata(test_metadata, metadata_path)

    assert os.path.exists(metadata_path)

    # Verify content
    import json

    with open(metadata_path, "r") as f:
        loaded_metadata = json.load(f)

    assert loaded_metadata == test_metadata


def test_normalize_tile(processor):
    """Test tile normalization."""
    # Create test tile
    tile = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    normalized = processor._normalize_tile(tile)

    # Check output type and shape
    assert normalized.dtype == np.uint8
    assert normalized.shape == tile.shape


def test_augment_tile(processor):
    """Test tile augmentation."""
    # Create test tile
    tile = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    augmented = processor._augment_tile(tile)

    # Check output type and shape
    assert augmented.dtype == np.uint8
    assert augmented.shape == tile.shape

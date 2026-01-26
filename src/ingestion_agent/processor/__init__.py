"""Image processor module for downloading, validation, tiling and augmentation."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import aioboto3
import aiofiles
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from prometheus_client import Counter, Histogram

from ..config import Config

logger = logging.getLogger("ingestion_agent.processor")

# Metrics
images_downloaded = Counter("images_downloaded_total", "Total images downloaded")
images_validated = Counter("images_validated_total", "Total images validated successfully")
tiles_generated = Counter("tiles_generated_total", "Total tiles generated")
download_time = Histogram("download_time_seconds", "Time to download images")
processing_time = Histogram("processing_time_seconds", "Time to process images")


class ImageProcessor:
    """Handles image download, validation, tiling and augmentation."""

    def __init__(self, config: Config):
        """Initialize the processor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.session = aioboto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region,
        )
        self.augmentation_pipeline = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline.

        Returns:
            Albumentations composition
        """
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]
        )

    async def download_image(self, bucket: str, key: str, dest: str) -> str:
        """Download image from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key
            dest: Destination file path

        Returns:
            Path to downloaded file
        """
        with download_time.time():
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            async with self.session.client("s3", endpoint_url=self.config.s3_endpoint_url) as s3:
                await s3.download_file(bucket, key, dest)

            images_downloaded.inc()
            logger.info(f"Downloaded {key} to {dest}")
            return dest

    async def validate_image(self, path: str) -> bool:
        """Validate image file integrity and format.

        Args:
            path: Path to image file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file exists and has content
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                logger.error(f"Image file invalid or empty: {path}")
                return False

            # Try to open and verify the image
            with Image.open(path) as img:
                img.verify()

            # Reopen for actual validation (verify() closes the file)
            with Image.open(path) as img:
                img.load()

                # Check basic properties
                if img.size[0] < 64 or img.size[1] < 64:
                    logger.error(f"Image too small: {img.size}")
                    return False

            images_validated.inc()
            logger.info(f"Validated image: {path}")
            return True

        except Exception as e:
            logger.error(f"Image validation failed for {path}: {e}")
            return False

    def calculate_checksum(self, path: str) -> str:
        """Calculate SHA256 checksum of file.

        Args:
            path: Path to file

        Returns:
            Hex digest of checksum
        """
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def tile_image(self, image_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """Tile image into smaller patches.

        Args:
            image_path: Path to source image
            output_dir: Directory to save tiles

        Returns:
            List of tile metadata dictionaries
        """
        with processing_time.time():
            os.makedirs(output_dir, exist_ok=True)

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            height, width = image.shape[:2]
            tile_size = self.config.tile_size
            overlap = self.config.tile_overlap
            stride = tile_size - overlap

            tiles_metadata = []
            tile_idx = 0

            # Generate tiles
            for y in range(0, height - tile_size + 1, stride):
                for x in range(0, width - tile_size + 1, stride):
                    # Extract tile
                    tile = image[y : y + tile_size, x : x + tile_size]

                    # Apply normalization if configured
                    if self.config.normalize:
                        tile = self._normalize_tile(tile)

                    # Apply augmentation if configured
                    if self.config.apply_augmentation:
                        tile = self._augment_tile(tile)

                    # Save tile
                    tile_filename = f"{Path(image_path).stem}_tile_{tile_idx:04d}.png"
                    tile_path = os.path.join(output_dir, tile_filename)
                    cv2.imwrite(tile_path, tile)

                    # Create metadata
                    metadata = {
                        "tile_id": tile_idx,
                        "filename": tile_filename,
                        "source_image": os.path.basename(image_path),
                        "bbox": {"x": x, "y": y, "width": tile_size, "height": tile_size},
                        "size": {"width": tile.shape[1], "height": tile.shape[0]},
                        "checksum": self.calculate_checksum(tile_path),
                    }

                    tiles_metadata.append(metadata)
                    tile_idx += 1
                    tiles_generated.inc()

            logger.info(f"Generated {len(tiles_metadata)} tiles from {image_path}")
            return tiles_metadata

    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """Normalize tile to [0, 1] range.

        Args:
            tile: Input tile

        Returns:
            Normalized tile
        """
        tile = tile.astype(np.float32) / 255.0
        tile = (tile * 255).astype(np.uint8)
        return tile

    def _augment_tile(self, tile: np.ndarray) -> np.ndarray:
        """Apply augmentation to tile.

        Args:
            tile: Input tile

        Returns:
            Augmented tile
        """
        augmented = self.augmentation_pipeline(image=tile)
        return augmented["image"]

    async def save_tile_metadata(self, metadata_list: List[Dict[str, Any]], output_path: str):
        """Save tile metadata to JSON file.

        Args:
            metadata_list: List of tile metadata
            output_path: Path to output JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        async with aiofiles.open(output_path, "w") as f:
            await f.write(json.dumps(metadata_list, indent=2))

        logger.info(f"Saved metadata for {len(metadata_list)} tiles to {output_path}")


async def download_image(bucket: str, key: str, dest: str, config: Config) -> str:
    """Helper function to download image.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        dest: Destination path
        config: Application configuration

    Returns:
        Path to downloaded file
    """
    processor = ImageProcessor(config)
    return await processor.download_image(bucket, key, dest)

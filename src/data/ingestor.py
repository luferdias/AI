"""
Data ingestion module for drone and camera images.
Supports multiple image formats and sources.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageIngestor:
    """Handles ingestion of images from various sources."""
    
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize image ingestor.
        
        Args:
            data_dir: Directory where raw images will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_from_directory(
        self, 
        source_dir: Union[str, Path],
        validate: bool = True
    ) -> List[Path]:
        """
        Ingest images from a source directory.
        
        Args:
            source_dir: Directory containing source images
            validate: Whether to validate images during ingestion
            
        Returns:
            List of ingested image paths
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        ingested_files = []
        
        for img_path in source_dir.rglob('*'):
            if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    if validate:
                        self._validate_image(img_path)
                    
                    # Copy to data directory preserving relative structure
                    rel_path = img_path.relative_to(source_dir)
                    dest_path = self.data_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Load and save to ensure format compatibility
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        cv2.imwrite(str(dest_path), img)
                        ingested_files.append(dest_path)
                        logger.info(f"Ingested: {dest_path}")
                    else:
                        logger.warning(f"Could not read image: {img_path}")
                        
                except Exception as e:
                    logger.error(f"Error ingesting {img_path}: {e}")
                    
        logger.info(f"Total images ingested: {len(ingested_files)}")
        return ingested_files
    
    def _validate_image(self, img_path: Path) -> bool:
        """
        Validate that an image can be properly loaded.
        
        Args:
            img_path: Path to image file
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        try:
            img = Image.open(img_path)
            img.verify()
            
            # Also check with OpenCV
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise ValueError(f"OpenCV cannot read image: {img_path}")
                
            if cv_img.shape[0] < 32 or cv_img.shape[1] < 32:
                raise ValueError(f"Image too small: {img_path}")
                
            return True
            
        except Exception as e:
            raise ValueError(f"Invalid image {img_path}: {e}")
    
    def get_image_stats(self) -> dict:
        """
        Get statistics about ingested images.
        
        Returns:
            Dictionary with image statistics
        """
        stats = {
            'total_images': 0,
            'formats': {},
            'total_size_mb': 0,
            'avg_dimensions': [0, 0]
        }
        
        heights, widths = [], []
        
        for img_path in self.data_dir.rglob('*'):
            if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                stats['total_images'] += 1
                stats['formats'][img_path.suffix] = stats['formats'].get(img_path.suffix, 0) + 1
                stats['total_size_mb'] += img_path.stat().st_size / (1024 * 1024)
                
                # Get dimensions
                img = cv2.imread(str(img_path))
                if img is not None:
                    heights.append(img.shape[0])
                    widths.append(img.shape[1])
        
        if heights:
            stats['avg_dimensions'] = [int(np.mean(heights)), int(np.mean(widths))]
        
        return stats

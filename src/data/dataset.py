"""
Dataset classes for crack detection.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple
import logging

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CrackSegmentationDataset(Dataset):
    """Dataset for crack segmentation tasks."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing segmentation masks
            transform: Optional transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get list of images
        self.image_files = sorted([
            f for f in self.images_dir.glob('*.jpg')
        ] + [
            f for f in self.images_dir.glob('*.png')
        ])
        
        logger.info(f"Found {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask pair.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, mask)
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask if not found
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            logger.warning(f"Mask not found for {img_path.name}, using empty mask")
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask is the right shape
        if isinstance(mask, torch.Tensor):
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Normalize mask to 0-1
        mask = (mask > 0).float()
        
        return image, mask


class CrackDetectionDataset(Dataset):
    """Dataset for crack detection (bounding boxes)."""
    
    def __init__(
        self,
        images_dir: Path,
        annotations_dir: Path,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing YOLO format annotations
            transform: Optional transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        
        self.image_files = sorted([
            f for f in self.images_dir.glob('*.jpg')
        ] + [
            f for f in self.images_dir.glob('*.png')
        ])
        
        logger.info(f"Found {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        """Get image and annotations."""
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_path = self.annotations_dir / f"{img_path.stem}.txt"
        bboxes = []
        labels = []
        
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        label = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        labels.append(label)
                        bboxes.append(bbox)
        
        if self.transform:
            # Apply transforms if provided
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'image_path': str(img_path)
        }

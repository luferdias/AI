"""
Image preprocessing utilities for concrete crack detection.
Includes normalization, augmentation, and resizing.
"""

from typing import Tuple, Optional, List
import logging

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for model training and inference."""
    
    def __init__(
        self, 
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images
            mean: Mean values for normalization (ImageNet by default)
            std: Std values for normalization (ImageNet by default)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
    def get_training_transforms(self) -> A.Compose:
        """
        Get augmentation pipeline for training.
        Optimized for concrete crack detection.
        
        Returns:
            Albumentations compose object
        """
        transforms = [
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ),
            A.CLAHE(p=0.3),
        ]
        
        if self.normalize:
            transforms.append(
                A.Normalize(mean=self.mean, std=self.std)
            )
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def get_validation_transforms(self) -> A.Compose:
        """
        Get transforms for validation/inference.
        
        Returns:
            Albumentations compose object
        """
        transforms = [
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
        ]
        
        if self.normalize:
            transforms.append(
                A.Normalize(mean=self.mean, std=self.std)
            )
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def preprocess_single(
        self, 
        image: np.ndarray,
        for_training: bool = False
    ) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image (BGR or RGB format)
            for_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image tensor
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Apply transforms
        if for_training:
            transform = self.get_training_transforms()
        else:
            transform = self.get_validation_transforms()
        
        transformed = transform(image=image)
        return transformed['image']
    
    def enhance_contrast(
        self, 
        image: np.ndarray,
        clip_limit: float = 2.0
    ) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        Useful for improving crack visibility.
        
        Args:
            image: Input image
            clip_limit: CLAHE clip limit
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def detect_edges(
        self, 
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges in image using Canny edge detection.
        Can be used as preprocessing step or for visualization.
        
        Args:
            image: Input image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
            
        Returns:
            Edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges

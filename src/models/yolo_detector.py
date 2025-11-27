"""
YOLOv8 wrapper for crack detection.
Integrates Ultralytics YOLOv8 for object detection.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

import torch
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOv8CrackDetector:
    """
    YOLOv8-based crack detector.
    Wraps Ultralytics YOLO for crack detection tasks.
    """
    
    def __init__(
        self,
        model_size: str = 'n',  # n, s, m, l, x
        pretrained: bool = True,
        num_classes: int = 1
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            pretrained: Whether to use pretrained weights
            num_classes: Number of detection classes
        """
        self.model_size = model_size
        self.num_classes = num_classes
        
        if pretrained:
            model_name = f'yolov8{model_size}.pt'
        else:
            model_name = f'yolov8{model_size}.yaml'
        
        try:
            self.model = YOLO(model_name)
            logger.info(f"Loaded YOLOv8{model_size} model")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def train(
        self,
        data_config: Union[str, Path],
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = '0',
        **kwargs
    ):
        """
        Train YOLOv8 model.
        
        Args:
            data_config: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            device: Device to use (cuda device number or 'cpu')
            **kwargs: Additional training arguments
        """
        results = self.model.train(
            data=str(data_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            **kwargs
        )
        return results
    
    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = '0',
        **kwargs
    ) -> List[Dict]:
        """
        Run inference on images.
        
        Args:
            source: Image source (path, array, etc.)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size
            device: Device to use
            **kwargs: Additional inference arguments
            
        Returns:
            List of detection results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            **kwargs
        )
        return results
    
    def val(
        self,
        data_config: Union[str, Path],
        imgsz: int = 640,
        batch: int = 16,
        device: str = '0',
        **kwargs
    ):
        """
        Validate model on dataset.
        
        Args:
            data_config: Path to data configuration YAML
            imgsz: Image size
            batch: Batch size
            device: Device to use
            **kwargs: Additional validation arguments
        """
        results = self.model.val(
            data=str(data_config),
            imgsz=imgsz,
            batch=batch,
            device=device,
            **kwargs
        )
        return results
    
    def export(
        self,
        format: str = 'onnx',
        imgsz: int = 640,
        **kwargs
    ):
        """
        Export model to different formats for deployment.
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            imgsz: Image size
            **kwargs: Additional export arguments
        """
        return self.model.export(
            format=format,
            imgsz=imgsz,
            **kwargs
        )
    
    def save(self, path: Union[str, Path]):
        """
        Save model weights.
        
        Args:
            path: Path to save model
        """
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """
        Load model weights.
        
        Args:
            path: Path to model weights
        """
        self.model = YOLO(str(path))
        logger.info(f"Model loaded from {path}")
    
    @property
    def device(self):
        """Get model device."""
        return self.model.device
    
    def to(self, device: str):
        """
        Move model to device.
        
        Args:
            device: Target device
        """
        self.model.to(device)
        return self

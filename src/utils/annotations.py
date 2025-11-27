"""
Utility functions for data annotation and management.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AnnotationManager:
    """Manages image annotations for crack detection."""
    
    def __init__(self, annotations_dir: Path):
        """
        Initialize annotation manager.
        
        Args:
            annotations_dir: Directory to store annotations
        """
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def save_coco_annotation(
        self,
        image_id: int,
        image_path: Path,
        masks: List[np.ndarray],
        categories: List[int],
        output_file: Optional[Path] = None
    ):
        """
        Save annotations in COCO format.
        
        Args:
            image_id: Unique image identifier
            image_path: Path to image
            masks: List of binary masks
            categories: List of category IDs
            output_file: Output JSON file path
        """
        if output_file is None:
            output_file = self.annotations_dir / f"{image_path.stem}.json"
        
        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        
        # Create COCO format annotation
        annotation = {
            'image': {
                'id': image_id,
                'file_name': image_path.name,
                'width': width,
                'height': height
            },
            'annotations': []
        }
        
        for idx, (mask, category_id) in enumerate(zip(masks, categories)):
            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                if area > 0:
                    # Convert contour to segmentation format
                    segmentation = contour.flatten().tolist()
                    
                    annotation['annotations'].append({
                        'id': len(annotation['annotations']),
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [x, y, w, h],
                        'area': float(area),
                        'segmentation': [segmentation],
                        'iscrowd': 0
                    })
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        logger.info(f"Saved annotation to {output_file}")
    
    def save_yolo_annotation(
        self,
        image_path: Path,
        bboxes: List[List[float]],
        classes: List[int],
        output_file: Optional[Path] = None
    ):
        """
        Save annotations in YOLO format.
        
        Args:
            image_path: Path to image
            bboxes: List of bounding boxes [x_center, y_center, width, height] (normalized)
            classes: List of class IDs
            output_file: Output text file path
        """
        if output_file is None:
            output_file = self.annotations_dir / f"{image_path.stem}.txt"
        
        with open(output_file, 'w') as f:
            for bbox, class_id in zip(bboxes, classes):
                # YOLO format: class x_center y_center width height
                line = f"{class_id} {' '.join(map(str, bbox))}\n"
                f.write(line)
        
        logger.info(f"Saved YOLO annotation to {output_file}")
    
    def load_coco_annotation(self, annotation_file: Path) -> Dict:
        """
        Load COCO format annotation.
        
        Args:
            annotation_file: Path to annotation JSON file
            
        Returns:
            Dictionary with annotation data
        """
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
        return annotation
    
    def convert_mask_to_bbox(
        self,
        mask: np.ndarray,
        normalize: bool = False,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None
    ) -> List[float]:
        """
        Convert binary mask to bounding box.
        
        Args:
            mask: Binary mask
            normalize: Whether to normalize coordinates
            img_width: Image width (required if normalize=True)
            img_height: Image height (required if normalize=True)
            
        Returns:
            Bounding box [x_center, y_center, width, height]
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return [0, 0, 0, 0]
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to center format
        x_center = x + w / 2
        y_center = y + h / 2
        
        if normalize:
            if img_width is None or img_height is None:
                raise ValueError("Image dimensions required for normalization")
            x_center /= img_width
            y_center /= img_height
            w /= img_width
            h /= img_height
        
        return [x_center, y_center, w, h]
    
    def visualize_annotations(
        self,
        image_path: Path,
        annotation_file: Path,
        output_path: Optional[Path] = None
    ):
        """
        Visualize annotations on image.
        
        Args:
            image_path: Path to image
            annotation_file: Path to annotation file
            output_path: Path to save visualization
        """
        img = cv2.imread(str(image_path))
        annotation = self.load_coco_annotation(annotation_file)
        
        # Draw annotations
        for ann in annotation.get('annotations', []):
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class {ann['category_id']}"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        if output_path:
            cv2.imwrite(str(output_path), img)
            logger.info(f"Saved visualization to {output_path}")
        
        return img

#!/usr/bin/env python3
"""
Inference script for crack detection.
Runs inference on images using trained models.
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import cv2
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplab import DeepLabV3Plus
from src.models.yolo_detector import YOLOv8CrackDetector
from src.data.preprocessor import ImagePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_type: str, model_path: Path, device: str):
    """Load trained model."""
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1).to(device)
    elif model_type == 'deeplab':
        model = DeepLabV3Plus(n_channels=3, n_classes=1).to(device)
    elif model_type == 'yolo':
        model = YOLOv8CrackDetector(pretrained=False)
        model.load(model_path)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded weights from {model_path}")
    
    model.eval()
    return model


def run_inference(model, image_path: Path, model_type: str, device: str):
    """Run inference on single image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    logger.info(f"Processing {image_path.name}")
    
    start_time = time.time()
    
    if model_type == 'yolo':
        # YOLO inference
        results = model.predict(image, conf=0.25)
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(f"Inference time: {inference_time:.2f}ms")
        
        # Display results
        if results:
            result = results[0]
            logger.info(f"Detected {len(result.boxes)} cracks")
    else:
        # Segmentation inference
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        preprocessed = preprocessor.preprocess_single(image, for_training=False)
        preprocessed = preprocessed.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(preprocessed)
            pred_mask = torch.sigmoid(output) > 0.5
        
        inference_time = (time.time() - start_time) * 1000
        
        logger.info(f"Inference time: {inference_time:.2f}ms")
        
        # Calculate coverage
        coverage = pred_mask.float().mean().item()
        logger.info(f"Crack coverage: {coverage*100:.2f}%")
    
    return inference_time


def main():
    parser = argparse.ArgumentParser(description='Run crack detection inference')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['unet', 'deeplab', 'yolo'],
        required=True,
        help='Model type'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Path to directory of images'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        raise ValueError("Must provide either --image or --image-dir")
    
    # Load model
    model_path = Path(args.model_path)
    model = load_model(args.model_type, model_path, args.device)
    
    # Process images
    inference_times = []
    
    if args.image:
        time_ms = run_inference(model, Path(args.image), args.model_type, args.device)
        inference_times.append(time_ms)
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
        for img_path in image_dir.glob('*.jpg'):
            time_ms = run_inference(model, img_path, args.model_type, args.device)
            inference_times.append(time_ms)
    
    # Report statistics
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        logger.info(f"\nAverage inference time: {avg_time:.2f}ms")
        logger.info(f"Target latency: 200-500ms ✓" if avg_time < 500 else f"Target latency: 200-500ms ✗")


if __name__ == '__main__':
    main()

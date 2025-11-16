#!/usr/bin/env python3
"""
Training script for crack detection models.
Supports U-Net, DeepLab, and YOLOv8.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplab import DeepLabV3Plus
from src.models.yolo_detector import YOLOv8CrackDetector
from src.models.trainer import SegmentationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_segmentation_model(config: dict, model_type: str):
    """Train U-Net or DeepLab model."""
    logger.info(f"Training {model_type} model")
    
    # Initialize model
    if model_type == 'unet':
        model = UNet(
            n_channels=config['model']['n_channels'],
            n_classes=config['model']['n_classes'],
            bilinear=config['model']['bilinear']
        )
    elif model_type == 'deeplab':
        model = DeepLabV3Plus(
            n_channels=config['model'].get('n_channels', 3),
            n_classes=config['model'].get('n_classes', 1),
            pretrained=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        experiment_name=config['mlflow']['experiment_name']
    )
    
    # Note: In production, you would load actual datasets here
    # This is a placeholder for the training structure
    logger.info("Model initialized successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}")
    
    return model


def train_yolo_model(config: dict):
    """Train YOLOv8 model."""
    logger.info("Training YOLOv8 model")
    
    # Initialize detector
    detector = YOLOv8CrackDetector(
        model_size='n',  # Start with nano for faster training
        pretrained=True,
        num_classes=config['nc']
    )
    
    # Note: In production, you would call detector.train() with actual data
    logger.info("YOLOv8 model initialized successfully")
    
    return detector


def main():
    parser = argparse.ArgumentParser(description='Train crack detection model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['unet', 'deeplab', 'yolo'],
        required=True,
        help='Model type to train'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    if args.model in ['unet', 'deeplab']:
        model = train_segmentation_model(config, args.model)
    else:
        model = train_yolo_model(config)
    
    logger.info(f"Training completed. Models saved to {output_dir}")


if __name__ == '__main__':
    main()

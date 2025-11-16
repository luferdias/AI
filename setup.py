#!/usr/bin/env python3
"""
Setup script for the crack detection pipeline.
Initializes directories and validates environment.
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 10):
        logger.error("Python 3.10 or higher is required")
        return False
    logger.info(f"✓ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if key dependencies are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'ultralytics',
        'fastapi',
        'mlflow',
        'dvc',
        'opencv-python',
        'albumentations'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} not found")
    
    return missing


def create_directories():
    """Create necessary directories."""
    directories = [
        'data/raw',
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/val/images',
        'data/processed/val/masks',
        'data/processed/test/images',
        'data/processed/test/masks',
        'data/annotations',
        'models',
        'logs',
        'mlruns',
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.warning("✗ No GPU available - will use CPU (slower)")
            return False
    except ImportError:
        logger.warning("Cannot check GPU - PyTorch not installed")
        return False


def create_env_template():
    """Create .env template file."""
    env_template = """# Environment Configuration for Crack Detection Pipeline

# Model Configuration
MODEL_TYPE=unet
MODEL_PATH=models/best_model.pth
DEVICE=cuda

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# DVC Configuration
DVC_REMOTE_URL=s3://your-bucket/dvc-storage
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Training Configuration
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_EPOCHS=100
"""
    
    env_file = Path('.env.template')
    if not env_file.exists():
        env_file.write_text(env_template)
        logger.info("✓ Created .env.template")
    else:
        logger.info("✓ .env.template already exists")


def main():
    """Main setup function."""
    logger.info("=" * 60)
    logger.info("Concrete Crack Detection Pipeline - Setup")
    logger.info("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        logger.warning(f"\nMissing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
    
    # Create directories
    logger.info("\nCreating directory structure...")
    create_directories()
    
    # Check GPU
    logger.info("\nChecking GPU availability...")
    check_gpu()
    
    # Create .env template
    logger.info("\nCreating configuration templates...")
    create_env_template()
    
    logger.info("\n" + "=" * 60)
    logger.info("Setup complete! Next steps:")
    logger.info("=" * 60)
    logger.info("1. Install missing dependencies: pip install -r requirements.txt")
    logger.info("2. Copy .env.template to .env and configure")
    logger.info("3. Add your training data to data/raw/")
    logger.info("4. Review configs/unet_config.yaml")
    logger.info("5. Start training: python scripts/train.py")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

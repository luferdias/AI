"""Configuration management for the ingestion agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    
    # Cloud storage
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "drone-images")
    s3_endpoint_url: Optional[str] = os.getenv("S3_ENDPOINT_URL")
    
    # Processing
    tile_size: int = int(os.getenv("TILE_SIZE", "256"))
    tile_overlap: int = int(os.getenv("TILE_OVERLAP", "0"))
    normalize: bool = os.getenv("NORMALIZE", "true").lower() == "true"
    apply_augmentation: bool = os.getenv("APPLY_AUGMENTATION", "false").lower() == "true"
    
    # DVC
    dvc_remote: str = os.getenv("DVC_REMOTE", "storage")
    dvc_data_path: str = os.getenv("DVC_DATA_PATH", "data/tiles")
    
    # Processing limits
    max_parallel_downloads: int = int(os.getenv("MAX_PARALLEL_DOWNLOADS", "5"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff_factor: float = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0"))
    
    # Notification
    notification_url: Optional[str] = os.getenv("NOTIFICATION_URL")
    
    # Paths
    temp_dir: str = os.getenv("TEMP_DIR", "/tmp/ingestion")
    output_dir: str = os.getenv("OUTPUT_DIR", "data/tiles")
    
    # Observability
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))


def get_config() -> Config:
    """Get application configuration."""
    return Config()

"""Main application orchestrating the ingestion pipeline."""

import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, Any

from prometheus_client import start_http_server

from .config import Config, get_config
from .logging_config import setup_logging
from .watcher import S3EventWatcher, parse_event
from .processor import ImageProcessor
from .dvc_integration import DVCManager
from .notification import NotificationService
from .retry import RetryManager

logger = None


class IngestionAgent:
    """Main orchestrator for the ingestion pipeline."""
    
    def __init__(self, config: Config):
        """Initialize the ingestion agent.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.processor = ImageProcessor(config)
        self.dvc_manager = DVCManager(config.dvc_remote)
        self.notification_service = NotificationService(config.notification_url)
        self.retry_manager = RetryManager(
            max_retries=config.max_retries,
            backoff_factor=config.retry_backoff_factor
        )
        
        # Create necessary directories
        os.makedirs(config.temp_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
    async def process_event(self, event: Dict[str, Any]):
        """Process a single S3 event.
        
        Args:
            event: Event dictionary containing bucket and key information
        """
        bucket, key = parse_event(event)
        
        try:
            logger.info(f"Processing event for {bucket}/{key}")
            
            # Download image with retry
            filename = Path(key).name
            local_path = os.path.join(self.config.temp_dir, filename)
            
            await self.retry_manager.execute_with_retry(
                self.processor.download_image,
                bucket,
                key,
                local_path
            )
            
            # Validate image
            is_valid = await self.processor.validate_image(local_path)
            if not is_valid:
                logger.error(f"Image validation failed for {key}")
                await self.notification_service.notify_processing_failed(
                    "Image validation failed",
                    {'bucket': bucket, 'key': key}
                )
                return
            
            # Generate tiles
            tile_dir = os.path.join(self.config.output_dir, Path(filename).stem)
            tiles_metadata = await self.processor.tile_image(local_path, tile_dir)
            
            # Save metadata
            metadata_path = os.path.join(tile_dir, 'metadata.json')
            await self.processor.save_tile_metadata(tiles_metadata, metadata_path)
            
            # Version with DVC
            dvc_success = await self.retry_manager.execute_with_retry(
                self.dvc_manager.add,
                self.config.output_dir
            )
            
            if dvc_success:
                push_success = await self.retry_manager.execute_with_retry(
                    self.dvc_manager.push
                )
                
                if push_success:
                    # Calculate metrics
                    metrics = {
                        'num_images': 1,
                        'num_tiles': len(tiles_metadata),
                        'source_key': key,
                        'output_path': self.config.output_dir
                    }
                    
                    # Notify pipeline ready
                    await self.notification_service.notify_pipeline_ready(
                        dvc_ref=f"{self.config.output_dir}.dvc",
                        metrics=metrics,
                        metadata={'tiles': tiles_metadata[:5]}  # Send first 5 as sample
                    )
                    
                    logger.info(f"Successfully processed {key}: {len(tiles_metadata)} tiles created")
                else:
                    logger.error("DVC push failed")
            else:
                logger.error("DVC add failed")
            
            # Cleanup temp file
            if os.path.exists(local_path):
                os.remove(local_path)
                
        except Exception as e:
            logger.error(f"Error processing event {bucket}/{key}: {e}", exc_info=True)
            await self.notification_service.notify_processing_failed(
                str(e),
                {'bucket': bucket, 'key': key}
            )
            raise
    
    async def run(self):
        """Run the ingestion agent."""
        logger.info("Starting Ingestion Agent")
        
        # Start Prometheus metrics server
        start_http_server(self.config.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        
        # Create watcher
        watcher = S3EventWatcher(self.config, self.process_event)
        
        # Start polling
        await watcher.start_polling()


async def main():
    """Main entry point."""
    global logger
    
    # Load configuration
    config = get_config()
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    # Create and run agent
    agent = IngestionAgent(config)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

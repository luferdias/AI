"""Watcher module for monitoring cloud storage events."""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional
import aioboto3
from prometheus_client import Counter

from ..config import Config

logger = logging.getLogger("ingestion_agent.watcher")

# Metrics
events_received = Counter("events_received_total", "Total events received")
events_processed = Counter("events_processed_total", "Total events processed successfully")
events_failed = Counter("events_failed_total", "Total events failed to process")


class S3EventWatcher:
    """Watches for S3 events and processes new objects."""
    
    def __init__(self, config: Config, event_handler: Callable):
        """Initialize the watcher.
        
        Args:
            config: Application configuration
            event_handler: Async function to handle events
        """
        self.config = config
        self.event_handler = event_handler
        self.session = aioboto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region
        )
        self._running = False
        
    async def start_polling(self, interval: int = 10):
        """Start polling S3 bucket for new objects.
        
        Args:
            interval: Polling interval in seconds
        """
        self._running = True
        logger.info(f"Starting S3 polling on bucket {self.config.s3_bucket}")
        
        processed_keys = set()
        
        while self._running:
            try:
                async with self.session.client(
                    's3',
                    endpoint_url=self.config.s3_endpoint_url
                ) as s3:
                    response = await s3.list_objects_v2(
                        Bucket=self.config.s3_bucket,
                        Prefix='images/'
                    )
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            key = obj['Key']
                            if key not in processed_keys and key.lower().endswith(('.jpg', '.jpeg', '.png')):
                                event = self._create_event(key, obj)
                                events_received.inc()
                                
                                try:
                                    await self.event_handler(event)
                                    processed_keys.add(key)
                                    events_processed.inc()
                                    logger.info(f"Processed event for {key}")
                                except Exception as e:
                                    events_failed.inc()
                                    logger.error(f"Failed to process {key}: {e}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                await asyncio.sleep(interval)
    
    def _create_event(self, key: str, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Create an event object from S3 object metadata.
        
        Args:
            key: S3 object key
            obj: S3 object metadata
            
        Returns:
            Event dictionary
        """
        return {
            'bucket': self.config.s3_bucket,
            'key': key,
            'size': obj.get('Size', 0),
            'etag': obj.get('ETag', '').strip('"'),
            'last_modified': obj.get('LastModified').isoformat() if obj.get('LastModified') else None
        }
    
    async def process_sqs_message(self, message: Dict[str, Any]):
        """Process an SQS message containing S3 event notification.
        
        Args:
            message: SQS message body
        """
        events_received.inc()
        
        try:
            # Parse S3 event notification
            if 'Records' in message:
                for record in message['Records']:
                    if record.get('eventSource') == 'aws:s3':
                        s3_info = record['s3']
                        event = {
                            'bucket': s3_info['bucket']['name'],
                            'key': s3_info['object']['key'],
                            'size': s3_info['object'].get('size', 0),
                            'etag': s3_info['object'].get('eTag', '').strip('"')
                        }
                        
                        await self.event_handler(event)
                        events_processed.inc()
                        logger.info(f"Processed SQS event for {event['key']}")
        except Exception as e:
            events_failed.inc()
            logger.error(f"Failed to process SQS message: {e}")
            raise
    
    def stop(self):
        """Stop the watcher."""
        self._running = False
        logger.info("Stopping S3 watcher")


def parse_event(event: Dict[str, Any]) -> tuple[str, str]:
    """Parse event to extract bucket and key.
    
    Args:
        event: Event dictionary
        
    Returns:
        Tuple of (bucket, key)
    """
    return event['bucket'], event['key']

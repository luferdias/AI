"""Notification module for pipeline event publishing."""

import logging
from typing import Any, Dict, Optional

import aiohttp
from prometheus_client import Counter

logger = logging.getLogger("ingestion_agent.notification")

# Metrics
notifications_sent = Counter("notifications_sent_total", "Total notifications sent")
notifications_failed = Counter("notifications_failed_total", "Total notifications failed")


class NotificationService:
    """Handles publishing pipeline completion events."""

    def __init__(self, notification_url: Optional[str] = None):
        """Initialize notification service.

        Args:
            notification_url: URL to POST notifications to
        """
        self.notification_url = notification_url

    async def notify_pipeline_ready(
        self, dvc_ref: str, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Notify that pipeline processing is complete.

        Args:
            dvc_ref: DVC reference to versioned data
            metrics: Processing metrics (image count, size, etc.)
            metadata: Optional additional metadata

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.notification_url:
            logger.warning("No notification URL configured, skipping notification")
            return False

        payload = {
            "event": "pipeline_ready",
            "dvc_reference": dvc_ref,
            "metrics": metrics,
            "metadata": metadata or {},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.notification_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        notifications_sent.inc()
                        logger.info(f"Notification sent successfully to {self.notification_url}")
                        return True
                    else:
                        notifications_failed.inc()
                        logger.error(f"Notification failed with status {response.status}")
                        return False

        except Exception as e:
            notifications_failed.inc()
            logger.error(f"Failed to send notification: {e}")
            return False

    async def notify_processing_failed(self, error: str, event_info: Dict[str, Any]) -> bool:
        """Notify that processing failed.

        Args:
            error: Error message
            event_info: Information about the failed event

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.notification_url:
            return False

        payload = {"event": "processing_failed", "error": error, "event_info": event_info}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.notification_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status >= 200 and response.status < 300

        except Exception as e:
            logger.error(f"Failed to send failure notification: {e}")
            return False


async def notify_pipeline_ready(
    dvc_ref: str, notification_url: Optional[str] = None, **metrics
) -> bool:
    """Helper function to notify pipeline ready.

    Args:
        dvc_ref: DVC reference
        notification_url: URL to send notification to
        **metrics: Metrics as keyword arguments

    Returns:
        True if successful, False otherwise
    """
    service = NotificationService(notification_url)
    return await service.notify_pipeline_ready(dvc_ref, metrics)

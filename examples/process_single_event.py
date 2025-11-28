"""Example script demonstrating event processing."""

import asyncio
import json

from ingestion_agent.config import Config
from ingestion_agent.main import IngestionAgent


async def main():
    """Example of processing a single event."""
    # Configure agent
    config = Config(
        s3_bucket="drone-images",
        tile_size=256,
        normalize=True,
        apply_augmentation=False
    )
    
    # Create agent
    agent = IngestionAgent(config)
    
    # Example event
    event = {
        'bucket': 'drone-images',
        'key': 'images/sample_drone_001.jpg'
    }
    
    print(f"Processing event: {json.dumps(event, indent=2)}")
    
    # Process event
    await agent.process_event(event)
    
    print("Processing complete!")


if __name__ == "__main__":
    asyncio.run(main())

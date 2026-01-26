# Example Event Payloads

## S3 Event Notification

Example of an S3 event notification that triggers processing:

```json
{
  "Records": [
    {
      "eventVersion": "2.1",
      "eventSource": "aws:s3",
      "awsRegion": "us-east-1",
      "eventTime": "2025-01-15T10:30:00.000Z",
      "eventName": "ObjectCreated:Put",
      "s3": {
        "bucket": {
          "name": "drone-images",
          "arn": "arn:aws:s3:::drone-images"
        },
        "object": {
          "key": "images/drone_flight_001.jpg",
          "size": 2048000,
          "eTag": "abc123def456",
          "sequencer": "0055AED6DCD90281E5"
        }
      }
    }
  ]
}
```

## Pipeline Ready Notification

Example notification sent when processing completes:

```json
{
  "event": "pipeline_ready",
  "dvc_reference": "data/tiles.dvc",
  "metrics": {
    "num_images": 1,
    "num_tiles": 64,
    "source_key": "images/drone_flight_001.jpg",
    "output_path": "data/tiles"
  },
  "metadata": {
    "tiles": [
      {
        "tile_id": 0,
        "filename": "drone_flight_001_tile_0000.png",
        "source_image": "drone_flight_001.jpg",
        "bbox": {
          "x": 0,
          "y": 0,
          "width": 256,
          "height": 256
        },
        "size": {
          "width": 256,
          "height": 256
        },
        "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      }
    ]
  }
}
```

## Processing Failed Notification

Example notification sent when processing fails:

```json
{
  "event": "processing_failed",
  "error": "Image validation failed",
  "event_info": {
    "bucket": "drone-images",
    "key": "images/corrupted_image.jpg"
  }
}
```

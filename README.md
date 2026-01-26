# Ingestion Agent

Automated image ingestion and preprocessing agent for drone/camera images with DVC versioning.

## Overview

This agent monitors cloud storage buckets for new images, downloads them, applies preprocessing (tiling, normalization, augmentation), versions the data with DVC, and publishes events for downstream pipeline processing.

## Features

- **Automated Monitoring**: Watches S3/cloud buckets for new images via polling or event notifications
- **Image Processing**: 
  - Configurable image tiling (e.g., 256×256 patches)
  - Normalization and basic augmentations
  - Metadata extraction (GPS, timestamps, bounding boxes)
- **Data Versioning**: Automatic DVC integration for reproducible data pipelines
- **Resilience**: 
  - Exponential backoff retry logic
  - Configurable parallel processing limits
  - Dead-letter queue support
- **Observability**:
  - Structured JSON logging
  - Prometheus metrics endpoint
  - Distributed tracing ready
- **Security**: 
  - Credentials via environment variables/secrets
  - Encryption in transit

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   S3/Cloud  │─────▶│   Watcher    │─────▶│  Processor  │
│   Bucket    │      │   Module     │      │   Module    │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ Notification│◀─────│     DVC      │◀─────│   Tiles +   │
│   Service   │      │ Integration  │      │  Metadata   │
└─────────────┘      └──────────────┘      └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- DVC installed
- AWS credentials or MinIO for testing

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/luferdias/AI.git
   cd AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Configure environment variables**:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export S3_BUCKET="drone-images"
   export DVC_REMOTE="storage"
   export NOTIFICATION_URL="http://localhost:8080/pipeline/ready"
   ```

4. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/dvc-storage
   ```

5. **Run the agent**:
   ```bash
   python -m ingestion_agent.main
   ```

### Testing with MinIO

For local testing without AWS:

1. **Start MinIO**:
   ```bash
   docker run -p 9000:9000 -p 9001:9001 \
     -e MINIO_ROOT_USER=minioadmin \
     -e MINIO_ROOT_PASSWORD=minioadmin \
     minio/minio server /data --console-address ":9001"
   ```

2. **Configure for MinIO**:
   ```bash
   export AWS_ACCESS_KEY_ID="minioadmin"
   export AWS_SECRET_ACCESS_KEY="minioadmin"
   export S3_ENDPOINT_URL="http://localhost:9000"
   export S3_BUCKET="test-bucket"
   ```

3. **Create bucket and upload test images**:
   ```bash
   # Use MinIO console at http://localhost:9001
   # Or use AWS CLI with --endpoint-url
   aws --endpoint-url http://localhost:9000 s3 mb s3://test-bucket
   aws --endpoint-url http://localhost:9000 s3 cp test.jpg s3://test-bucket/images/
   ```

## Docker Deployment

### Build the image

```bash
docker build -t ingestion-agent:latest .
```

### Run the container

```bash
docker run -d \
  --name ingestion-agent \
  -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID="your-key" \
  -e AWS_SECRET_ACCESS_KEY="your-secret" \
  -e S3_BUCKET="drone-images" \
  -v $(pwd)/data:/app/data \
  ingestion-agent:latest
```

## Kubernetes Deployment

### Deploy to Kubernetes

1. **Update secrets** in `k8s/manifests/secret.yaml`:
   ```yaml
   stringData:
     AWS_ACCESS_KEY_ID: "your-access-key"
     AWS_SECRET_ACCESS_KEY: "your-secret-key"
   ```

2. **Apply manifests**:
   ```bash
   kubectl apply -f k8s/manifests/configmap.yaml
   kubectl apply -f k8s/manifests/secret.yaml
   kubectl apply -f k8s/manifests/deployment.yaml
   kubectl apply -f k8s/manifests/service.yaml
   kubectl apply -f k8s/manifests/hpa.yaml
   ```

3. **Check status**:
   ```bash
   kubectl get pods -l app=ingestion-agent
   kubectl logs -f deployment/ingestion-agent
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `AWS_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET` | Source bucket name | `drone-images` |
| `S3_ENDPOINT_URL` | Custom S3 endpoint (for MinIO) | - |
| `TILE_SIZE` | Tile size in pixels | `256` |
| `TILE_OVERLAP` | Overlap between tiles | `0` |
| `NORMALIZE` | Apply normalization | `true` |
| `APPLY_AUGMENTATION` | Apply augmentations | `false` |
| `MAX_PARALLEL_DOWNLOADS` | Max concurrent downloads | `5` |
| `MAX_RETRIES` | Max retry attempts | `3` |
| `RETRY_BACKOFF_FACTOR` | Backoff multiplier | `2.0` |
| `DVC_REMOTE` | DVC remote name | `storage` |
| `NOTIFICATION_URL` | Webhook URL for notifications | - |
| `LOG_LEVEL` | Logging level | `INFO` |
| `PROMETHEUS_PORT` | Metrics port | `8000` |

## Event Payload

### Pipeline Ready Event

When processing completes, the agent sends a POST request to `NOTIFICATION_URL`:

```json
{
  "event": "pipeline_ready",
  "dvc_reference": "data/tiles.dvc",
  "metrics": {
    "num_images": 10,
    "num_tiles": 640,
    "source_key": "images/drone_001.jpg",
    "output_path": "data/tiles"
  },
  "metadata": {
    "tiles": [
      {
        "tile_id": 0,
        "filename": "drone_001_tile_0000.png",
        "source_image": "drone_001.jpg",
        "bbox": {"x": 0, "y": 0, "width": 256, "height": 256},
        "checksum": "abc123..."
      }
    ]
  }
}
```

## Testing

### Run unit tests

```bash
pytest tests/unit/ -v
```

### Run tests with coverage

```bash
pytest tests/unit/ --cov=ingestion_agent --cov-report=html
```

### Run integration tests

```bash
pytest tests/integration/ -v -m integration
```

Note: Integration tests require MinIO to be running locally.

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `events_received_total`: Total events received
- `events_processed_total`: Total events processed successfully
- `events_failed_total`: Total failed events
- `images_downloaded_total`: Total images downloaded
- `images_validated_total`: Total images validated
- `tiles_generated_total`: Total tiles generated
- `dvc_add_success_total`: Successful DVC add operations
- `dvc_push_success_total`: Successful DVC push operations
- `notifications_sent_total`: Total notifications sent

### Logs

Structured JSON logs are written to stdout:

```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "name": "ingestion_agent.processor",
  "severity": "INFO",
  "message": "Generated 64 tiles from images/drone_001.jpg"
}
```

## Development

### Code formatting

```bash
black src/ tests/
isort src/ tests/
```

### Linting

```bash
flake8 src/ tests/ --max-line-length=100
mypy src/
```

### Pre-commit hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## CI/CD

GitHub Actions workflow runs on every push:
- Linting (black, isort, flake8)
- Unit tests with coverage
- Docker image build
- Security scanning with Trivy

## Troubleshooting

### Issue: Images not being processed

1. Check bucket polling is working: `kubectl logs deployment/ingestion-agent`
2. Verify S3 credentials are correct
3. Check bucket permissions allow ListBucket and GetObject

### Issue: DVC push fails

1. Verify DVC remote is configured: `dvc remote list`
2. Check remote credentials and permissions
3. Ensure git repository is initialized

### Issue: High memory usage

1. Reduce `MAX_PARALLEL_DOWNLOADS`
2. Decrease `TILE_SIZE`
3. Adjust HPA limits in Kubernetes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run linting and tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: https://github.com/luferdias/AI/issues
- Documentation: See `/docs` folder

## Roadmap

- [ ] Support for additional cloud providers (Azure, GCP)
- [ ] GPU acceleration for preprocessing
- [ ] Real-time streaming from SQS/Kafka
- [ ] Advanced augmentation strategies
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for video frame extraction

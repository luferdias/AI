# Quick Start Guide

Get the ingestion agent running in 5 minutes!

## Option 1: Local Development (Fastest)

```bash
# 1. Clone and install
git clone https://github.com/luferdias/AI.git
cd AI
pip install -r requirements.txt
pip install -e .

# 2. Set minimal configuration
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export S3_BUCKET="test-bucket"

# 3. Run tests to verify
pytest tests/unit/ -v

# Success! The agent is ready.
```

## Option 2: Docker (Recommended for Testing)

```bash
# 1. Build the image
docker build -t ingestion-agent:latest .

# 2. Run with environment variables
docker run -it --rm \
  -e AWS_ACCESS_KEY_ID="your-key" \
  -e AWS_SECRET_ACCESS_KEY="your-secret" \
  -e S3_BUCKET="drone-images" \
  -v $(pwd)/data:/app/data \
  ingestion-agent:latest

# The agent will start polling the bucket!
```

## Option 3: Kubernetes (Production)

```bash
# 1. Update secrets in k8s/manifests/secret.yaml
# 2. Deploy
kubectl apply -f k8s/manifests/

# 3. Check status
kubectl get pods -l app=ingestion-agent
kubectl logs -f deployment/ingestion-agent

# 4. Access metrics
kubectl port-forward svc/ingestion-agent 8000:8000
curl http://localhost:8000/metrics
```

## Option 4: Local MinIO Testing

Perfect for testing without AWS:

```bash
# 1. Start MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# 2. Configure for MinIO
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export S3_ENDPOINT_URL="http://localhost:9000"
export S3_BUCKET="test-bucket"

# 3. Create bucket and upload test image
# Visit http://localhost:9001 in browser
# Or use AWS CLI:
aws --endpoint-url http://localhost:9000 s3 mb s3://test-bucket
aws --endpoint-url http://localhost:9000 s3 cp test.jpg s3://test-bucket/images/

# 4. Run the agent
python -m ingestion_agent.main
```

## Verify It's Working

### Check Logs
You should see JSON logs like:
```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "name": "ingestion_agent.watcher",
  "severity": "INFO",
  "message": "Starting S3 polling on bucket test-bucket"
}
```

### Check Metrics
Visit http://localhost:8000/metrics to see:
```
events_received_total 5
events_processed_total 5
images_downloaded_total 5
tiles_generated_total 320
```

### Check Output
Look for processed tiles in `data/tiles/`:
```bash
ls -la data/tiles/
# You'll see subdirectories for each image with tiles and metadata.json
```

## Common Issues

### "Connection refused" when accessing S3
- Check your AWS credentials
- Verify S3_ENDPOINT_URL is set correctly for MinIO
- Ensure bucket exists and is accessible

### "No module named 'ingestion_agent'"
- Run: `pip install -e .` from the project root

### Tests failing
- Install dependencies: `pip install -r requirements.txt`
- Install test dependencies: `pip install pytest pytest-asyncio pytest-cov`

## What's Next?

1. **Configure DVC**: Set up DVC remote for data versioning
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/dvc-storage
   ```

2. **Set up notifications**: Configure `NOTIFICATION_URL` to receive pipeline events

3. **Deploy to production**: Use the Kubernetes manifests in `k8s/manifests/`

4. **Monitor**: Set up Prometheus to scrape the `/metrics` endpoint

5. **Scale**: Adjust `MAX_PARALLEL_DOWNLOADS` and HPA settings as needed

## Need Help?

- 📖 Full documentation: See [README.md](README.md)
- 🔧 Configuration reference: See [README.md#configuration](README.md#configuration)
- 💡 Examples: Check the `examples/` directory
- 🐛 Issues: https://github.com/luferdias/AI/issues

Happy processing! 🚀

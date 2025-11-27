# Concrete Crack Detection Pipeline

A complete production-ready pipeline for detecting cracks and pathologies in reinforced concrete from drone and camera imagery.

## 🎯 Features

- **Multi-Model Support**: U-Net, DeepLabV3+, and YOLOv8 architectures
- **High Performance**: Optimized for <200-500ms inference latency
- **High Recall**: Optimized detection for critical infrastructure monitoring
- **Production Ready**: Docker/Kubernetes deployment with monitoring
- **MLOps Integration**: MLflow for experiment tracking and model management
- **Data Versioning**: DVC for dataset version control
- **Comprehensive API**: FastAPI-based inference service

## 🏗️ Architecture

```
├── src/
│   ├── data/           # Data ingestion and preprocessing
│   ├── models/         # Model architectures (U-Net, DeepLab, YOLO)
│   ├── api/            # FastAPI inference service
│   └── utils/          # Annotation and utility functions
├── configs/            # Configuration files
├── deployments/        # Docker and Kubernetes configs
├── scripts/            # Training and inference scripts
├── tests/              # Unit and integration tests
└── data/               # Data storage (versioned with DVC)
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for production deployment)

### Setup

```bash
# Clone repository
git clone https://github.com/luferdias/AI.git
cd AI

# Install dependencies
pip install -r requirements.txt

# Initialize DVC (optional)
dvc init
dvc remote add -d storage s3://your-bucket/dvc-storage
```

## 🚀 Quick Start

### 1. Data Preparation

```python
from pathlib import Path
from src.data.ingestor import ImageIngestor
from src.data.preprocessor import ImagePreprocessor

# Ingest images from drone/camera
ingestor = ImageIngestor(data_dir="data/raw")
ingestor.ingest_from_directory("path/to/source/images")

# Preprocess images
preprocessor = ImagePreprocessor(target_size=(512, 512))
```

### 2. Training Models

#### U-Net Training

```bash
python scripts/train.py \
    --config configs/unet_config.yaml \
    --model unet \
    --output-dir models/
```

#### YOLOv8 Training

```bash
python scripts/train.py \
    --config configs/yolo_data.yaml \
    --model yolo \
    --output-dir models/
```

### 3. Running Inference

```bash
python scripts/inference.py \
    --model-type unet \
    --model-path models/best_model.pth \
    --image path/to/image.jpg
```

## 🌐 API Usage

### Starting the API Server

```bash
# Set environment variables
export MODEL_TYPE=unet
export MODEL_PATH=models/best_model.pth
export DEVICE=cuda

# Run server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Response Format

```json
{
  "success": true,
  "model_type": "unet",
  "inference_time_ms": 145.2,
  "detections": [
    {
      "bbox": [100, 150, 200, 250],
      "area": 5000,
      "confidence": 0.92
    }
  ],
  "confidence_score": 0.92
}
```

## 🐳 Docker Deployment

### Build and Run

```bash
cd deployments/docker

# Build image
docker build -t crack-detection-api -f Dockerfile ../..

# Run with docker-compose (includes MLflow, Prometheus, Grafana)
docker-compose up -d
```

### Access Services

- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## ☸️ Kubernetes Deployment

```bash
# Apply deployment
kubectl apply -f deployments/kubernetes/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment crack-detection-api --replicas=5
```

## 📊 Model Performance

### Target Metrics

- **Inference Latency**: < 200-500ms per image
- **Recall**: Optimized for high recall (minimize false negatives)
- **Precision**: Balanced to reduce false positives
- **Cost**: Optimized for cloud deployment

### Optimization Strategies

1. **Model Quantization**: INT8 quantization for faster inference
2. **Batch Processing**: Process multiple images together
3. **GPU Utilization**: CUDA optimization and mixed precision
4. **Model Pruning**: Remove redundant parameters
5. **ONNX Export**: Convert to ONNX for deployment

## 🔍 Monitoring

### MLflow Tracking

```python
import mlflow

# View experiments
mlflow.set_tracking_uri("http://localhost:5000")
experiments = mlflow.search_experiments()
```

### Prometheus Metrics

- `inference_requests_total`: Total number of inference requests
- `inference_latency_seconds`: Inference latency histogram
- `prediction_confidence`: Prediction confidence distribution

### Grafana Dashboards

Import the provided dashboard for visualization:
- Request rate
- Latency percentiles (p50, p95, p99)
- Error rates
- Model performance metrics

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/test_models.py -v
```

## 📝 Data Versioning with DVC

```bash
# Track data
dvc add data/raw
dvc add data/processed

# Commit and push
git add data/raw.dvc data/processed.dvc .gitignore
git commit -m "Add dataset"
dvc push

# Pull data on another machine
dvc pull
```

## 🔧 Configuration

### Model Configuration (configs/unet_config.yaml)

```yaml
model:
  type: "unet"
  n_channels: 3
  n_classes: 1

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  
inference:
  confidence_threshold: 0.5
  min_detection_area: 100
```

## 📚 Model Architectures

### U-Net
- Best for: Dense pixel-wise segmentation
- Parameters: ~7.8M
- Input size: 512x512
- Inference time: ~50-100ms (GPU)

### DeepLabV3+
- Best for: High-accuracy segmentation
- Parameters: ~41M
- Input size: 512x512
- Inference time: ~150-200ms (GPU)

### YOLOv8
- Best for: Fast object detection
- Parameters: 3-25M (depending on variant)
- Input size: 640x640
- Inference time: ~30-80ms (GPU)

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## 📖 API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch and torchvision for deep learning frameworks
- Ultralytics for YOLOv8 implementation
- FastAPI for the web framework
- MLflow for experiment tracking

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for infrastructure monitoring and safety**

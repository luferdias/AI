# AI - Concrete Crack Detection Pipeline

Complete production-ready pipeline for detecting cracks and pathologies in reinforced concrete from drone and camera imagery.

## 🎯 Overview

This repository implements a comprehensive ML pipeline for concrete crack detection, featuring:

- **Multiple Model Architectures**: U-Net, DeepLabV3+, and YOLOv8
- **High-Performance Inference**: Optimized for <200-500ms latency
- **Production-Ready Deployment**: Docker/Kubernetes with full monitoring
- **MLOps Integration**: MLflow tracking, DVC versioning
- **High Recall Optimization**: Critical for infrastructure safety

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py --config configs/unet_config.yaml --model unet

# Run inference
python scripts/inference.py --model-type unet --model-path models/best_model.pth --image sample.jpg

# Start API server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## 📚 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed instructions on:
- Installation and setup
- Model training and fine-tuning
- API usage and deployment
- Kubernetes scaling
- Monitoring and metrics

## 🏗️ Architecture

```
├── src/
│   ├── data/           # Ingestion, preprocessing, datasets
│   ├── models/         # U-Net, DeepLab, YOLO implementations
│   ├── api/            # FastAPI inference service
│   └── utils/          # Annotations and utilities
├── configs/            # Model configurations
├── deployments/        # Docker & Kubernetes
├── scripts/            # Training & inference scripts
└── tests/              # Unit & integration tests
```

## 🔧 Key Features

### Data Pipeline
- Multi-source image ingestion (drones, cameras)
- Advanced preprocessing and augmentation
- DVC-based version control

### Models
- **U-Net**: Fast segmentation (~50-100ms)
- **DeepLabV3+**: High-accuracy segmentation (~150-200ms)
- **YOLOv8**: Fast detection (~30-80ms)

### Deployment
- Docker containerization
- Kubernetes orchestration
- Auto-scaling (HPA)
- GPU support

### Monitoring
- Prometheus metrics
- Grafana dashboards
- MLflow experiment tracking
- Real-time performance monitoring

## 📊 Performance Targets

- **Latency**: < 500ms per image
- **Recall**: Maximized (critical for safety)
- **Throughput**: Auto-scaling to handle load
- **Cost**: Optimized for cloud deployment

## 🧪 Testing

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📄 License

MIT License - see LICENSE file for details.

---

**Built for infrastructure monitoring and safety**

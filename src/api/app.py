"""
FastAPI inference service for crack detection.
Optimized for low latency (<200-500ms) with monitoring.
"""

import io
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging

import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

from ..models.unet import UNet
from ..models.deeplab import DeepLabV3Plus
from ..models.yolo_detector import YOLOv8CrackDetector
from ..data.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_COUNTER = Counter(
    'inference_requests_total', 
    'Total inference requests',
    ['model_type', 'status']
)
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['model_type']
)
PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    ['model_type']
)

app = FastAPI(
    title="Concrete Crack Detection API",
    description="API for detecting cracks and pathologies in concrete structures",
    version="1.0.0"
)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    model_type: str
    inference_time_ms: float
    detections: List[Dict]
    segmentation_mask: Optional[str] = None
    confidence_score: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


class InferenceService:
    """Manages model loading and inference."""
    
    def __init__(
        self,
        model_type: str = 'unet',
        model_path: Optional[Path] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        target_size: tuple = (512, 512)
    ):
        """
        Initialize inference service.
        
        Args:
            model_type: Type of model ('unet', 'deeplab', 'yolo')
            model_path: Path to model weights
            device: Device to use for inference
            target_size: Target image size
        """
        self.model_type = model_type
        self.device = device
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Loaded {model_type} model on {device}")
    
    def _load_model(self, model_path: Optional[Path]):
        """Load model based on type."""
        if self.model_type == 'unet':
            model = UNet(n_channels=3, n_classes=1).to(self.device)
        elif self.model_type == 'deeplab':
            model = DeepLabV3Plus(n_channels=3, n_classes=1).to(self.device)
        elif self.model_type == 'yolo':
            if model_path:
                model = YOLOv8CrackDetector(pretrained=False)
                model.load(model_path)
            else:
                model = YOLOv8CrackDetector(pretrained=True)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights if provided
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from {model_path}")
        
        return model
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        
        try:
            if self.model_type == 'yolo':
                # YOLO inference
                results = self.model.predict(image, conf=0.25)
                detections = []
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        detections.append({
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0])
                        })
                
                avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.0
                
                result_dict = {
                    'detections': detections,
                    'confidence_score': avg_confidence
                }
            else:
                # Segmentation inference
                # Preprocess
                preprocessed = self.preprocessor.preprocess_single(image, for_training=False)
                preprocessed = preprocessed.unsqueeze(0).to(self.device)
                
                # Inference
                output = self.model(preprocessed)
                pred_mask = torch.sigmoid(output) > 0.5
                
                # Convert to numpy
                pred_mask = pred_mask.squeeze().cpu().numpy()
                
                # Calculate confidence as mean of probability
                confidence = torch.sigmoid(output).mean().item()
                
                # Find contours for detections
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    pred_mask_uint8, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                detections = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter small detections
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'area': int(cv2.contourArea(contour)),
                            'confidence': confidence
                        })
                
                result_dict = {
                    'detections': detections,
                    'confidence_score': confidence,
                    'mask_coverage': float(pred_mask.sum() / pred_mask.size)
                }
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update metrics
            INFERENCE_COUNTER.labels(
                model_type=self.model_type, 
                status='success'
            ).inc()
            INFERENCE_LATENCY.labels(
                model_type=self.model_type
            ).observe(inference_time / 1000)
            PREDICTION_CONFIDENCE.labels(
                model_type=self.model_type
            ).observe(result_dict['confidence_score'])
            
            result_dict['inference_time_ms'] = inference_time
            
            return result_dict
            
        except Exception as e:
            INFERENCE_COUNTER.labels(
                model_type=self.model_type,
                status='error'
            ).inc()
            logger.error(f"Inference error: {e}")
            raise


# Global inference service instance
inference_service: Optional[InferenceService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference service on startup."""
    global inference_service
    
    # Get configuration from environment variables
    import os
    model_type = os.getenv('MODEL_TYPE', 'unet')
    model_path = os.getenv('MODEL_PATH')
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_service = InferenceService(
        model_type=model_type,
        model_path=Path(model_path) if model_path else None,
        device=device
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_service is not None,
        device=inference_service.device if inference_service else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict cracks in uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = inference_service.predict(image_np)
        
        return PredictionResponse(
            success=True,
            model_type=inference_service.model_type,
            inference_time_ms=results['inference_time_ms'],
            detections=results['detections'],
            confidence_score=results['confidence_score']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Concrete Crack Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Upload image for crack detection",
            "/metrics": "Prometheus metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_TYPE=unet
ENV DEVICE=cuda

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

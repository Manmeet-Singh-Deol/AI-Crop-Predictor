# Production Dockerfile for AgroAI Crop Disease Predictor & Agronomist
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using lightweight CPU-only PyTorch wheels (avoids 2.5GB CUDA bloat)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir opencv-python-headless fastapi "uvicorn[standard]" pillow reportlab httpx python-multipart numpy

# Copy backend, frontend, model weights, and application files
COPY backend ./backend
COPY frontend ./frontend
COPY run.py .
COPY requirements.txt .

# Expose standard web port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["python", "run.py"]

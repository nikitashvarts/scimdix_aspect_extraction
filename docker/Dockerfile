# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including CUDA support
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Create directories for results
RUN mkdir -p results/experiments results/models results/logs

# Default command
CMD ["python3", "scripts/test_cpu_training.py"]
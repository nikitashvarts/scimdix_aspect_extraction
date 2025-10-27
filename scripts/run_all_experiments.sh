#!/bin/bash
# GPU Training Scripts for Aspect Extraction
# Uses GPU=1 (second GPU on the server)

# Set GPU to use
export CUDA_VISIBLE_DEVICES=1

# Build Docker image
echo "🐳 Building Docker image..."
docker build -f docker/Dockerfile -t aspect-extraction .

# Quick test (1 epoch, small batch)
echo "🧪 Running quick test on GPU=1..."
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py test

# Baseline experiments
echo "📊 Running baseline experiments on GPU=1..."
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py baseline_ru
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py baseline_kz

# Main zero-shot experiment  
echo "🎯 Running zero-shot experiment on GPU=1..."
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py zero_shot_ru_to_kz

# LODO experiments
echo "🔄 Running LODO experiments on GPU=1..."
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py lodo_it
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py lodo_ling
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py lodo_med
docker run --gpus '"device=1"' -v $(pwd)/results:/app/results -e CUDA_VISIBLE_DEVICES=1 aspect-extraction python3 scripts/run_experiment.py lodo_psy

echo "✅ All experiments completed!"
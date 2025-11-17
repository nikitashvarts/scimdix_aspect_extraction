#!/bin/bash
# Fallback script for servers with Docker issues

echo "🐍 Running experiment directly with Python (no Docker)"
echo "📋 Make sure you have activated virtual environment!"

EXPERIMENT=${1:-test}
BATCH_SIZE=${2:-2}
EPOCHS=${3:-1}

echo "🚀 Running experiment: $EXPERIMENT"
echo "📊 Configuration: batch_size=$BATCH_SIZE, epochs=$EPOCHS"

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Create results directory
mkdir -p results/experiments results/models results/logs

# Run directly
PYTHONPATH=. python3 scripts/run_experiment.py $EXPERIMENT --batch_size $BATCH_SIZE --epochs $EPOCHS --device cuda

echo "✅ Experiment completed!"
echo "📁 Results saved in: results/experiments/$EXPERIMENT/"
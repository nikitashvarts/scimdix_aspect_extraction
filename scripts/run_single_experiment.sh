#!/bin/bash
# Simple script to run individual experiments on GPU=1

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name> [batch_size] [epochs]"
    echo ""
    echo "Available experiments:"
    echo "  test                - Quick test (1 epoch)"
    echo "  baseline_ru         - Russian baseline"
    echo "  baseline_kz         - Kazakh baseline"
    echo "  zero_shot_ru_to_kz  - Main zero-shot experiment"
    echo "  lodo_it             - Leave-out IT domain"
    echo "  lodo_ling           - Leave-out linguistics domain"
    echo "  lodo_med            - Leave-out medical domain"
    echo "  lodo_psy            - Leave-out psychology domain"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 baseline_ru 16 10"
    echo "  $0 zero_shot_ru_to_kz 32 20"
    exit 1
fi

EXPERIMENT=$1
BATCH_SIZE=${2:-32}
EPOCHS=${3:-20}

echo "🚀 Running experiment: $EXPERIMENT"
echo "📊 Configuration: batch_size=$BATCH_SIZE, epochs=$EPOCHS"
echo "💻 Using GPU=1"

# Build image if it doesn't exist
if [[ "$(docker images -q aspect-extraction 2> /dev/null)" == "" ]]; then
    echo "🐳 Building Docker image..."
    
    # Try building with main Dockerfile first
    if docker build -f docker/Dockerfile -t aspect-extraction . 2>/dev/null; then
        echo "✅ Built with main Dockerfile"
    else
        echo "⚠️  Main Dockerfile failed, trying simple version..."
        if docker build -f docker/Dockerfile.simple -t aspect-extraction . 2>/dev/null; then
            echo "✅ Built with simple Dockerfile"
        else
            echo "❌ Both Dockerfiles failed. Check Docker installation and network."
            exit 1
        fi
    fi
fi

# Run experiment on GPU=1
docker run --gpus '"device=1"' \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/datasets:/app/datasets \
    -e CUDA_VISIBLE_DEVICES=1 \
    aspect-extraction \
    python3 scripts/run_experiment.py $EXPERIMENT --batch_size $BATCH_SIZE --epochs $EPOCHS --device cuda

echo "✅ Experiment completed!"
echo "📁 Results saved in: results/experiments/$EXPERIMENT/"
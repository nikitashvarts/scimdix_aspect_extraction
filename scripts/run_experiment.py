#!/usr/bin/env python3
"""
GPU training script for aspect extraction experiments.
Designed to run inside Docker container with CUDA support.
"""

import sys
import argparse
from src.model.trainer import run_experiment
from src.model.config import get_training_config

def main():
    parser = argparse.ArgumentParser(description="Run aspect extraction experiments")
    parser.add_argument(
        "experiment", 
        choices=[
            "baseline_ru", "baseline_kz", "zero_shot_ru_to_kz",
            "lodo_it", "lodo_ling", "lodo_med", "lodo_psy",
            "test"  # Quick test
        ],
        help="Experiment to run"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42], help="Random seeds")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting experiment: {args.experiment}")
    print(f"📊 Configuration: batch_size={args.batch_size}, epochs={args.epochs}, device={args.device}")
    print(f"🎲 Seeds: {args.seeds}")
    
    # Create training config
    if args.experiment == "test":
        # Quick test configuration
        training_config = get_training_config(
            batch_size=2,
            num_epochs=1
        )
        training_config.seeds = [42]
        experiment_name = "baseline_ru"  # Use small experiment
    else:
        # Full training configuration
        training_config = get_training_config(
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        training_config.device = args.device
        training_config.seeds = args.seeds
        experiment_name = args.experiment
    
    try:
        # Run experiment
        results = run_experiment(experiment_name, training_config)
        
        print("\n✅ Training completed successfully!")
        
        # Print summary
        if 'aggregated_metrics' in results:
            metrics = results['aggregated_metrics']
            print("\n📈 Final Results:")
            for metric_name in ['final_test_micro_f1_mean', 'final_test_macro_f1_mean']:
                if metric_name in metrics:
                    print(f"  {metric_name}: {metrics[metric_name]:.4f}")
        
        print(f"\n💾 Results saved to: results/experiments/{experiment_name}/")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
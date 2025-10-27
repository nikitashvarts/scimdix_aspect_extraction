"""
Quick CPU test for aspect extraction training pipeline.
Tests the complete system with minimal parameters for fast execution.
"""

import torch
from src.model.trainer import run_experiment
from src.model.config import get_training_config

def main():
    """Run a quick CPU test of the training pipeline."""
    
    print("🧪 Starting CPU Test for Aspect Extraction Training")
    print("=" * 60)
    
    # Force CPU usage
    device = "cpu"
    if torch.cuda.is_available():
        print("⚠️  CUDA available but forcing CPU for testing")
    
    print(f"💻 Device: {device}")
    
    # Create minimal training config for CPU
    training_config = get_training_config(
        experiment_type="baseline",
        batch_size=1,      # Very small batch for CPU
        num_epochs=2       # Just 2 epochs for quick test
    )
    
    # Override config for CPU testing
    training_config.device = device
    training_config.seeds = [42]  # Single seed for quick test
    training_config.early_stopping_patience = 1  # Early stop quickly
    training_config.eval_steps = 50  # Evaluate often
    training_config.logging_steps = 25  # Log often
    
    print(f"📊 Config: batch_size={training_config.batch_size}, "
          f"epochs={training_config.num_epochs}, seeds={training_config.seeds}")
    
    try:
        # Run baseline Russian experiment (smallest dataset)
        print("\n🚀 Starting baseline_ru experiment...")
        results = run_experiment(
            experiment_name="baseline_ru",
            training_config=training_config
        )
        
        print("\n✅ Training completed successfully!")
        
        # Print summary results
        if 'aggregated_metrics' in results:
            metrics = results['aggregated_metrics']
            print("\n📈 Final Results:")
            
            # Look for key metrics
            for metric_name in ['final_test_micro_f1_mean', 'final_test_macro_f1_mean', 'final_test_loss_mean']:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    print(f"  {metric_name}: {value:.4f}")
        
        print("\n💾 Results saved to: results/experiments/baseline_ru/")
        print("🎉 CPU test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🟢 All systems working! Ready for full training.")
    else:
        print("\n🔴 Test failed. Check configuration and data.")
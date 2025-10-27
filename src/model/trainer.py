"""
Trainer for aspect extraction model with multi-seed support, early stopping, and experiment tracking.
Supports different experimental scenarios: baseline, zero-shot, LODO.
"""

import os
import json
import random
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from datetime import datetime

from src.model.model import AspectExtractionModel, create_label_mapping
from src.model.config import ModelTrainingConfig, ExperimentConfig
from src.model.data_loader import create_data_loaders, get_file_paths_for_experiment
from src.model.evaluator import SpanLevelEvaluator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" for metrics to maximize, "min" for metrics to minimize
            restore_best_weights: Whether to restore best weights on early stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score to evaluate
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights from {self.patience} epochs ago")
        
        return self.early_stop


class AspectExtractionTrainer:
    """Trainer for aspect extraction with multi-seed support."""
    
    def __init__(
        self,
        model: AspectExtractionModel,
        training_config: ModelTrainingConfig,
        experiment_config: ExperimentConfig,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Aspect extraction model
            training_config: Training configuration
            experiment_config: Experiment configuration  
            train_loader: Training data loader
            test_loader: Test data loader
            val_loader: Optional validation data loader
        """
        self.model = model
        self.training_config = training_config
        self.experiment_config = experiment_config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        
        # Setup evaluator
        label_to_id, _ = create_label_mapping()
        self.evaluator = SpanLevelEvaluator(label_to_id)
        
        # Setup device
        self.device = torch.device(training_config.device)
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_scores = {}
        self.training_history = []
        
        # Create output directories
        self.setup_output_directories()
        
        logger.info(f"Trainer initialized for experiment: {experiment_config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    def setup_output_directories(self):
        """Create output directories for experiment."""
        self.output_dir = os.path.join(
            self.experiment_config.output_dir,
            self.experiment_config.experiment_name
        )
        self.model_dir = os.path.join(
            self.experiment_config.model_save_dir,
            self.experiment_config.experiment_name
        )
        self.logs_dir = os.path.join(
            self.experiment_config.logs_dir,
            self.experiment_config.experiment_name
        )
        
        for directory in [self.output_dir, self.model_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Set random seed to {seed}")
    
    def setup_optimizer_and_scheduler(self) -> Tuple[AdamW, Any]:
        """Setup optimizer and learning rate scheduler."""
        # Get parameter groups for dual learning rates
        encoder_params, head_crf_params = self.model.get_trainable_parameters()
        
        optimizer = AdamW([
            {
                'params': encoder_params,
                'lr': self.training_config.encoder_lr,
                'weight_decay': self.training_config.weight_decay
            },
            {
                'params': head_crf_params,
                'lr': self.training_config.head_crf_lr,
                'weight_decay': self.training_config.weight_decay
            }
        ])
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.training_config.num_epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer setup: encoder_lr={self.training_config.encoder_lr}, "
                   f"head_crf_lr={self.training_config.head_crf_lr}")
        logger.info(f"Scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.training_config.num_epochs}",
            leave=False
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr_enc': f"{scheduler.get_last_lr()[0]:.2e}",
                'lr_head': f"{scheduler.get_last_lr()[1]:.2e}"
            })
        
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'learning_rate_encoder': scheduler.get_last_lr()[0],
            'learning_rate_head_crf': scheduler.get_last_lr()[1]
        }
        
        return epoch_metrics
    
    def evaluate(self, data_loader: DataLoader, split_name: str = "eval") -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_attention_masks = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                predictions = outputs['predictions']
                
                # Collect results
                total_loss += loss.item()
                num_batches += 1
                
                # Convert to lists for metric calculation
                labels = batch['labels'].cpu().numpy()
                preds = predictions.cpu().numpy()
                attention_mask = batch['attention_mask'].cpu().numpy()
                
                all_predictions.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_attention_masks.extend(attention_mask.tolist())
        
        # Calculate basic metrics
        avg_loss = total_loss / num_batches
        
        # Calculate span-level metrics using evaluator
        try:
            eval_results = self.evaluator.evaluate(
                true_labels_list=all_labels,
                pred_labels_list=all_predictions,
                attention_masks_list=all_attention_masks
            )
            
            # Extract key metrics
            micro_f1 = eval_results['micro_avg'].f1
            macro_f1 = eval_results['macro_avg'].f1
            micro_precision = eval_results['micro_avg'].precision
            micro_recall = eval_results['micro_avg'].recall
            
        except Exception as e:
            logger.warning(f"Failed to calculate span metrics: {e}")
            micro_f1 = 0.0
            macro_f1 = 0.0
            micro_precision = 0.0
            micro_recall = 0.0
        
        metrics = {
            f'{split_name}_loss': avg_loss,
            f'{split_name}_micro_f1': micro_f1,
            f'{split_name}_macro_f1': macro_f1,
            f'{split_name}_micro_precision': micro_precision,
            f'{split_name}_micro_recall': micro_recall,
            f'{split_name}_samples': len(all_predictions)
        }
        
        return metrics
    
    def train_single_seed(self, seed: int) -> Dict[str, Any]:
        """Train model with a single seed."""
        logger.info(f"Starting training with seed {seed}")
        
        # Set seed
        self.set_seed(seed)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler()
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience,
            mode=self.training_config.early_stopping_mode,
            restore_best_weights=True
        )
        
        # Training history for this seed
        seed_history = []
        
        # Training loop
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(optimizer, scheduler, epoch)
            
            # Evaluate
            if self.val_loader is not None:
                eval_metrics = self.evaluate(self.val_loader, "val")
            else:
                eval_metrics = self.evaluate(self.test_loader, "test")
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['seed'] = seed
            
            seed_history.append(epoch_metrics)
            
            # Logging
            logger.info(
                f"Seed {seed}, Epoch {epoch+1}/{self.training_config.num_epochs} - "
                f"train_loss: {train_metrics['train_loss']:.4f}, "
                f"val_loss: {eval_metrics.get('val_loss', eval_metrics.get('test_loss', 0)):.4f}"
            )
            
            # Early stopping check
            eval_score = eval_metrics.get(
                self.training_config.early_stopping_metric,
                eval_metrics.get('val_loss', eval_metrics.get('test_loss', 0))
            )
            
            if early_stopping(eval_score, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Final evaluation
        final_test_metrics = self.evaluate(self.test_loader, "final_test")
        
        # Save model for this seed
        model_path = os.path.join(self.model_dir, f"model_seed_{seed}.pt")
        self.model.save_model(model_path)
        
        return {
            'seed': seed,
            'training_history': seed_history,
            'final_metrics': final_test_metrics,
            'model_path': model_path,
            'best_score': early_stopping.best_score
        }
    
    def train_multi_seed(self) -> Dict[str, Any]:
        """Train model with multiple seeds and aggregate results."""
        logger.info(f"Starting multi-seed training with seeds: {self.training_config.seeds}")
        
        all_results = []
        aggregated_metrics = {}
        
        for seed in self.training_config.seeds:
            seed_results = self.train_single_seed(seed)
            all_results.append(seed_results)
        
        # Aggregate final metrics across seeds
        final_metrics_by_seed = [r['final_metrics'] for r in all_results]
        
        for metric_name in final_metrics_by_seed[0].keys():
            values = [metrics[metric_name] for metrics in final_metrics_by_seed]
            aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregated_metrics[f"{metric_name}_std"] = np.std(values)
            aggregated_metrics[f"{metric_name}_values"] = values
        
        # Save results
        results = {
            'experiment_config': asdict(self.experiment_config),
            'training_config': asdict(self.training_config),
            'seed_results': all_results,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Multi-seed training completed. Results saved to {results_path}")
        
        return results


def run_experiment(
    experiment_name: str,
    training_config: Optional[ModelTrainingConfig] = None,
    data_dir: str = "datasets/prepared"
) -> Dict[str, Any]:
    """
    Run a complete experiment with given configuration.
    
    Args:
        experiment_name: Name of experiment to run
        training_config: Optional training config override
        data_dir: Directory containing prepared data
        
    Returns:
        Experiment results dictionary
    """
    from src.model.config import get_experiment_config, get_training_config
    
    # Get configurations
    experiment_config = get_experiment_config(experiment_name)
    if training_config is None:
        training_config = get_training_config()
    
    # Print configuration summary
    from src.model.config import print_config_summary
    print_config_summary(training_config, experiment_config)
    
    # Setup model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    label_to_id, id_to_label = create_label_mapping()
    
    model = AspectExtractionModel(
        model_name=training_config.model_name,
        num_labels=len(label_to_id),
        dropout_rate=training_config.dropout_rate
    )
    
    # Get file paths
    train_files, test_files = get_file_paths_for_experiment(experiment_config, data_dir)
    
    logger.info(f"Training files: {train_files}")
    logger.info(f"Test files: {test_files}")
    
    # Create data loaders
    train_loader, test_loader, val_loader = create_data_loaders(
        train_files=train_files,
        test_files=test_files,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        batch_size=training_config.batch_size,
        max_length=training_config.max_length
    )
    
    # Create trainer
    trainer = AspectExtractionTrainer(
        model=model,
        training_config=training_config,
        experiment_config=experiment_config,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader
    )
    
    # Run training
    results = trainer.train_multi_seed()
    
    return results


# Example usage
if __name__ == "__main__":
    # Test training setup
    try:
        print("Testing trainer setup...")
        
        # Quick test with small config
        from src.model.config import get_training_config, get_experiment_config
        
        training_config = get_training_config(
            experiment_type="baseline",
            batch_size=2,  # Small for testing
            num_epochs=1   # Just one epoch for testing
        )
        training_config.seeds = [42]  # Single seed for testing
        
        experiment_config = get_experiment_config("baseline_ru")
        
        print(f"Training config: batch_size={training_config.batch_size}, epochs={training_config.num_epochs}")
        print(f"Experiment: {experiment_config.experiment_name}")
        
        # Test file path generation
        from src.model.data_loader import get_file_paths_for_experiment
        train_files, test_files = get_file_paths_for_experiment(experiment_config)
        
        print(f"Train files: {train_files}")
        print(f"Test files: {test_files}")
        
        print("✅ Trainer setup test completed!")
        
    except Exception as e:
        print(f"❌ Error in trainer setup: {e}")
        import traceback
        traceback.print_exc()
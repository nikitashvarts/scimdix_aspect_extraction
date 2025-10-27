"""
Visualization module for training progress and metrics.
Creates real-time plots that update during training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import logging

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Real-time visualization of training progress."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
            experiment_name: Name of the experiment
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.plots_dir = os.path.join(save_dir, "plots")
        
        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Training history storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'epochs': [],
            'steps': [],
            'step_losses': []
        }
        
        # Plot file paths
        self.loss_plot_path = os.path.join(self.plots_dir, "training_loss.png")
        self.metrics_plot_path = os.path.join(self.plots_dir, "training_metrics.png")
        self.step_loss_plot_path = os.path.join(self.plots_dir, "step_losses.png")
        
        logger.info(f"Visualizer initialized for {experiment_name}")
        logger.info(f"Plots will be saved to: {self.plots_dir}")
    
    def log_step_loss(self, step: int, loss: float):
        """Log loss for individual training step."""
        self.history['steps'].append(step)
        self.history['step_losses'].append(loss)
        
        # Update step loss plot every 10 steps
        if step % 10 == 0:
            self.plot_step_losses()
    
    def log_epoch_metrics(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float,
        train_f1: float = None,
        val_f1: float = None
    ):
        """Log metrics for completed epoch."""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if train_f1 is not None:
            self.history['train_f1'].append(train_f1)
        if val_f1 is not None:
            self.history['val_f1'].append(val_f1)
        
        # Update all plots
        self.plot_training_loss()
        self.plot_training_metrics()
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if train_f1 and val_f1:
            logger.info(f"Epoch {epoch}: train_f1={train_f1:.4f}, val_f1={val_f1:.4f}")
    
    def plot_training_loss(self):
        """Plot training and validation loss over epochs."""
        if len(self.history['epochs']) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['train_loss'], 
                'b-', label='Training Loss', linewidth=2)
        plt.plot(self.history['epochs'], self.history['val_loss'], 
                'r-', label='Validation Loss', linewidth=2)
        
        plt.title(f'Training Progress - {self.experiment_name}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add current values as text
        if self.history['train_loss']:
            latest_train = self.history['train_loss'][-1]
            latest_val = self.history['val_loss'][-1]
            plt.text(0.02, 0.98, f'Latest: Train={latest_train:.4f}, Val={latest_val:.4f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Loss plot updated: {self.loss_plot_path}")
    
    def plot_training_metrics(self):
        """Plot F1 scores over epochs."""
        if len(self.history['epochs']) == 0 or len(self.history['train_f1']) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'][:len(self.history['train_f1'])], 
                self.history['train_f1'], 
                'g-', label='Training F1', linewidth=2)
        plt.plot(self.history['epochs'][:len(self.history['val_f1'])], 
                self.history['val_f1'], 
                'orange', label='Validation F1', linewidth=2)
        
        plt.title(f'F1 Scores - {self.experiment_name}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add current values as text
        if self.history['train_f1']:
            latest_train_f1 = self.history['train_f1'][-1]
            latest_val_f1 = self.history['val_f1'][-1] if self.history['val_f1'] else 0
            plt.text(0.02, 0.98, f'Latest: Train F1={latest_train_f1:.4f}, Val F1={latest_val_f1:.4f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.metrics_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Metrics plot updated: {self.metrics_plot_path}")
    
    def plot_step_losses(self):
        """Plot loss for individual training steps."""
        if len(self.history['steps']) == 0:
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['steps'], self.history['step_losses'], 
                'b-', alpha=0.7, linewidth=1)
        
        # Add smoothed line if enough data points
        if len(self.history['step_losses']) > 20:
            window_size = min(50, len(self.history['step_losses']) // 10)
            smoothed = np.convolve(self.history['step_losses'], 
                                 np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = self.history['steps'][window_size-1:]
            plt.plot(smoothed_steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.legend()
        
        plt.title(f'Step-by-Step Loss - {self.experiment_name}', fontsize=14)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add latest loss value
        if self.history['step_losses']:
            latest_loss = self.history['step_losses'][-1]
            latest_step = self.history['steps'][-1]
            plt.text(0.02, 0.98, f'Step {latest_step}: Loss={latest_loss:.4f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.step_loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Step loss plot updated: {self.step_loss_plot_path}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.plots_dir, "training_history.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'history': serializable_history
            }, f, indent=2)
        
        logger.info(f"Training history saved to: {history_path}")
    
    def create_final_summary_plot(self):
        """Create a comprehensive summary plot with all metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Summary - {self.experiment_name}', fontsize=16)
        
        # Loss plot
        if self.history['epochs']:
            axes[0, 0].plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Train')
            axes[0, 0].plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Validation')
            axes[0, 0].set_title('Loss Over Epochs')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # F1 scores
        if self.history['train_f1']:
            axes[0, 1].plot(self.history['epochs'][:len(self.history['train_f1'])], 
                           self.history['train_f1'], 'g-', label='Train F1')
            axes[0, 1].plot(self.history['epochs'][:len(self.history['val_f1'])], 
                           self.history['val_f1'], 'orange', label='Val F1')
            axes[0, 1].set_title('F1 Scores Over Epochs')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        
        # Step losses
        if self.history['steps']:
            axes[1, 0].plot(self.history['steps'], self.history['step_losses'], 'b-', alpha=0.6)
            axes[1, 0].set_title('Step-by-Step Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training statistics
        axes[1, 1].axis('off')
        
        # Calculate statistics safely
        final_train_loss = f"{self.history['train_loss'][-1]:.4f}" if self.history['train_loss'] else 'N/A'
        final_val_loss = f"{self.history['val_loss'][-1]:.4f}" if self.history['val_loss'] else 'N/A'
        best_train_f1 = f"{max(self.history['train_f1']):.4f}" if self.history['train_f1'] else 'N/A'
        best_val_f1 = f"{max(self.history['val_f1']):.4f}" if self.history['val_f1'] else 'N/A'
        
        stats_text = f"""
Training Statistics:

Total Epochs: {len(self.history['epochs'])}
Total Steps: {len(self.history['steps'])}

Final Training Loss: {final_train_loss}
Final Validation Loss: {final_val_loss}

Best Training F1: {best_train_f1}
Best Validation F1: {best_val_f1}

Experiment: {self.experiment_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        summary_path = os.path.join(self.plots_dir, "training_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Final summary plot saved to: {summary_path}")


# Example usage
if __name__ == "__main__":
    import time
    
    # Test visualizer
    visualizer = TrainingVisualizer("test_results", "test_experiment")
    
    # Simulate training progress
    for epoch in range(1, 6):
        # Simulate decreasing loss
        train_loss = 2.0 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
        val_loss = train_loss + np.random.normal(0, 0.05)
        
        # Simulate increasing F1
        train_f1 = 1 - np.exp(-epoch * 0.5) + np.random.normal(0, 0.02)
        val_f1 = train_f1 - np.random.normal(0.05, 0.02)
        
        visualizer.log_epoch_metrics(epoch, train_loss, val_loss, train_f1, val_f1)
        
        # Simulate step losses within epoch
        for step in range((epoch-1)*10, epoch*10):
            step_loss = train_loss + np.random.normal(0, 0.2)
            visualizer.log_step_loss(step, step_loss)
        
        time.sleep(0.1)  # Small delay for demonstration
    
    # Save final results
    visualizer.save_history()
    visualizer.create_final_summary_plot()
    
    print("✅ Visualization test completed!")
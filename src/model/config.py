"""
Training configuration for aspect extraction model.
Contains all hyperparameters and settings for different experimental scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class ModelTrainingConfig:
    """Configuration for model training parameters."""
    
    # Model architecture
    model_name: str = "xlm-roberta-large"
    max_length: int = 192  # Maximum sequence length
    dropout_rate: float = 0.1
    
    # Learning rates (dual LR strategy)
    encoder_lr: float = 2e-5     # Lower LR for pretrained encoder
    head_crf_lr: float = 1e-4    # Higher LR for classification head + CRF
    
    # Training parameters
    batch_size: int = 64         # For GPU (reduce to 1-2 for CPU)
    num_epochs: int = 20
    max_grad_norm: float = 1.0   # Gradient clipping
    weight_decay: float = 0.01
    
    # Learning rate scheduling
    warmup_ratio: float = 0.1    # 10% of training steps for warmup
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"
    
    # Early stopping
    early_stopping_patience: int = 8
    early_stopping_metric: str = "test_micro_f1"  # Use test_micro_f1 since we don't have val split
    early_stopping_mode: str = "max"  # "max" for F1, "min" for loss
    
    # Evaluation and logging
    eval_steps: int = 200        # Evaluate every N steps
    save_steps: int = 200        # Save checkpoint every N steps
    logging_steps: int = 10      # Log metrics every N steps
    
    # Reproducibility
    seeds: List[int] = None      # Multiple seeds for statistical significance
    
    # Device settings
    device: str = "auto"         # "auto", "cpu", "cuda"
    fp16: bool = False           # Mixed precision training
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.seeds is None:
            self.seeds = [13, 21, 42]  # Default seeds as requested
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust batch size for CPU
        if self.device == "cpu" and self.batch_size > 2:
            print(f"Warning: Reducing batch_size from {self.batch_size} to 2 for CPU training")
            self.batch_size = 2


@dataclass
class ExperimentConfig:
    """Configuration for different experimental scenarios."""
    
    # Experiment type
    experiment_name: str = "baseline"
    experiment_type: str = "baseline"  # "baseline", "zero_shot", "lodo", "cross_domain_lang"
    
    # Data configuration
    train_languages: List[str] = None
    test_languages: List[str] = None
    train_domains: List[str] = None
    test_domains: List[str] = None
    exclude_domain: Optional[str] = None  # For LODO experiments
    
    # Output paths
    output_dir: str = "results/experiments"
    model_save_dir: str = "results/models"
    logs_dir: str = "results/logs"
    
    def __post_init__(self):
        """Set default configurations for different experiment types."""
        if self.train_languages is None:
            self.train_languages = ["ru"]
        if self.test_languages is None:
            self.test_languages = ["ru"]
        if self.train_domains is None:
            self.train_domains = ["it", "ling", "med", "psy"]
        if self.test_domains is None:
            self.test_domains = ["it", "ling", "med", "psy"]


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    # Baseline experiments (in-language)
    "baseline_ru_ru_all_domains": ExperimentConfig(
        experiment_name="baseline_ru_ru_all_domains",
        experiment_type="baseline",
        train_languages=["ru"],
        test_languages=["ru"]
    ),
    
    "baseline_kz_kz_all_domains": ExperimentConfig(
        experiment_name="baseline_kz_kz_all_domains", 
        experiment_type="baseline",
        train_languages=["kz"],
        test_languages=["kz"]
    ),
    
    # Zero-shot transfer (main experiment)
    "zero_shot_ru_to_kz_all_domains": ExperimentConfig(
        experiment_name="zero_shot_ru_to_kz_all_domains",
        experiment_type="zero_shot",
        train_languages=["ru"],
        test_languages=["kz"]
    ),
    
    # LODO (Leave-One-Domain-Out) experiments
    "lodo_it_ru_ru": ExperimentConfig(
        experiment_name="lodo_it_ru_ru",
        experiment_type="lodo", 
        train_domains=["ling", "med", "psy"],
        test_domains=["it"],
        exclude_domain="it",
        train_languages=["ru"],
        test_languages=["ru"]
    ),
    
    "lodo_ling_ru_ru": ExperimentConfig(
        experiment_name="lodo_ling_ru_ru",
        experiment_type="lodo",
        train_domains=["it", "med", "psy"], 
        test_domains=["ling"],
        exclude_domain="ling",
        train_languages=["ru"],
        test_languages=["ru"]
    ),
    
    "lodo_med_ru_ru": ExperimentConfig(
        experiment_name="lodo_med_ru_ru",
        experiment_type="lodo",
        train_domains=["it", "ling", "psy"],
        test_domains=["med"],
        exclude_domain="med",
        train_languages=["ru"],
        test_languages=["ru"]
    ),
    
    "lodo_psy_ru_ru": ExperimentConfig(
        experiment_name="lodo_psy_ru_ru", 
        experiment_type="lodo",
        train_domains=["it", "ling", "med"],
        test_domains=["psy"],
        exclude_domain="psy",
        train_languages=["ru"],
        test_languages=["ru"]
    ),
    
    "lodo_it_kz_kz": ExperimentConfig(
        experiment_name="lodo_it_kz_kz",
        experiment_type="lodo", 
        train_domains=["ling", "med", "psy"],
        test_domains=["it"],
        exclude_domain="it",
        train_languages=["kz"],
        test_languages=["kz"]
    ),
    
    "lodo_ling_kz_kz": ExperimentConfig(
        experiment_name="lodo_ling_kz_kz",
        experiment_type="lodo",
        train_domains=["it", "med", "psy"], 
        test_domains=["ling"],
        exclude_domain="ling",
        train_languages=["kz"],
        test_languages=["kz"]
    ),
    
    "lodo_med_kz_kz": ExperimentConfig(
        experiment_name="lodo_med_kz_kz",
        experiment_type="lodo",
        train_domains=["it", "ling", "psy"],
        test_domains=["med"],
        exclude_domain="med",
        train_languages=["kz"],
        test_languages=["kz"]
    ),
    
    "lodo_psy_kz_kz": ExperimentConfig(
        experiment_name="lodo_psy_kz_kz", 
        experiment_type="lodo",
        train_domains=["it", "ling", "med"],
        test_domains=["psy"],
        exclude_domain="psy",
        train_languages=["kz"],
        test_languages=["kz"]
    ),

    # Cross-domain + cross-language experiments
    "cross_lodo_it_ru_kz": ExperimentConfig(
        experiment_name="cross_lodo_it_ru_kz",
        experiment_type="cross_domain_lang", 
        train_domains=["ling", "med", "psy"],
        test_domains=["it"],
        exclude_domain="it",
        train_languages=["ru"],
        test_languages=["kz"]
    ),
    
    "cross_lodo_ling_ru_kz": ExperimentConfig(
        experiment_name="cross_lodo_ling_ru_kz",
        experiment_type="cross_domain_lang",
        train_domains=["it", "med", "psy"], 
        test_domains=["ling"],
        exclude_domain="ling",
        train_languages=["ru"],
        test_languages=["kz"]
    ),
    
    "cross_lodo_med_ru_kz": ExperimentConfig(
        experiment_name="cross_lodo_med_ru_kz",
        experiment_type="cross_domain_lang",
        train_domains=["it", "ling", "psy"],
        test_domains=["med"],
        exclude_domain="med",
        train_languages=["ru"],
        test_languages=["kz"]
    ),
    
    "cross_lodo_psy_ru_kz": ExperimentConfig(
        experiment_name="cross_lodo_psy_ru_kz", 
        experiment_type="cross_domain_lang",
        train_domains=["it", "ling", "med"],
        test_domains=["psy"],
        exclude_domain="psy",
        train_languages=["ru"],
        test_languages=["kz"]
    )
}


def get_training_config(
    experiment_type: str = "baseline",
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    **kwargs
) -> ModelTrainingConfig:
    """
    Get training configuration with optional overrides.
    
    Args:
        experiment_type: Type of experiment
        batch_size: Override default batch size
        num_epochs: Override default number of epochs
        **kwargs: Additional configuration overrides
        
    Returns:
        ModelTrainingConfig instance
    """
    config = ModelTrainingConfig()
    
    # Apply overrides
    if batch_size is not None:
        config.batch_size = batch_size
    if num_epochs is not None:
        config.num_epochs = num_epochs
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    return config


def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """
    Get predefined experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        KeyError: If experiment name is not found
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise KeyError(f"Unknown experiment '{experiment_name}'. Available: {available}")
    
    return EXPERIMENT_CONFIGS[experiment_name]


def print_config_summary(
    training_config: ModelTrainingConfig,
    experiment_config: ExperimentConfig
):
    """Print a summary of training and experiment configuration."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Experiment: {experiment_config.experiment_name}")
    print(f"   Type: {experiment_config.experiment_type}")
    print(f"   Train languages: {experiment_config.train_languages}")
    print(f"   Test languages: {experiment_config.test_languages}")
    print(f"   Train domains: {experiment_config.train_domains}")
    print(f"   Test domains: {experiment_config.test_domains}")
    
    print(f"\n🧠 Model: {training_config.model_name}")
    print(f"   Max length: {training_config.max_length}")
    print(f"   Dropout: {training_config.dropout_rate}")
    
    print("\n🎯 Training:")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Epochs: {training_config.num_epochs}")
    print(f"   Encoder LR: {training_config.encoder_lr}")
    print(f"   Head+CRF LR: {training_config.head_crf_lr}")
    print(f"   Weight decay: {training_config.weight_decay}")
    print(f"   Grad clipping: {training_config.max_grad_norm}")
    print(f"   Seeds: {training_config.seeds}")
    
    print("\n💻 Hardware:")
    print(f"   Device: {training_config.device}")
    print(f"   FP16: {training_config.fp16}")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test configuration creation
    training_config = get_training_config("baseline", batch_size=16)
    experiment_config = get_experiment_config("zero_shot_ru_to_kz")
    
    print_config_summary(training_config, experiment_config)
    
    print("\nAvailable experiments:")
    for name in EXPERIMENT_CONFIGS.keys():
        print(f"  - {name}")
    
    print("\n✅ Configuration test completed!")
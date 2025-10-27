"""
Neural network model for aspect extraction using XLM-RoBERTa + CRF.
Implements token-level classification with conditional random fields for better sequence labeling.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from typing import Optional, Tuple, Dict, Any

from src.config import config


class AspectExtractionModel(nn.Module):
    """
    XLM-RoBERTa + Linear + CRF model for aspect extraction.
    
    Architecture:
    1. XLM-RoBERTa encoder (frozen/unfrozen based on config)
    2. Linear classification head 
    3. CRF layer for sequence labeling
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 15,  # 7 aspects × 2 (B-/I-) + O
        dropout_rate: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize the model.
        
        Args:
            model_name: Pretrained model name
            num_labels: Number of BIO labels
            dropout_rate: Dropout probability
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load pretrained model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier layer weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            labels: True labels [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary with loss, logits, and predictions
        """
        # Encoder forward pass
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout and classification
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_labels]
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Calculate CRF loss
            # Create mask for CRF: include attention mask but ensure first timestep is always on
            crf_mask = attention_mask.bool()
            
            # Replace -100 with 0 for CRF (CRF doesn't handle -100)
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0  # Use 'O' label for ignored tokens
            
            # CRF loss (negative log likelihood)
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
            outputs["loss"] = loss
        
        # Get predictions using CRF decode
        crf_mask = attention_mask.bool()
        predictions = self.crf.decode(logits, mask=crf_mask)
        
        # Pad predictions to same length as input
        batch_size, max_len = input_ids.shape
        padded_predictions = []
        for pred in predictions:
            padded_pred = pred + [0] * (max_len - len(pred))  # Pad with 0 (O label)
            padded_predictions.append(padded_pred[:max_len])  # Ensure exact length
        
        outputs["predictions"] = torch.tensor(padded_predictions, device=input_ids.device)
        
        return outputs
    
    def get_trainable_parameters(self) -> Tuple[list, list]:
        """
        Get separate parameter groups for different learning rates.
        
        Returns:
            Tuple of (encoder_params, head_params) for dual learning rates
        """
        encoder_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('encoder.'):
                    encoder_params.append(param)
                else:  # classifier and CRF parameters
                    head_params.append(param)
        
        return encoder_params, head_params
    
    def save_model(self, save_path: str):
        """Save model to file (for trainer compatibility)."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'config': self.config
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'config': self.config
        }, os.path.join(save_directory, 'pytorch_model.bin'))
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """Load model from directory."""
        import os
        
        checkpoint = torch.load(
            os.path.join(model_directory, 'pytorch_model.bin'),
            map_location='cpu'
        )
        
        # Create model with saved configuration
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_directory}")
        return model
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load model from single file (for trainer compatibility)."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model with saved configuration
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_path}")
        return model


def create_label_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create mapping between labels and IDs.
    
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    labels = ['O']  # Outside label
    
    # Add B- and I- labels for each aspect
    for aspect in config.ASPECT_CATEGORIES:
        labels.extend([f'B-{aspect}', f'I-{aspect}'])
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def get_model_info(model: AspectExtractionModel) -> Dict[str, Any]:
    """
    Get model information and statistics.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    encoder_params, head_params = model.get_trainable_parameters()
    encoder_param_count = sum(p.numel() for p in encoder_params)
    head_param_count = sum(p.numel() for p in head_params)
    
    return {
        'model_name': model.model_name,
        'num_labels': model.num_labels,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'encoder_parameters': encoder_param_count,
        'head_parameters': head_param_count,
        'encoder_trainable': encoder_param_count > 0,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Rough estimate
    }


# Example usage and testing
if __name__ == "__main__":
    # Create label mapping
    label2id, id2label = create_label_mapping()
    print(f"Created {len(label2id)} labels: {list(label2id.keys())}")
    
    # Create model
    model = AspectExtractionModel(num_labels=len(label2id))
    print(f"Model created: {model.__class__.__name__}")
    
    # Print model info
    info = get_model_info(model)
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, len(label2id), (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)
        print("\nTest forward pass:")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Predictions shape: {outputs['predictions'].shape}")
        
    print("\n✅ Model test completed successfully!")
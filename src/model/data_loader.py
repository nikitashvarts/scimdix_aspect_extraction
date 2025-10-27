"""
Data loader for CoNLL format files with XLM-RoBERTa tokenization alignment.
Handles loading, tokenization, and batch creation for aspect extraction training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """A single training/test example for aspect extraction."""
    
    text_tokens: List[str]  # Original word tokens
    labels: List[str]       # BIO labels for each token
    text: str              # Original text (for reference)
    
    def __post_init__(self):
        """Validate that tokens and labels have same length."""
        if len(self.text_tokens) != len(self.labels):
            raise ValueError(
                f"Tokens ({len(self.text_tokens)}) and labels ({len(self.labels)}) "
                f"must have same length. Text: {self.text[:50]}..."
            )


@dataclass
class InputFeatures:
    """Features for a single example after tokenization."""
    
    input_ids: List[int]           # Token IDs for XLM-RoBERTa
    attention_mask: List[int]      # Attention mask
    labels: List[int]              # Label IDs (-100 for special tokens)
    original_tokens: List[str]     # Original word tokens (for debugging)
    subword_mask: List[bool]       # True for first subword of each word
    
    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to tensor dictionary for model input."""
        return {
            'input_ids': torch.tensor(self.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels, dtype=torch.long)
        }


class CoNLLDataset(Dataset):
    """Dataset class for loading CoNLL format files."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 384,
        pad_token_label_id: int = -100
    ):
        """
        Initialize CoNLL dataset.
        
        Args:
            file_path: Path to CoNLL format file
            tokenizer: XLM-RoBERTa tokenizer
            label_to_id: Mapping from label strings to IDs
            max_length: Maximum sequence length
            pad_token_label_id: Label ID for padding tokens
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
        
        # Load examples from file
        self.examples = self._load_examples()
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
        
        # Convert to features
        self.features = self._convert_examples_to_features()
        logger.info(f"Converted to {len(self.features)} features")
    
    def _load_examples(self) -> List[InputExample]:
        """Load examples from CoNLL file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CoNLL file not found: {self.file_path}")
        
        examples = []
        current_tokens = []
        current_labels = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line == "":  # Empty line = end of sentence
                    if current_tokens:
                        # Reconstruct original text from SentencePiece tokens
                        text = self._reconstruct_text_from_sentencepiece(current_tokens)
                        examples.append(InputExample(
                            text_tokens=current_tokens.copy(),
                            labels=current_labels.copy(),
                            text=text
                        ))
                        current_tokens = []
                        current_labels = []
                
                elif not line.startswith("#"):  # Skip comments
                    parts = line.split()  # Use split() instead of split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_tokens.append(token)
                        current_labels.append(label)
        
        # Handle last sentence if file doesn't end with empty line
        if current_tokens:
            text = self._reconstruct_text_from_sentencepiece(current_tokens)
            examples.append(InputExample(
                text_tokens=current_tokens.copy(),
                labels=current_labels.copy(),
                text=text
            ))
        
        return examples
    
    def _reconstruct_text_from_sentencepiece(self, tokens: List[str]) -> str:
        """Reconstruct text from SentencePiece tokens."""
        text_parts = []
        for token in tokens:
            if token.startswith('▁'):  # SentencePiece space prefix
                text_parts.append(' ' + token[1:])  # Remove ▁ and add space
            else:
                text_parts.append(token)  # Continuation of previous word
        
        # Join and clean up spacing
        text = ''.join(text_parts).strip()
        return text
    
    def _convert_examples_to_features(self) -> List[InputFeatures]:
        """Convert examples to model features with tokenization alignment."""
        features = []
        
        for example in self.examples:
            feature = self._convert_single_example(example)
            if feature is not None:
                features.append(feature)
        
        return features
    
    def _convert_single_example(self, example: InputExample) -> Optional[InputFeatures]:
        """Convert a single example to features."""
        tokens = []
        labels = []
        subword_mask = []
        
        # Add [CLS] token
        tokens.append(self.tokenizer.cls_token)
        labels.append(self.pad_token_label_id)  # CLS gets ignored in loss
        subword_mask.append(False)  # CLS is not a real word
        
        # Process each token (already subword tokens from SentencePiece)
        for token, label in zip(example.text_tokens, example.labels):
            # Add the token directly (it's already a subword)
            tokens.append(token)
            
            # Determine if this is the start of a new word (▁ prefix)
            is_word_start = token.startswith('▁')
            subword_mask.append(is_word_start)
            
            # Add label
            if label in self.label_to_id:
                labels.append(self.label_to_id[label])
            else:
                logger.warning(f"Unknown label '{label}' in {self.file_path}")
                labels.append(self.label_to_id.get('O', 0))
        
        # Add [SEP] token
        tokens.append(self.tokenizer.sep_token)
        labels.append(self.pad_token_label_id)
        subword_mask.append(False)
        
        # Check length limit
        if len(tokens) > self.max_length:
            # Truncate (keep CLS and SEP)
            tokens = tokens[:self.max_length-1] + [self.tokenizer.sep_token]
            labels = labels[:self.max_length-1] + [self.pad_token_label_id]
            subword_mask = subword_mask[:self.max_length-1] + [False]
            logger.warning(f"Truncated sequence from {len(tokens)} to {self.max_length}")
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [self.pad_token_label_id] * padding_length
            subword_mask += [False] * padding_length
        
        return InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            original_tokens=example.text_tokens,
            subword_mask=subword_mask
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        return self.features[idx].to_tensor_dict()
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        label_counts = {}
        
        for feature in self.features:
            for label_id in feature.labels:
                if label_id != self.pad_token_label_id:
                    label = id_to_label.get(label_id, f"ID_{label_id}")
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts


def create_data_loaders(
    train_files: List[str],
    test_files: List[str],
    tokenizer: AutoTokenizer,
    label_to_id: Dict[str, int],
    batch_size: int = 32,
    max_length: int = 384,
    num_workers: int = 0,
    val_files: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        train_files: List of training CoNLL files
        test_files: List of test CoNLL files  
        tokenizer: XLM-RoBERTa tokenizer
        label_to_id: Label to ID mapping
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        val_files: Optional validation files
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader)
    """
    # Combine files into single datasets
    train_dataset = _create_combined_dataset(
        train_files, tokenizer, label_to_id, max_length
    )
    
    test_dataset = _create_combined_dataset(
        test_files, tokenizer, label_to_id, max_length
    )
    
    val_dataset = None
    if val_files:
        val_dataset = _create_combined_dataset(
            val_files, tokenizer, label_to_id, max_length
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=None  # Default collate works for our tensor dicts
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=None
        )
    
    # Print dataset info
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Test dataset: {len(test_dataset)} examples")
    if val_dataset:
        logger.info(f"Validation dataset: {len(val_dataset)} examples")
    
    return train_loader, test_loader, val_loader


def _create_combined_dataset(
    file_paths: List[str],
    tokenizer: AutoTokenizer, 
    label_to_id: Dict[str, int],
    max_length: int
) -> torch.utils.data.ConcatDataset:
    """Create a combined dataset from multiple CoNLL files."""
    datasets = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            dataset = CoNLLDataset(file_path, tokenizer, label_to_id, max_length)
            datasets.append(dataset)
            logger.info(f"Added {len(dataset)} examples from {file_path}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not datasets:
        raise ValueError(f"No valid files found in {file_paths}")
    
    return torch.utils.data.ConcatDataset(datasets)


def get_file_paths_for_experiment(
    experiment_config,
    data_dir: str = "datasets/prepared"
) -> Tuple[List[str], List[str]]:
    """
    Get file paths for training and testing based on experiment configuration.
    
    Args:
        experiment_config: ExperimentConfig instance
        data_dir: Base directory containing processed data
        
    Returns:
        Tuple of (train_files, test_files)
    """
    train_files = []
    test_files = []
    
    # Build training file paths
    for lang in experiment_config.train_languages:
        for domain in experiment_config.train_domains:
            file_path = os.path.join(
                data_dir, lang, "train", f"{domain}.conll"
            )
            train_files.append(file_path)
    
    # Build test file paths  
    for lang in experiment_config.test_languages:
        file_path = os.path.join(
            data_dir, lang, "test", "all_domains.conll"
        )
        if file_path not in test_files:  # Avoid duplicates
            test_files.append(file_path)
    
    return train_files, test_files


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from src.model.model import create_label_mapping
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Create label mapping
    label_to_id, id_to_label = create_label_mapping()
    
    # Test with a sample file (if it exists)
    sample_files = [
        "datasets/prepared/ru/train/it.conll",
        "datasets/prepared/ru/test/all_domains.conll"
    ]
    
    print("Testing data loader...")
    print(f"Label mapping: {label_to_id}")
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"\nTesting file: {file_path}")
            
            try:
                dataset = CoNLLDataset(
                    file_path=file_path,
                    tokenizer=tokenizer,
                    label_to_id=label_to_id,
                    max_length=384
                )
                
                print(f"  Examples: {len(dataset)}")
                
                # Show label distribution
                label_dist = dataset.get_label_distribution()
                print(f"  Label distribution: {label_dist}")
                
                # Test first batch
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
                    print(f"  Sample labels shape: {sample['labels'].shape}")
                    print(f"  Sample attention_mask shape: {sample['attention_mask'].shape}")
                
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print("\n✅ Data loader test completed!")
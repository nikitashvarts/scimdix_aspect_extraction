"""
Data preparation module for Scientific Aspect Extraction project.
Simple and straightforward implementation for converting raw CSV data to CoNLL format.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

from src.config import config, setup_directories


class DataPreparator:
    """Simple data preparation class for aspect extraction."""
    
    def __init__(self):
        """Initialize with XLM-RoBERTa tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        setup_directories()
    
    def parse_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file and return DataFrame."""
        return pd.read_csv(file_path)
    
    def extract_aspects_from_annotation(self, annotation: str) -> List[Tuple[str, str]]:
        """Extract aspects from annotation column using regex pattern [text|F1|CATEGORY]."""
        aspects = []
        pattern = config.ANNOTATION_PATTERN
        
        for match in re.finditer(pattern, annotation):
            text = match.group(1).strip()
            category = match.group(3).strip()
            aspects.append((text, category))
        
        return aspects
    
    def extract_aspects_from_column(self, aspects_text: str) -> List[Tuple[str, str]]:
        """Extract aspects from aspects column using pattern 'F1 CATEGORY text'."""
        aspects = []
        pattern = config.ASPECTS_PATTERN
        
        for line in aspects_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                category = match.group(2).strip()
                text = match.group(3).strip()
                aspects.append((text, category))
        
        return aspects
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using XLM-RoBERTa tokenizer."""
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def align_aspects_with_tokens(self, text: str, aspects: List[Tuple[str, str]]) -> List[str]:
        """Create BIO tags for tokenized text."""
        tokens = self.tokenize_text(text)
        bio_tags = ['O'] * len(tokens)
        
        # Simple word-based alignment
        text_lower = text.lower()
        
        for aspect_text, category in aspects:
            aspect_lower = aspect_text.lower()
            
            # Find aspect in original text
            start_idx = text_lower.find(aspect_lower)
            if start_idx == -1:
                continue
            
            # Find corresponding tokens
            char_to_token = self._create_char_to_token_mapping(text, tokens)
            
            aspect_start_token = char_to_token.get(start_idx)
            aspect_end_token = char_to_token.get(start_idx + len(aspect_text) - 1)
            
            if aspect_start_token is not None and aspect_end_token is not None:
                # Apply BIO tagging
                bio_tags[aspect_start_token] = f'B-{category}'
                for i in range(aspect_start_token + 1, aspect_end_token + 1):
                    if i < len(bio_tags):
                        bio_tags[i] = f'I-{category}'
        
        return tokens, bio_tags
    
    def _create_char_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """Create mapping from character positions to token indices."""
        char_to_token = {}
        current_pos = 0
        
        for token_idx, token in enumerate(tokens):
            # Remove special tokenizer symbols
            clean_token = token.replace('▁', ' ').replace('##', '')
            
            # Find token in text starting from current position
            token_start = text.lower().find(clean_token.lower(), current_pos)
            if token_start != -1:
                for char_pos in range(token_start, token_start + len(clean_token)):
                    char_to_token[char_pos] = token_idx
                current_pos = token_start + len(clean_token)
        
        return char_to_token
    
    def save_conll_file(self, tokens_and_tags: List[Tuple[List[str], List[str]]], output_path: Path):
        """Save tokens and BIO tags to CoNLL format file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for tokens, bio_tags in tokens_and_tags:
                for token, tag in zip(tokens, bio_tags):
                    f.write(f"{token} {tag}\n")
                f.write("\n")  # Empty line between sentences
    
    def process_domain_file(self, file_path: Path, language: str) -> List[Tuple[List[str], List[str]]]:
        """Process single domain CSV file and return tokenized data with BIO tags."""
        print(f"Processing {file_path}")
        
        df = self.parse_csv_file(file_path)
        tokens_and_tags = []
        
        for _, row in df.iterrows():
            text = row['abstract']
            annotation = row['aspect_annotation']
            aspects_column = row['aspects']
            
            # Extract aspects from both sources
            aspects1 = self.extract_aspects_from_annotation(annotation)
            aspects2 = self.extract_aspects_from_column(aspects_column)
            
            # Combine and deduplicate aspects
            all_aspects = list(set(aspects1 + aspects2))
            
            # Tokenize and align
            tokens, bio_tags = self.align_aspects_with_tokens(text, all_aspects)
            tokens_and_tags.append((tokens, bio_tags))
        
        return tokens_and_tags
    
    def process_language(self, language: str):
        """Process all files for one language."""
        print(f"\n=== Processing {language.upper()} language ===")
        
        # Process training files by domain
        domain_files = config.get_domain_files(language, "train")
        all_train_data = []
        
        for domain, file_path in domain_files.items():
            if file_path.exists():
                domain_data = self.process_domain_file(file_path, language)
                all_train_data.extend(domain_data)
                
                # Save individual domain file
                output_path = config.get_prepared_data_path(language) / config.TRAIN_SUBDIR / f"{domain}.conll"
                self.save_conll_file(domain_data, output_path)
                print(f"  Saved: {output_path}")
            else:
                print(f"  Warning: File not found - {file_path}")
        
        # Save combined training data
        if all_train_data:
            combined_output = config.get_prepared_data_path(language) / config.TRAIN_SUBDIR / "all_domains.conll"
            self.save_conll_file(all_train_data, combined_output)
            print(f"  Saved combined: {combined_output}")
        
        # Process test file
        test_files = config.get_domain_files(language, "test")
        for domain, file_path in test_files.items():
            if file_path.exists():
                test_data = self.process_domain_file(file_path, language)
                output_path = config.get_prepared_data_path(language) / config.TEST_SUBDIR / f"{domain}.conll"
                self.save_conll_file(test_data, output_path)
                print(f"  Saved: {output_path}")
            else:
                print(f"  Warning: File not found - {file_path}")
    
    def run(self):
        """Main method to process all languages."""
        print("=== Starting Data Preparation ===")
        print(f"Tokenizer: {config.TOKENIZER_NAME}")
        print(f"Output format: {config.OUTPUT_FORMAT}")
        
        for language in config.LANGUAGES:
            self.process_language(language)
        
        print("\n=== Data Preparation Complete ===")


def main():
    """Entry point for data preparation."""
    preparator = DataPreparator()
    preparator.run()


if __name__ == "__main__":
    main()
"""
Data preparation module for Scientific Aspect Extraction project.
Simple and straightforward implementation for converting raw CSV data to CoNLL format.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple
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
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex rules."""
        # Simple sentence splitting for Russian and Kazakh
        # Split on sentence-ending punctuation followed by space and capital letter
        # Using explicit character classes instead of ranges to avoid Unicode issues
        sentences = re.split(r'[.!?]+\s+(?=[А-ЯЁӘІҢҒҮҰҚӨҺA-Z])', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                # Remove trailing punctuation for consistency
                sentence = re.sub(r'[.!?]+$', '', sentence)
                clean_sentences.append(sentence)
        
        return clean_sentences
    
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
    
    def align_aspects_with_sentences(self, sentences: List[str], aspects: List[Tuple[str, str]]) -> List[Tuple[List[str], List[str]]]:
        """Create BIO tags for each sentence separately."""
        sentence_data = []
        
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            bio_tags = ['O'] * len(tokens)
            
            # Check token count
            if len(tokens) > config.MAX_SEQUENCE_LENGTH:
                print(f"  Warning: Sentence has {len(tokens)} tokens (>{config.MAX_SEQUENCE_LENGTH}), truncating")
                tokens = tokens[:config.MAX_SEQUENCE_LENGTH]
                bio_tags = bio_tags[:config.MAX_SEQUENCE_LENGTH]
            
            # Find aspects that appear in this sentence
            sentence_lower = sentence.lower()
            
            for aspect_text, category in aspects:
                aspect_text = aspect_text.strip().lower()
                if not aspect_text:
                    continue
                
                # Check if aspect appears in this sentence
                if aspect_text in sentence_lower:
                    # Find position using tokenizer's text reconstruction
                    try:
                        reconstructed_text = self.tokenizer.convert_tokens_to_string(tokens).lower()
                        
                        # Find aspect in reconstructed text
                        aspect_start = reconstructed_text.find(aspect_text)
                        if aspect_start == -1:
                            continue
                        
                        # Map to token positions
                        char_pos = 0
                        for i, token in enumerate(tokens):
                            token_text = token.replace('▁', ' ').strip()
                            if not token_text:
                                continue
                            
                            token_start = char_pos
                            token_end = char_pos + len(token_text)
                            
                            # Check if token overlaps with aspect
                            aspect_end = aspect_start + len(aspect_text)
                            if not (token_end <= aspect_start or token_start >= aspect_end):
                                if bio_tags[i] == 'O':  # Don't overwrite existing tags
                                    if token_start <= aspect_start < token_end:
                                        bio_tags[i] = f'B-{category}'
                                    else:
                                        bio_tags[i] = f'I-{category}'
                            
                            char_pos = token_end + 1  # +1 for potential space
                            
                    except Exception:
                        # Fallback to simple word matching
                        continue
            
            sentence_data.append((tokens, bio_tags))
        
        return sentence_data
    
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
        all_sentence_data = []
        
        for doc_idx, row in df.iterrows():
            text = row['abstract']
            aspects_column = row['aspects']
            
            # Extract aspects only from aspects column
            aspects = self.extract_aspects_from_column(aspects_column)
            
            if not aspects:
                print("  Warning: No aspects found in row")
                continue
            
            # Split text into sentences
            sentences = self.split_into_sentences(text)
            
            # Process each sentence separately
            sentence_data = self.align_aspects_with_sentences(sentences, aspects)
            all_sentence_data.extend(sentence_data)
            
            # Debug info for first few documents
            if doc_idx < 3:
                print(f"  Doc {doc_idx}: {len(sentences)} sentences, {len(aspects)} aspects: {[cat for _, cat in aspects]}")
                avg_tokens = sum(len(tokens) for tokens, _ in sentence_data) / len(sentence_data) if sentence_data else 0
                print(f"    Average tokens per sentence: {avg_tokens:.1f}")
        
        print(f"  Processed {len(df)} documents → {len(all_sentence_data)} sentences")
        return all_sentence_data
    
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
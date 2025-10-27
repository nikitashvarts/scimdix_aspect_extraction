"""
Statistics calculation module for Scientific Aspect Extraction project.
Analyzes prepared data and generates comprehensive statistics.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

from src.config import config, PREPARED_DATA_DIR


class StatisticsCalculator:
    """Simple statistics calculator for aspect extraction data."""
    
    def __init__(self):
        """Initialize statistics calculator."""
        self.stats = {}
    
    def read_conll_file(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        """Read CoNLL file and return list of (tokens, tags) tuples."""
        sentences = []
        current_tokens = []
        current_tags = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Empty line - end of sentence
                    if current_tokens and current_tags:
                        sentences.append((current_tokens, current_tags))
                        current_tokens = []
                        current_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[1]
                        current_tokens.append(token)
                        current_tags.append(tag)
        
        # Add last sentence if exists
        if current_tokens and current_tags:
            sentences.append((current_tokens, current_tags))
        
        return sentences
    
    def count_sentences_and_tokens(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, int]:
        """Count sentences and tokens."""
        total_sentences = len(sentences)
        total_tokens = sum(len(tokens) for tokens, _ in sentences)
        
        return {
            'sentences': total_sentences,
            'tokens': total_tokens,
            'avg_tokens_per_sentence': total_tokens / total_sentences if total_sentences > 0 else 0
        }
    
    def count_aspects(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, int]:
        """Count aspects by category."""
        aspect_counts = Counter()
        aspect_token_counts = Counter()
        
        for tokens, tags in sentences:
            current_aspect = None
            current_length = 0
            
            for tag in tags:
                if tag.startswith('B-'):
                    # Finish previous aspect
                    if current_aspect:
                        aspect_counts[current_aspect] += 1
                        aspect_token_counts[current_aspect] += current_length
                    
                    # Start new aspect
                    current_aspect = tag[2:]  # Remove 'B-'
                    current_length = 1
                    
                elif tag.startswith('I-') and current_aspect:
                    current_length += 1
                    
                elif tag == 'O':
                    # Finish previous aspect
                    if current_aspect:
                        aspect_counts[current_aspect] += 1
                        aspect_token_counts[current_aspect] += current_length
                        current_aspect = None
                        current_length = 0
            
            # Finish last aspect in sentence
            if current_aspect:
                aspect_counts[current_aspect] += 1
                aspect_token_counts[current_aspect] += current_length
        
        # Calculate average lengths
        aspect_avg_lengths = {}
        for category in aspect_counts:
            if aspect_counts[category] > 0:
                aspect_avg_lengths[category] = aspect_token_counts[category] / aspect_counts[category]
        
        return {
            'aspect_counts': dict(aspect_counts),
            'aspect_token_counts': dict(aspect_token_counts),
            'aspect_avg_lengths': aspect_avg_lengths,
            'total_aspects': sum(aspect_counts.values())
        }
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze single CoNLL file."""
        if not file_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        print(f"Analyzing: {file_path}")
        
        sentences = self.read_conll_file(file_path)
        
        # Basic counts
        basic_stats = self.count_sentences_and_tokens(sentences)
        
        # Aspect statistics
        aspect_stats = self.count_aspects(sentences)
        
        return {
            'file_path': str(file_path),
            'basic_stats': basic_stats,
            'aspect_stats': aspect_stats
        }
    
    def analyze_language(self, language: str) -> Dict:
        """Analyze all files for one language."""
        print(f"\n=== Analyzing {language.upper()} language ===")
        
        lang_stats = {
            'language': language,
            'train': {},
            'test': {},
            'summary': {}
        }
        
        # Analyze training files
        train_path = config.get_prepared_data_path(language) / config.TRAIN_SUBDIR
        if train_path.exists():
            for domain in config.DOMAINS:
                domain_file = train_path / f"{domain}.conll"
                if domain_file.exists():
                    lang_stats['train'][domain] = self.analyze_file(domain_file)
            
            # Analyze combined file
            combined_file = train_path / "all_domains.conll"
            if combined_file.exists():
                lang_stats['train']['all_domains'] = self.analyze_file(combined_file)
        
        # Analyze test files
        test_path = config.get_prepared_data_path(language) / config.TEST_SUBDIR
        if test_path.exists():
            test_file = test_path / "all_domains.conll"
            if test_file.exists():
                lang_stats['test']['all_domains'] = self.analyze_file(test_file)
        
        # Calculate summary statistics
        lang_stats['summary'] = self.calculate_language_summary(lang_stats)
        
        return lang_stats
    
    def calculate_language_summary(self, lang_stats: Dict) -> Dict:
        """Calculate summary statistics for a language."""
        summary = {
            'total_train_sentences': 0,
            'total_train_tokens': 0,
            'total_train_aspects': 0,
            'total_test_sentences': 0,
            'total_test_tokens': 0,
            'total_test_aspects': 0,
            'aspect_distribution': Counter()
        }
        
        # Sum training stats
        for domain_stats in lang_stats['train'].values():
            if 'error' not in domain_stats:
                summary['total_train_sentences'] += domain_stats['basic_stats']['sentences']
                summary['total_train_tokens'] += domain_stats['basic_stats']['tokens']
                summary['total_train_aspects'] += domain_stats['aspect_stats']['total_aspects']
                
                for category, count in domain_stats['aspect_stats']['aspect_counts'].items():
                    summary['aspect_distribution'][category] += count
        
        # Sum test stats
        for domain_stats in lang_stats['test'].values():
            if 'error' not in domain_stats:
                summary['total_test_sentences'] += domain_stats['basic_stats']['sentences']
                summary['total_test_tokens'] += domain_stats['basic_stats']['tokens']
                summary['total_test_aspects'] += domain_stats['aspect_stats']['total_aspects']
        
        # Convert Counter to dict for JSON serialization
        summary['aspect_distribution'] = dict(summary['aspect_distribution'])
        
        return summary
    
    def calculate_overall_summary(self, all_stats: Dict) -> Dict:
        """Calculate overall project statistics."""
        overall = {
            'total_languages': len(config.LANGUAGES),
            'total_domains': len(config.DOMAINS),
            'tokenizer_used': config.TOKENIZER_NAME,
            'aspect_categories': config.ASPECT_CATEGORIES,
            'languages': {},
            'cross_language_comparison': {}
        }
        
        # Collect stats per language
        for lang, lang_stats in all_stats.items():
            if lang != 'overall':
                overall['languages'][lang] = lang_stats['summary']
        
        # Cross-language comparison
        if len(overall['languages']) > 1:
            for category in config.ASPECT_CATEGORIES:
                overall['cross_language_comparison'][category] = {}
                for lang in overall['languages']:
                    count = overall['languages'][lang]['aspect_distribution'].get(category, 0)
                    overall['cross_language_comparison'][category][lang] = count
        
        return overall
    
    def save_statistics(self, output_path: Path):
        """Save statistics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nStatistics saved to: {output_path}")
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("DATASET STATISTICS SUMMARY")
        print("="*60)
        
        overall = self.stats.get('overall', {})
        
        print(f"Tokenizer: {overall.get('tokenizer_used', 'N/A')}")
        print(f"Languages: {overall.get('total_languages', 0)}")
        print(f"Domains: {overall.get('total_domains', 0)}")
        print(f"Aspect Categories: {len(overall.get('aspect_categories', []))}")
        
        print("\nPer Language Statistics:")
        for lang, lang_data in overall.get('languages', {}).items():
            print(f"\n{lang.upper()}:")
            print(f"  Train: {lang_data['total_train_sentences']} sentences, "
                  f"{lang_data['total_train_tokens']} tokens, "
                  f"{lang_data['total_train_aspects']} aspects")
            print(f"  Test:  {lang_data['total_test_sentences']} sentences, "
                  f"{lang_data['total_test_tokens']} tokens, "
                  f"{lang_data['total_test_aspects']} aspects")
            
            print("  Aspect distribution:")
            for category, count in lang_data['aspect_distribution'].items():
                print(f"    {category}: {count}")
        
        print("\n" + "="*60)
    
    def run(self):
        """Main method to calculate all statistics."""
        print("=== Starting Statistics Calculation ===")
        
        self.stats = {}
        
        # Analyze each language
        for language in config.LANGUAGES:
            self.stats[language] = self.analyze_language(language)
        
        # Calculate overall statistics
        self.stats['overall'] = self.calculate_overall_summary(self.stats)
        
        # Save statistics
        output_path = PREPARED_DATA_DIR / config.STATISTICS_FILE
        self.save_statistics(output_path)
        
        # Print summary
        self.print_summary()
        
        print("\n=== Statistics Calculation Complete ===")


def main():
    """Entry point for statistics calculation."""
    calculator = StatisticsCalculator()
    calculator.run()


if __name__ == "__main__":
    main()
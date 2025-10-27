"""
Configuration file for the Scientific Aspect Extraction project.
Contains all configuration parameters for data preparation, training, and evaluation.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
PREPARED_DATA_DIR = PROJECT_ROOT / "datasets" / "prepared"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data preparation configuration
class DataConfig:
    """Configuration for data preparation module."""
    
    # Tokenizer settings
    TOKENIZER_NAME = "xlm-roberta-base"
    
    # Supported languages
    LANGUAGES = ["ru", "kz"]
    
    # Supported domains
    DOMAINS = ["it", "ling", "med", "psy"]
    
    # Aspect categories (extracted from data analysis)
    ASPECT_CATEGORIES = [
        "AIM",      # Research aims/objectives
        "METHOD",   # Research methods/approaches
        "MATERIAL", # Research materials/data
        "TASK",     # Specific tasks/goals
        "TOOL",     # Tools/instruments/software
        "RESULT",   # Research results/findings
        "USAGE"     # Applications/practical use
    ]
    
    # BIO tagging scheme
    BIO_SCHEME = {
        "O": "O",  # Outside
        "B": "B-", # Beginning
        "I": "I-"  # Inside
    }
    
    # File extensions
    OUTPUT_FORMAT = "conll"  # .conll files
    
    # Data structure configuration
    TRAIN_SUBDIR = "train"
    TEST_SUBDIR = "test"
    
    # Statistics output
    STATISTICS_FILE = "statistics.json"
    
    # Parsing configuration
    ANNOTATION_PATTERN = r'\[([^|]+)\|([^|]+)\|([^|]+)\]'  # Regex for [text|F1|CATEGORY]
    ASPECTS_PATTERN = r'^(F\d+)\s+(\w+)\s+(.+)$'  # Regex for "F3 TASK text"
    
    # Validation settings
    MAX_SEQUENCE_LENGTH = 512  # Maximum tokens per sequence
    MIN_ASPECT_LENGTH = 1      # Minimum tokens in aspect
    
    @classmethod
    def get_raw_data_path(cls, language: str) -> Path:
        """Get path to raw data for specific language."""
        return RAW_DATA_DIR / language
    
    @classmethod
    def get_prepared_data_path(cls, language: str) -> Path:
        """Get path to prepared data for specific language."""
        return PREPARED_DATA_DIR / language
    
    @classmethod
    def get_domain_files(cls, language: str, split: str = "train") -> dict:
        """Get mapping of domain -> file path for specific language and split."""
        base_path = cls.get_raw_data_path(language) / split
        files = {}
        
        if split == "train":
            for domain in cls.DOMAINS:
                files[domain] = base_path / f"train-{domain}-{language}-aspects.csv"
        else:  # test
            files["all_domains"] = base_path / f"testset-{language}-aspects.csv"
            
        return files

# Global configuration instance
config = DataConfig()

# Create necessary directories
def setup_directories():
    """Create all necessary directories for the project."""
    directories = [
        PREPARED_DATA_DIR,
        RESULTS_DIR,
    ]
    
    for lang in config.LANGUAGES:
        directories.extend([
            config.get_prepared_data_path(lang),
            config.get_prepared_data_path(lang) / config.TRAIN_SUBDIR,
            config.get_prepared_data_path(lang) / config.TEST_SUBDIR,
        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_directories()
    print("Project directories created successfully!")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Prepared data directory: {PREPARED_DATA_DIR}")
    print(f"Tokenizer: {config.TOKENIZER_NAME}")
    print(f"Supported languages: {config.LANGUAGES}")
    print(f"Supported domains: {config.DOMAINS}")
    print(f"Aspect categories: {config.ASPECT_CATEGORIES}")
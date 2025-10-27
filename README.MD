# SciMDIX Aspect Extraction

Machine Learning project for aspect extraction in Russian and Kazakh scientific texts.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download Stanza models for sentence segmentation:
```bash
python scripts/download_stanza_models.py
```

## Usage

### Data Preparation
Convert raw CSV data to CoNLL format for training:

```bash
python scripts/prepare_data.py
```

### Testing
Run tests to verify functionality:

```bash
PYTHONPATH=. python tests/test_data_preparation.py
```

## Features

- **Multilingual Support**: Russian and Kazakh languages
- **Advanced Sentence Segmentation**: Using Stanza NLP library
- **XLM-RoBERTa Tokenization**: For consistent multilingual processing  
- **BIO Tagging**: Standard format for aspect extraction
- **Domain Coverage**: IT, linguistics, medicine, psychology
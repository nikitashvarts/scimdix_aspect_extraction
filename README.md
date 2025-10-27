# 🔬 Aspect Extraction for Scientific Texts

**Multilingual Aspect Extraction with Transfer Learning: Russian → Kazakh**

This project implements XLM-RoBERTa + CRF model for aspect extraction in scientific texts, with focus on zero-shot transfer learning from Russian to Kazakh.

## 🎯 Research Goal

Transfer learning approach to extract scientific aspects (AIM, METHOD, MATERIAL, TASK, TOOL, RESULT, USAGE) from Kazakh scientific texts using Russian training data.

## 🚀 Quick Start

### For GPU Server (Recommended)

```bash
# Clone and setup
git clone <repository>
cd scimdix_aspect_extraction
chmod +x scripts/*.sh

# Quick test (1 epoch)
scripts/run_single_experiment.sh test

# Main zero-shot experiment
scripts/run_single_experiment.sh zero_shot_ru_to_kz

# Run all experiments
scripts/run_all_experiments.sh
```

### For Local Development

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Test locally
PYTHONPATH=. python scripts/test_cpu_training.py

# Run experiment natively
scripts/run_native.sh test
```

## 📊 Experiments

- **`baseline_ru`** - Russian baseline (ru→ru)
- **`baseline_kz`** - Kazakh baseline (kz→kz) 
- **`zero_shot_ru_to_kz`** - Main experiment: Russian→Kazakh transfer
- **`lodo_*`** - Leave-One-Domain-Out validation (IT, linguistics, medical, psychology)

## 🏗️ Architecture

- **Model**: XLM-RoBERTa-base + Linear + CRF
- **Training**: Dual learning rates (encoder: 2e-5, head+CRF: 1e-4)
- **Evaluation**: Span-level precision, recall, F1 with exact matching
- **Multi-seed**: [13, 21, 42] for statistical significance

## 📁 Project Structure

```
├── src/                    # Core source code
│   ├── data/              # Data preparation pipeline
│   └── model/             # Model, trainer, evaluator
├── scripts/               # Execution scripts
│   ├── run_single_experiment.sh
│   ├── run_all_experiments.sh
│   ├── run_native.sh
│   └── test_cpu_training.py
├── docker/                # Docker configurations
│   ├── Dockerfile
│   ├── Dockerfile.simple
│   └── docker-compose.yml
├── docs/                  # Documentation
├── datasets/              # Training and test data
│   ├── raw/              # Original CSV files
│   └── prepared/         # Processed CoNLL files
└── results/              # Experiment outputs
    ├── experiments/      # Metrics and logs
    ├── models/          # Saved model weights
    └── logs/            # Training logs
```

## 🔧 Configuration

Key parameters in `src/model/config.py`:
- Batch size: 32 (GPU) / 1-2 (CPU)
- Max sequence length: 384 tokens
- Epochs: 20 for full training
- Early stopping: 3 epochs patience
- GPU: Configured for GPU=1 by default

## 📈 Expected Results

The system evaluates on 7 aspect classes with span-level metrics:
- **Micro-averaged F1**: Overall performance
- **Macro-averaged F1**: Performance across all classes
- **Per-class metrics**: Precision, recall, F1 for each aspect
- **Confusion matrix**: Error analysis

## 📚 Documentation

- [Docker GPU Guide](docs/DOCKER_GUIDE.md) - Detailed deployment instructions
- [Data Preparation](src/data/README.md) - Data processing pipeline
- [Model Architecture](src/model/README.md) - Technical details

## 🛠️ Troubleshooting

### Docker Issues
```bash
# Try simple Dockerfile
docker build -f docker/Dockerfile.simple -t aspect-extraction .

# Or run natively
scripts/run_native.sh test
```

### GPU Configuration
```bash
# Check GPU availability
nvidia-smi

# Set specific GPU
export CUDA_VISIBLE_DEVICES=1
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{scimdix_aspect_extraction,
  title={Multilingual Aspect Extraction for Scientific Texts: Russian-Kazakh Transfer Learning},
  author={Your Name},
  year={2025},
  note={Research implementation for scientific text analysis}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📧 Contact

For questions about the research or implementation, please open an issue or contact us.

---

**Status**: ✅ Ready for GPU training | 🧪 Tested on RTX 3090 | 🐳 Docker deployed
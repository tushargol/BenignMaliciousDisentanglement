# Benign vs. Malicious Anomaly Disentanglement (Sherlock / 02-Semiurban)

An advanced Intrusion Detection System (IDS) that intelligently distinguishes between **malicious attacks** and **benign operational anomalies** in industrial control systems using the Sherlock 02-Semiurban dataset.

## Project Overview

This project addresses a critical challenge in industrial cybersecurity: reducing false alarms caused by legitimate operational activities (maintenance, switching operations) while maintaining high detection rates for actual malicious attacks. The system achieves this through a novel two-stage machine learning pipeline.

## Architecture

### Two-Stage Pipeline
1. **Stage 1 - Autoencoder**: Trained on normal + benign data to detect any anomalies
2. **Stage 2 - Classifier**: Distinguishes between benign and malicious anomalies

### Key Features
- **Event-derived time-series** processing with configurable windowing
- **Per-attack-type analysis** for comprehensive security assessment
- **SHAP interpretability** for model transparency
- **Auto-tuning** of optimal thresholds
- **Baseline comparison** with Isolation Forest

## Performance Results

### Overall Metrics
- **Normal Traffic**: F1=0.966, Precision=0.946, Recall=0.987
- **Benign Anomalies**: F1=0.584, Precision=0.536, Recall=0.642  
- **Malicious Attacks**: F1=0.931, Precision=0.998, Recall=0.873

### Security Performance
- **Malicious Recall**: 87.3% (high attack detection rate)
- **False Positive Rate on Normal**: 0.07% (minimal false alarms)
- **False Positive Rate on Benign**: 0% (perfect benign suppression)

### Per-Attack Detection Rates
- **drift-off**: 99.7% recall
- **control-and-freeze**: 98.6% recall  
- **arp-spoof**: 86.2% recall
- **industroyer**: 44.3% recall (challenging attack type)

## Quick Start

### Prerequisites
- Python 3.8+
- Sherlock 02-Semiurban dataset

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BenignVsMalicious

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

This project uses the **Sherlock 02-Semiurban Dataset** for industrial control system anomaly detection.

**Dataset Source**: https://zenodo.org/records/18467070

**Citation**: If you use this dataset in your research, please cite the original authors as specified in the Zenodo record.

**Download Instructions**:
1. Visit the Zenodo repository linked above
2. Download the `02-Semiurban` dataset
3. Extract and place the dataset folder at:
   - `data/02-Semiurban/` (recommended) or `02-Semiurban/` at repo root

Expected structure:
```
02-Semiurban/
├── ipal/
│   ├── train/events.json
│   └── test/events.json
└── raw/
    ├── train/events.jsonl
    └── test/events.jsonl
```

**Note**: The dataset directories are excluded from git via `.gitignore` to respect file size limits and licensing terms.

## Usage

### Basic Usage

```bash
# Run complete pipeline
python run_pipeline.py --stage all

# Run individual stages
python run_pipeline.py --stage prepare        # Data preparation
python run_pipeline.py --stage train_ae       # Train autoencoder
python run_pipeline.py --stage train_classifier # Train classifier
python run_pipeline.py --stage evaluate       # Evaluate performance
```

### Advanced Options

```bash
# Auto-tune thresholds for optimal performance
python run_pipeline.py --stage all --auto_tune

# Ablation studies
python run_pipeline.py --stage all --ablation stage1_only    # AE-only
python run_pipeline.py --stage all --ablation no_classifier  # No stage-2

# Custom thresholds
python run_pipeline.py --stage all --ae_percentile 85 --clf_threshold 0.6

# Constrain false positive rates
python run_pipeline.py --stage all --auto_tune --max_fpr_normal 0.01 --max_fpr_benign 0.03
```

## Project Structure

```
BenignVsMalicious/
├── src/
│   ├── data/          # Event loading, time-series processing, windowing
│   ├── models/        # Autoencoder, classifier, baseline models
│   ├── training/      # Training loops and utilities
│   ├── evaluation/    # Metrics, per-attack analysis, SHAP
│   ├── features/      # Feature extraction and engineering
│   └── utils/         # Helper utilities
├── experiments/
│   └── example.yaml   # Configuration file
├── outputs/           # Generated models, results, reports
├── 02-Semiurban/      # Dataset (or data/02-Semiurban/)
├── run_pipeline.py    # Main pipeline script
├── requirements.txt   # Python dependencies
└── README.md
```

## Outputs

After running the pipeline, you'll find:

- `outputs/prepared.pkl` – Processed datasets
- `outputs/models/` – Trained models (autoencoder, classifier, baseline)
- `outputs/eval_report.json` – Comprehensive performance metrics
- `outputs/shap_top_features.json` – Feature importance analysis
- `outputs/auto_tune_results.csv` – Threshold optimization results

## Configuration

Edit `experiments/example.yaml` to customize:

```yaml
scenario: "02-Semiurban"
window_size_s: 90          # Time window size
stride_s: 20               # Window stride
resample_step_s: 1         # Resampling frequency

stage1:                    # Autoencoder config
  hidden_dim: 256
  latent_dim: 64
  epochs: 25

stage2:                    # Classifier config
  hidden_dim: 128
  epochs: 30
  attack_family_weights:   # Emphasize hard attacks
    arp-spoof: 4.0
    industroyer: 4.0
```

## Model Interpretability

The system provides SHAP-based explanations for classifier decisions:

```python
# Top contributing features (example)
f41: 0.0117    # Most important feature
f83: 0.0116    # Second most important
f13: 0.0106    # Third most important
```

## Ablation Studies

Compare different approaches:

```bash
# AE-only vs Two-stage
python run_pipeline.py --stage all --ablation stage1_only

# With/without classifier
python run_pipeline.py --stage all --ablation no_classifier
```

## Requirements

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `torch` - Deep learning framework
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `shap` - Model explainability
- `pyyaml` - Configuration parsing
- `tqdm` - Progress bars

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of academic research on industrial control system security.

## Research Applications

This research has applications in:
- **Industrial Control System Security**
- **Critical Infrastructure Protection**
- **Anomaly Detection in IoT Networks**
- **False Alarm Reduction in Security Operations**

## Key Research Contributions

- **Novel two-stage architecture** for benign vs malicious disentanglement
- **Comprehensive evaluation** on real-world industrial dataset
- **Interpretability framework** using SHAP for model transparency
- **Auto-tuning methodology** for optimal threshold selection
- **Per-attack analysis** for detailed security assessment

## Acknowledgments

This research uses the **Sherlock 02-Semiurban Dataset** provided by the original researchers and hosted on Zenodo. We acknowledge their contribution to the industrial control system security community for making this valuable dataset publicly available.

The dataset enables research in anomaly detection for industrial environments and contains realistic attack scenarios critical for advancing cybersecurity in critical infrastructure.

For full dataset details, licensing terms, and citation information, please visit: https://zenodo.org/records/18467070


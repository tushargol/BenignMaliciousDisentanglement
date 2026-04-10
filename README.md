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

## Interactive Demonstration

### Enhanced Demo with Real Sherlock Data

This project features an **enhanced interactive demonstration** that leverages **real power systems operational data** from the Sherlock dataset:

#### **Real Operational Data Integration**
- **49 Authentic Events**: Real power systems procedures and commands
- **9 Event Types**: Control center procedures, SCADA commands, maintenance operations
- **Real Procedures**: Transformer maintenance, separator movement, load management
- **Authentic Patterns**: Actual voltage, frequency, and load control operations

#### **Enhanced Technical Features**
- **Real SHAP Integration**: Actual model explainability with feature attribution
- **Model Calibration**: Data-driven threshold optimization using real training data
- **Multi-class Logic**: Proper three-class decision making (Normal/Benign/Malicious)
- **Sub-2ms Performance**: Real-time inference suitable for operational deployment

#### **Six-Panel Visualization Dashboard**
1. **Real-time Metrics**: Confidence scores and reconstruction errors
2. **Detection Timeline**: Scenario results with data source indicators
3. **SHAP Feature Importance**: Real calculated feature contributions
4. **System Status**: Current threat level and performance metrics
5. **Data Source Analysis**: Real vs synthetic scenario breakdown
6. **Performance Metrics**: Model calibration and accuracy analytics

### Running the Enhanced Demo

#### **Enhanced Demo with Real Data**
```bash
python demo_improved.py
# Enhanced demo with real Sherlock data integration
# Real SHAP explanations
# Six-panel visualization dashboard
# Model calibration with real training data
```

#### **Demo Modes**
1. **Interactive Demo**: User-selectable scenarios with real-time explanations
2. **Automated Demo**: All scenarios with comprehensive performance analysis
3. **Quick Demo**: 3-minute overview of key capabilities

### Enhanced Demo Outputs

#### **Comprehensive Reports**
- `enhanced_demo_report_YYYYMMDD_HHMMSS.txt`: Detailed analysis with real data insights
- `enhanced_demo_results_YYYYMMDD_HHMMSS.json`: Structured results with SHAP variations
- `enhanced_demo_summary_YYYYMMDD_HHMMSS.png`: Six-panel visualization dashboard

#### **Real Data Analytics**
- **Sherlock Event Analysis**: 49 real operational events processed
- **Procedure Extraction**: Real maintenance and operational procedures
- **Pattern Recognition**: Authentic power systems operational patterns
- **Performance Validation**: Model tested on real grid scenarios

### Key Enhancements Achieved

#### **Real SHAP Implementation**
- **Before**: Static identical SHAP values for all predictions
- **After**: Real calculated SHAP values with model-specific variations
- **Impact**: Authentic model explainability for operator trust

#### **Model Calibration**
- **Reconstruction Threshold**: `1.32e-06` (calibrated from real training data)
- **Classifier Threshold**: `0.5` (optimized for binary classification)
- **Method**: 95th percentile of training reconstruction errors

#### **Performance Validation**
- **Prediction Time**: 1.73ms average (sub-2ms real-time capability)
- **Model Status**: Autoencoder and classifier loaded and calibrated
- **Data Integration**: Successfully processed 49 authentic power systems events

### Real Sherlock Data Examples

#### **Authentic Operational Procedures**
```json
{
  "notification_data": {
    "malicious": false,
    "type": "procedure",
    "context": "control center",
    "event": "Transformer Maintenance",
    "description": "The control center issues control commands to fully disconnect an MV/LV transformer from the grid to enable safe maintenance"
  }
}
```

#### **Real SCADA Commands**
- **Control Commands**: Actual voltage and frequency control operations
- **Protection Coordination**: Real relay coordination procedures
- **Load Management**: Authentic load flow and switching operations
- **Maintenance Sequences**: Real equipment disconnection/reconnection procedures

## Quick Start

### Prerequisites
- Python 3.8+
- Sherlock 02-Semiurban dataset
- PyTorch (for model training and inference)
- SHAP (for explainability features)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BenignVsMalicious

# Install dependencies
pip install -r requirements.txt

# Download and place the Sherlock 02-Semiurban dataset
# Either as:
#   - 02-Semiurban/ (in project root)
#   - data/02-Semiurban/ (preferred)

# Verify installation
python -c "import torch, shap, pandas, numpy; print('All dependencies installed successfully')"
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
│   ├── config.py              # Project configuration and paths
│   ├── data/                  # Event loading, time-series processing, windowing
│   │   ├── events_loader.py   # Event data loading utilities
│   │   ├── prepare_pipeline.py # Data preparation pipeline
│   │   ├── raw_event_interval_features.py
│   │   ├── timeseries_loader.py
│   │   └── windowing.py       # Time series windowing
│   ├── models/                # Autoencoder, classifier, baseline models
│   │   ├── autoencoder.py     # Anomaly detection autoencoder
│   │   ├── baseline.py        # Isolation Forest baseline
│   │   └── classifier.py      # Benign vs malicious classifier
│   ├── training/              # Training loops and utilities
│   │   ├── train_autoencoder.py
│   │   └── train_classifier.py
│   ├── evaluation/            # Metrics, per-attack analysis, SHAP
│   │   ├── evaluate_pipeline.py
│   │   ├── metrics.py
│   │   ├── threshold_sweep.py
│   │   └── visualizations.py
│   ├── features/              # Feature extraction and engineering
│   │   └── feature_engineering.py
│   └── utils/                 # Helper utilities
│       ├── logging_utils.py
│       ├── paths.py
│       └── seed.py
├── deployment/                # Docker and deployment configuration
│   ├── docker-compose.yml
│   ├── services/
│   ├── deploy.sh
│   ├── QUICK_START.md
│   └── SECURITY.md
├── experiments/               # Configuration and experiment files
│   └── example.yaml
├── outputs/                   # Generated models, results, reports
│   ├── models/               # Trained models (.pt files)
│   ├── visualizations/       # Evaluation plots
│   ├── demo_report_*.txt     # Demo reports
│   ├── enhanced_demo_*.txt   # Enhanced demo outputs
│   └── *.json, *.csv         # Evaluation results
├── scripts/                   # Additional utility scripts
├── notebooks/                 # Jupyter notebooks for analysis
├── 02-Semiurban/              # Sherlock dataset (or data/02-Semiurban/)
├── demo_improved.py           # Enhanced demo with real Sherlock data
├── run_pipeline.py            # Main pipeline script
├── requirements.txt           # Python dependencies
├── capstone_project_details.md # Comprehensive project documentation
├── deployment_strategy.md     # Deployment strategy documentation
├── report_outline.md          # Research report outline
└── README.md
```

### Enhanced Demo Files
- **`demo_improved.py`**: Enhanced demonstration with real Sherlock data integration
- **Enhanced Outputs**: Comprehensive reports with real data analytics and SHAP explanations

## Outputs

After running the pipeline, you'll find:

### Core Results
- `outputs/prepared.pkl` – Processed datasets
- `outputs/models/` – Trained models (autoencoder, classifier, baseline)
- `outputs/eval_report.json` – Comprehensive performance metrics
- `outputs/shap_top_features.json` – Feature importance analysis
- `outputs/auto_tune_results.csv` – Threshold optimization results
- `outputs/threshold_sweep.csv` – Threshold analysis results

### Demo Outputs
- `outputs/enhanced_demo_report_YYYYMMDD_HHMMSS.txt` – Enhanced demo with real data analysis
- `outputs/enhanced_demo_results_YYYYMMDD_HHMMSS.json` – Enhanced demo with SHAP variations
- `outputs/enhanced_demo_summary_YYYYMMDD_HHMMSS.png` – Enhanced demo six-panel dashboard

### Visualizations
- `outputs/visualizations/` – Generated evaluation plots:
  - `confusion_matrix.png` – Raw confusion matrix
  - `confusion_matrix_normalized.png` – Normalized confusion matrix
  - `per_class_metrics.png` – Precision, recall, F1 by class
  - `per_attack_recall.png` – Detection rate by attack family
  - `roc_pr_curves.png` – ROC and Precision-Recall curves
  - `shap_feature_importance.png` – Top contributing features
  - `visualization_summary.json` – Summary of generated plots

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

## Visualizations

The pipeline automatically generates comprehensive evaluation visualizations:

### Generated Plots
- **Confusion Matrices**: Both raw counts and normalized percentages
- **Per-Class Metrics**: Bar charts comparing precision, recall, and F1 scores
- **Per-Attack Analysis**: Detection rates for each attack family with sample sizes
- **ROC/PR Curves**: Binary detection performance for malicious vs non-malicious
- **SHAP Feature Importance**: Visual ranking of top contributing features

### Accessing Visualizations
All plots are saved as high-resolution PNG files in `outputs/visualizations/`:
- Professional quality for reports and presentations
- Clear legends and annotations for easy interpretation
- Consistent styling across all plots

### Customization
Visualizations can be customized by modifying `src/evaluation/visualizations.py`:
- Color schemes and styling
- Plot layouts and formats
- Additional metrics and analyses

## Deployment

### Power Systems Context

This Industrial IDS is specifically designed for **power systems and electrical grid infrastructure**, including:
- **SCADA Systems**: Supervisory Control and Data Acquisition networks
- **Substation Automation**: Protection relays, RTUs, and IEDs
- **Power Generation**: Plant control systems and turbine controllers
- **Transmission Networks**: Line monitoring and control systems
- **Distribution Systems**: Smart grid and distribution automation

### Prerequisites

#### Power Systems Requirements
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Hardware**: 8GB+ RAM, 4+ CPU cores recommended
- **Storage**: 10GB+ available space
- **Network**: Access to power system data sources (IEC 61850, DNP3, Modbus)
- **Compliance**: NERC CIP, IEC 62351, and regional grid security standards

#### Power Systems Data Sources
- **SCADA Historians**: OSIsoft PI, GE Historian, Schneider Electric EcoStruxure
- **Protection Relays**: SEL, Schweitzer, ABB, Siemens protection devices
- **RTUs/IEDs**: Remote Terminal Units and Intelligent Electronic Devices
- **PMUs**: Phasor Measurement Units for synchrophasor data
- **Smart Grid Devices**: AMI systems, distribution automation controllers

### Quick Deployment

#### Step 1: Train Models
```bash
# Train complete pipeline with power systems data
python run_pipeline.py --stage all

# For power systems specific configuration
python run_pipeline.py --stage all --config experiments/power_systems.yaml
```

#### Step 2: Deploy System
```bash
# Deploy all services for power systems
./deployment/deploy.sh

# Or use Docker Compose directly
docker-compose -f deployment/docker-compose.yml up -d
```

#### Step 3: Verify Deployment
```bash
# Check service status
./deployment/deploy.sh status

# Run deployment tests
./deployment/deploy.sh test

# View service logs
./deployment/deploy.sh logs
```

### Power Systems Architecture

#### Grid Infrastructure Components
```
Power Systems IDS Architecture
├── Substation Layer
│   ├── Protection Relay Monitoring
│   ├── Circuit Breaker Status
│   ├── Transformer Monitoring
│   └── Bus Bar Analysis
├── SCADA/EMS Layer
│   ├── Control Command Monitoring
│   ├── Set Point Analysis
│   ├── Alarm Correlation
│   └── Operator Interface Security
├── Field Device Layer
│   ├── RTU Communication Analysis
│   ├── IED Behavior Monitoring
│   ├── Sensor Data Validation
│   └── Actuator Control Verification
└── Grid Operations Layer
    ├── Load Flow Analysis
    ├── Frequency Stability Monitoring
    ├── Voltage Regulation Security
    └── Interchange Transaction Security
```

#### Power Systems Deployment Patterns

**Substation-Level Deployment**
- Single substation deployment
- Real-time protection monitoring (<5ms latency)
- Local alarm processing and operator notification
- Integration with substation HMI systems

**Control Center Deployment**
- Regional/TSO control center deployment
- Multiple substation monitoring
- Grid-wide situational awareness
- Integration with EMS/SCADA systems

**Wide Area Monitoring System (WAMS)**
- PMU data integration for synchrophasor analysis
- Grid stability monitoring across large geographical areas
- Real-time oscillation detection
- Integration with WAMS platforms

### Access Points

After successful deployment, access the system at:

- **Main Dashboard**: http://localhost:8080
- **Grid Monitoring Dashboard**: http://localhost:8080/grid
- **Grafana Power Metrics**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090

### Power Systems API Endpoints

#### Autoencoder Service (Stage 1)
- **URL**: http://localhost:8083
- **Health Check**: `GET /health`
- **Anomaly Detection**: `POST /predict`
- **Grid-Specific Metrics**: `GET /metrics/grid`
- **Batch Processing**: `POST /batch_predict`

#### Classifier Service (Stage 2)
- **URL**: http://localhost:8084
- **Health Check**: `GET /health`
- **Threat Classification**: `POST /classify`
- **Power System Threat Types**: `GET /threats/power-systems`
- **Metrics**: `GET /metrics`

#### Grid Alert Manager
- **URL**: http://localhost:8085
- **Health Check**: `GET /health`
- **Active Grid Alerts**: `GET /alerts/grid`
- **NERC CIP Compliance**: `GET /compliance/nerc`
- **Alert History**: `GET /alerts/history`

### Power Systems API Usage Examples

#### Substation Anomaly Detection
```bash
curl -X POST "http://localhost:8083/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [0.1, 0.2, 0.3, ...],
       "timestamp": "2024-01-01T12:00:00Z",
       "window_id": "substation_001_window_001",
       "metadata": {
         "substation_id": "SUB001",
         "voltage_level": "230kV",
         "device_type": "protection_relay"
       }
     }'
```

#### Power System Threat Classification
```bash
curl -X POST "http://localhost:8084/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [0.1, 0.2, 0.3, ...],
       "timestamp": "2024-01-01T12:00:00Z",
       "grid_context": {
         "location": "control_center",
         "system_state": "normal_operation",
         "load_level": "peak"
       }
     }'
```

### Power Systems Configuration

#### Environment Variables
Create `.env` file for power systems deployment:
```bash
# Power Systems Alert Configuration
ALERT_WEBHOOK_URL=https://your-grid-control-center-webhook
EMAIL_SMTP=smtp.yourutility.com
EMAIL_USER=grid-alerts@yourutility.com
EMAIL_PASS=your-password
SLACK_WEBHOOK=https://hooks.slack.com/your-grid-operations

# Power Systems Security Configuration
NERC_CIP_MODE=true
IEC_62351_COMPLIANCE=true
GRID_SECURITY_LEVEL=high
JWT_SECRET=your-grid-jwt-secret-key
API_KEY=your-grid-api-key

# Power Systems Data Sources
SCADA_HISTORIAN_URL=https://your-scada-historian
PMU_DATA_URL=https://your-pmu-collector
RELAY_COMM_PORT=24000
```

#### Power Systems Service Configuration
Edit `deployment/config/power_systems_config.yaml`:
```yaml
data_sources:
  - name: "scada_events"
    type: "iec61850"
    path: "/app/data/scada_events.jsonl"
    protocol: "IEC 61850 MMS"
  - name: "protection_relays"
    type: "sel_comtrade"
    path: "/app/data/relay_events.jsonl"
    protocol: "SEL COMTRADE"
  - name: "pmu_data"
    type: "ieee_c37.118"
    path: "/app/data/pmu_events.jsonl"
    protocol: "IEEE C37.118"

processing:
  window_size_seconds: 90
  stride_seconds: 20
  input_dim: 196
  power_systems_mode: true

autoencoder:
  threshold_percentile: 75
  batch_size: 32
  grid_aware_training: true

classifier:
  threshold: 0.5
  batch_size: 32
  power_systems_threat_types:
    - "relay_trip_attack"
    - "scada_command_injection"
    - "pmu_data_manipulation"
    - "voltage_stability_attack"
    - "frequency_interference"

alerts:
  enabled: true
  suppression_window: 300  # 5 minutes
  nerc_cip_compliance: true
  channels: ["email", "slack", "scada_hmi", "control_center"]
  grid_escalation:
    critical: ["control_center", "system_operator"]
    high: ["substation_operator", "shift_supervisor"]
    medium: ["maintenance_team", "engineering"]
```

### Power Systems Management Commands

#### Service Management
```bash
# Stop all power systems services
./deployment/deploy.sh stop

# Restart services
./deployment/deploy.sh restart

# View specific service logs
./deployment/deploy.sh logs autoencoder
./deployment/deploy.sh logs classifier

# View grid-specific logs
./deployment/deploy.sh logs | grep "GRID_ALERT"
```

#### Grid Operations Scaling
```bash
# Scale autoencoder service for multiple substations
docker-compose -f deployment/docker-compose.yml up -d --scale autoencoder=5

# Scale classifier service for regional monitoring
docker-compose -f deployment/docker-compose.yml up -d --scale classifier=3

# Scale alert manager for control center operations
docker-compose -f deployment/docker-compose.yml up -d --scale alert-manager=2
```

### Power Systems Data Integration

#### Supported Power Systems Protocols
- **IEC 61850**: Substation communication and control
- **DNP3**: Distributed Network Protocol
- **Modbus**: Industrial communication protocol
- **IEC 60870-5-104**: Telecontrol protocols
- **IEEE C37.118**: Synchrophasor data (PMU)
- **OPC-UA**: Unified architecture for industrial communication
- **SEL COMTRADE**: Relay event data format

#### Power Systems Data Pipeline
```python
# Power systems data flow architecture
Grid Events → Feature Extraction → Autoencoder → Classifier → Grid Alert Manager
     ↓              ↓                ↓           ↓              ↓
SCADA/PMU → Sliding Window → Anomaly Score → Threat Level → Operator Notification
```

#### Grid-Specific Feature Engineering
- **Electrical Parameters**: Voltage, current, frequency, power flow
- **Protection States**: Relay status, breaker positions, protection zones
- **Control Commands**: Set points, tap changers, capacitor switching
- **System States**: Load levels, generation dispatch, topology changes
- **Time Synchronization**: GPS-synchronized timestamps for PMU data

### Power Systems Monitoring and Maintenance

#### Grid Health Monitoring
```bash
# Check all power systems service health
curl http://localhost:8083/health
curl http://localhost:8084/health
curl http://localhost:8085/health

# View grid-specific metrics
curl http://localhost:8083/metrics/grid
curl http://localhost:8084/metrics/power-systems

# Check NERC CIP compliance status
curl http://localhost:8085/compliance/nerc
```

#### Power Systems Performance Monitoring
- **Grafana Grid Dashboards**: Real-time grid visualization
- **Prometheus Grid Metrics**: Detailed power system metrics
- **SCADA Integration**: Existing control system integration
- **PMU Data Monitoring**: Synchrophasor data quality and analysis

#### Grid Model Management
```bash
# Update models (after retraining with grid data)
cp outputs/models/*.pt deployment/models/
./deployment/deploy.sh restart

# Model rollback for grid operations
./deployment/deploy.sh rollback autoencoder v1.0

# Grid-specific model validation
./deployment/deploy.sh validate --grid-data
```

### Power Systems Security

#### Grid Security Standards Compliance
- **NERC CIP**: Critical Infrastructure Protection standards
- **IEC 62351**: Power systems security standards
- **NIST Cybersecurity Framework**: Grid cybersecurity guidelines
- **ISO 27001**: Information security for power utilities

#### Container Security for Power Systems
- **Distroless Runtime**: Minimal attack surface for critical infrastructure
- **Non-root Execution**: All services run as non-root users
- **Immutable Infrastructure**: No runtime modifications possible
- **Air-Gapped Deployment**: Option for isolated grid networks

#### Grid Network Security
- **Protocol Security**: IEC 62351 compliant secure communication
- **Network Segmentation**: Substation and control center isolation
- **Intrusion Detection**: Network-level monitoring integration
- **Access Control**: Role-based access for grid operators

#### Power Systems Access Control
```yaml
# Grid operator role-based permissions
roles:
  grid_operator: ["read_grid_metrics", "view_grid_alerts", "acknowledge_grid_alerts"]
  system_operator: ["read_grid_metrics", "view_grid_alerts", "control_system_access"]
  nerc_auditor: ["read_compliance_reports", "view_audit_logs"]
  grid_admin: ["all_permissions"]
```

### Power Systems Troubleshooting

#### Common Grid Issues

**SCADA Integration Problems**
```bash
# Check SCADA data connectivity
docker-compose -f deployment/docker-compose.yml logs | grep "SCADA"

# Verify IEC 61850 protocol parsing
curl -X POST "http://localhost:8083/test/iec61850" -d '{"test": true}'
```

**PMU Data Quality Issues**
```bash
# Check PMU data stream
docker-compose -f deployment/docker-compose.yml logs | grep "PMU"

# Verify synchrophasor data quality
curl http://localhost:8083/metrics/pmu_quality
```

**Grid Alert Flooding**
```bash
# Check alert suppression rules
curl http://localhost:8085/alerts/suppression_status

# Adjust alert thresholds for grid operations
./deployment/deploy.sh configure --alert-thresholds grid
```

#### Power Systems Performance Optimization

**Substation-Level Optimization**
- Deploy autoencoder at substation edge for <5ms response
- Optimize for protection relay monitoring priorities
- Integrate with local HMI systems for operator awareness

**Control Center Optimization**
- Horizontal scaling for regional monitoring
- Load balancing across multiple grid operators
- Integration with existing EMS/SCADA visualization

**WAMS Optimization**
- PMU data stream processing optimization
- Real-time oscillation detection algorithms
- Grid stability monitoring enhancement

### Power Systems Enterprise Features

#### High Availability for Critical Infrastructure
- **Redundant Substation Deployment**: N+1 redundancy for critical substations
- **Control Center Failover**: Automatic failover between control centers
- **Grid-Wide Correlation**: Cross-substation event correlation
- **Disaster Recovery**: Backup systems for grid emergencies

#### Advanced Grid Monitoring
- **Synchrophasor Analysis**: Real-time grid stability monitoring
- **Load Flow Security**: Power flow analysis and security assessment
- **Voltage Stability Monitoring**: Real-time voltage stability analysis
- **Frequency Response Monitoring**: Grid frequency dynamics tracking

#### Grid Integration Capabilities
- **EMS Integration**: Energy Management System integration
- **ADMS Integration**: Advanced Distribution Management Systems
- **WAMS Integration**: Wide Area Measurement System integration
- **Market Systems Integration**: Energy market and trading system security

## Ablation Studies

Compare different approaches:

```bash
# AE-only vs Two-stage
python run_pipeline.py --stage all --ablation stage1_only

# With/without classifier
python run_pipeline.py --stage all --ablation no_classifier
```

## New Features and Testing Guide

This section documents the new features added to enhance the project and provides instructions for testing them.

### 1. New Autoencoder Architectures

Three new autoencoder architectures have been added for temporal and spatial pattern detection:

#### LSTM Autoencoder
- **Location**: `src/models/lstm_autoencoder.py`
- **Features**: Captures temporal patterns in time-series data
- **Models**: `LSTMAutoencoder`, `ConvLSTMAutoencoder`

#### CNN Autoencoder
- **Location**: `src/models/cnn_autoencoder.py`
- **Features**: Spatial feature extraction with convolutional layers
- **Models**: `CNNAutoencoder`, `ResidualCNNAutoencoder`

#### VAE (Variational Autoencoder)
- **Location**: `src/models/vae_autoencoder.py`
- **Features**: Probabilistic latent space for uncertainty modeling
- **Models**: `VAEAutoencoder`, `BetaVAE`, `ConditionalVAE`

#### Testing New Architectures

To use a new architecture, modify the model loading in your training script:

```python
# Example: Using LSTM Autoencoder
from src.models.lstm_autoencoder import LSTMAutoencoder

model = LSTMAutoencoder(
    input_dim=196,
    hidden_dim=128,
    latent_dim=64,
    num_layers=2
)
```

### 2. Enhanced Threshold Optimization

Threshold optimization has been expanded with a broader search space and sensitivity heatmap generation.

#### New Features
- **Expanded Search Space**: AE percentile (60-95), classifier threshold (0.2-0.8)
- **Sensitivity Heatmap**: Visualizes performance across threshold combinations
- **CSV Export**: Detailed results of all threshold combinations

#### Testing Threshold Optimization

```bash
# Run with auto-tuning enabled
python run_pipeline.py --stage all --auto_tune

# View generated outputs
# - outputs/auto_tune_results.csv: Detailed threshold search results
# - outputs/threshold_sensitivity_heatmap.png: Sensitivity visualization
```

#### Configuration in YAML

Edit `experiments/example.yaml`:

```yaml
tuning:
  max_fpr_on_normal: 0.01
  max_fpr_on_benign: 0.03
  ae_percentile: 80.0      # New: Configurable via YAML
  clf_threshold: 0.5        # New: Configurable via YAML
  benign_ctrl_threshold: 0.01  # New: Configurable via YAML
```

### 3. Cross-Validation Module

Statistical validity assessment through k-fold cross-validation.

#### Location
- `src/evaluation/cross_validation.py`

#### Features
- **Autoencoder Cross-Validation**: `cross_validate_autoencoder()`
- **Classifier Cross-Validation**: `cross_validate_classifier()`
- **Result Summarization**: Mean, std, min, max across folds
- **Formatted Output**: Easy-to-read CV summaries

#### Testing Cross-Validation

```python
from src.evaluation.cross_validation import cross_validate_autoencoder, print_cv_summary

# Run 5-fold cross-validation for autoencoder
cv_results = cross_validate_autoencoder(
    X=X_train,
    input_dim=196,
    hidden_dim=256,
    latent_dim=64,
    n_folds=5,
    epochs=25,
    device='cuda'
)

# Print summary
print_cv_summary(cv_results, model_name="Autoencoder")
```

### 4. SHAP-Driven Feature Selection

Feature selection based on SHAP importance scores to prune low-importance features.

#### Location
- `src/features/feature_selection.py`

#### Features
- **Feature Selection**: Select top N% features by SHAP importance
- **Feature Pruning**: Remove bottom features from datasets
- **Ranking**: Get feature importance ranking
- **Pipeline Integration**: Apply selection to train/val/test sets

#### Testing Feature Selection

```python
from src.features.feature_selection import (
    select_features_by_shap,
    prune_features,
    apply_feature_selection_pipeline
)

# Select features (keep top 80%)
feature_indices, feature_names = select_features_by_shap(
    shap_values=shap_values,
    keep_percentage=0.8
)

# Prune datasets
X_train_pruned = prune_features(X_train, feature_indices)
X_val_pruned = prune_features(X_val, feature_indices)
X_test_pruned = prune_features(X_test, feature_indices)

# Or use the pipeline function
X_train_pruned, X_val_pruned, X_test_pruned, indices = apply_feature_selection_pipeline(
    X_train, X_val, X_test, shap_values, keep_percentage=0.8
)
```

### 5. Modular Pipeline Stages

Pipeline has been refactored for independent Stage 1 and Stage 2 execution.

#### Locations
- `src/pipeline/stage1.py`: Autoencoder training/inference
- `src/pipeline/stage2.py`: Classifier training/inference

#### Features
- **Independent Execution**: Train or use each stage separately
- **Model Loading**: Load pre-trained models independently
- **Inference**: Run inference for specific stages
- **Rule-Based Rescue**: Apply rescue logic for specific attack families

#### Testing Modular Pipeline

```python
from src.pipeline.stage1 import train_stage1, load_stage1, infer_stage1
from src.pipeline.stage2 import train_stage2, load_stage2, infer_stage2

# Train Stage 1 independently
ae_model = train_stage1(
    X_train=X_train,
    X_val=X_val,
    input_dim=196,
    hidden_dim=256,
    latent_dim=64,
    model_save_path=Path("outputs/models/autoencoder.pt")
)

# Load and run Stage 1 inference
ae_model = load_stage1(
    model_path=Path("outputs/models/autoencoder.pt"),
    input_dim=196
)
reconstruction_errors = infer_stage1(ae_model, X_test)

# Train Stage 2 independently
clf_model = train_stage2(
    X_train=X_anomalous,
    y_train=y_anomalous,
    X_val=X_val_anomalous,
    y_val=y_val_anomalous,
    input_dim=196,
    hidden_dim=128,
    model_save_path=Path("outputs/models/classifier.pt")
)
```

### 6. Experiment Tracking

Integration with MLflow and Weights & Biases for experiment tracking.

#### Location
- `src/experiment_tracking.py`

#### Features
- **MLflow Integration**: Log parameters, metrics, and artifacts
- **Weights & Biases Integration**: W&B experiment tracking
- **Simple Fallback**: File-based tracking when MLflow/W&B unavailable
- **Unified Interface**: Consistent API across backends

#### Testing Experiment Tracking

```python
from src.experiment_tracking import get_tracker

# Auto-detect available backend or specify explicitly
tracker = get_tracker(tracking_backend="mlflow")

# Initialize experiment
tracker.init_experiment(
    experiment_name="ids_experiment_001",
    config={"hidden_dim": 256, "latent_dim": 64}
)

# Log hyperparameters
tracker.log_params({"lr": 0.001, "batch_size": 256})

# Log metrics during training
tracker.log_metrics({"train_loss": 0.1, "val_loss": 0.15}, step=1)

# Log artifacts (models, visualizations)
tracker.log_artifact("outputs/models/autoencoder.pt", "autoencoder")

# End experiment run
tracker.end_run()
```

#### Installation

```bash
# For MLflow
pip install mlflow

# For Weights & Biases
pip install wandb
```

### 7. Unit Tests

Comprehensive test suite for critical functions.

#### Locations
- `tests/test_features.py`: Feature extraction tests
- `tests/test_pipeline.py`: Pipeline logic tests
- `pytest.ini`: Test configuration

#### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_features.py

# Run with coverage
pytest --cov=src tests/
```

#### Test Coverage

- **Feature Extraction**: Window statistics, edge cases, NaN handling
- **Pipeline Logic**: Confusion matrix, threshold classification, data splits
- **Sample Weighting**: Weight application in loss calculation
- **Data Leakage Prevention**: Train/test split validation

### 8. New Demo Scenarios

Enhanced demo with additional attack scenarios.

#### New Scenarios
- **Industroyer Attack**: Stealthy protocol manipulation (reflecting 44.3% detection rate)
- **Multi-Stage Attack**: Escalating attack from benign-looking to malicious

#### Testing New Demo Scenarios

```bash
# Run enhanced demo with new scenarios
python demo_improved.py

# The demo now includes:
# - industroyer_attack: Stealthy protocol manipulation scenario
# - multi_stage_attack: Escalating attack scenario
```

#### Feature Generation

The new scenarios have specialized feature generation functions:
- `generate_industroyer_attack_features()`: Subtle, hard-to-detect patterns
- `generate_multi_stage_attack_features()`: Escalating pattern from benign to malicious

### 9. Enhanced Visualizations

New visualization added to the evaluation pipeline.

#### SHAP Beeswarm Plot

**Location**: `src/evaluation/visualizations.py`

**Function**: `plot_shap_beeswarm()`

**Features**: Shows feature impact distribution across samples with color-coding for positive/negative contributions

#### Testing SHAP Beeswarm

```python
from src.evaluation.visualizations import plot_shap_beeswarm

# Generate beeswarm plot
fig = plot_shap_beeswarm(
    shap_values=shap_values,
    feature_names=feature_names,
    save_path=Path("outputs/visualizations/shap_beeswarm.png")
)
```

### Quick Testing Checklist

To test all new features systematically:

```bash
# 1. Test threshold optimization
python run_pipeline.py --stage all --auto_tune

# 2. Run unit tests
pytest -v

# 3. Test enhanced demo
python demo_improved.py

# 4. Test modular pipeline (requires Python script)
# Create a test script using examples from sections above

# 5. Test experiment tracking (requires MLflow or W&B installation)
# Create a test script using examples from section 6

# 6. Test cross-validation (requires Python script)
# Create a test script using examples from section 3

# 7. Test feature selection (requires Python script)
# Create a test script using examples from section 4
```

### Configuration Updates

The YAML configuration file has been enhanced with new parameters:

```yaml
# experiments/example.yaml
tuning:
  max_fpr_on_normal: 0.01
  max_fpr_on_benign: 0.03
  ae_percentile: 80.0      # NEW: Autoencoder threshold
  clf_threshold: 0.5        # NEW: Classifier threshold
  benign_ctrl_threshold: 0.01  # NEW: Benign control threshold

sample_weights:
  attack_family_weights:
    arp-spoof: 4.0
    industroyer: 8.0         # UPDATED: Increased from 4.0
    drift-off: 2.0
    control-and-freeze: 1.0
  inverse_freq_boost: true
  max_inverse_freq_weight: 8.0  # NEW: Max inverse frequency weight
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

For full dataset details, licensing terms, and citation information, please visit: https://sherlock.wattson.it


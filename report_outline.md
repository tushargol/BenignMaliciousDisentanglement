### Capstone report outline

**Implementation note**: The pipeline now includes per-attack-type analysis, SHAP interpretability, ablation support, and baseline comparison. Use `python run_pipeline.py --stage all` to reproduce experiments.

#### 1. Introduction
- Motivation: high false positives in power-grid IDS due to benign operational changes
- Goal: detect malicious attacks while suppressing benign anomaly alerts
- Contributions: label construction, two-stage model, evaluation on Sherlock (02-Semiurban)

#### 2. Background
- Process-aware intrusion detection for power grids
- Sherlock dataset + Wattson co-simulation
- Benign vs malicious anomaly disentanglement (multi-task, contrastive, two-stage)

#### 3. Dataset and problem formulation
- Scenario: 02-Semiurban
- Signals used (physical state / protocol / logs)
- Ground truth: `ipal/*/events.json` attack + benign event intervals
- Label semantics: normal vs benign anomaly vs malicious anomaly (windowed)

#### 4. Method
- Baseline: one-stage anomaly detector (no benign/malicious separation)
- Proposed: two-stage pipeline
  - Stage 1: anomaly detector
  - Stage 2: benign vs malicious classifier for anomalies
- Feature engineering and windowing

#### 5. Experimental setup
- Splits, thresholds, tuning
- Metrics: malicious recall, FPR on benign, FPR on normal, PR curves, F1, AUC

#### 6. Results
- Baseline vs proposed
- Per-attack-type analysis (DoS, Industroyer, FDIA, Control & Freeze)
- Case studies: examples of suppressed benign alarms and detected attacks

#### 7. Discussion
- Failure modes, limitations, dataset assumptions
- Generalization considerations across scenarios

#### 8. Conclusion and future work
- Summary of findings
- Future: contrastive embeddings, rule-augmented models, multi-scenario training


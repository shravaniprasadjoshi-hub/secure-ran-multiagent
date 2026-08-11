# Training, Validation & Testing Guide

## Data Split

| Split | Ratio | Rows (approx.) | Purpose |
|-------|-------|----------------|---------|
| Train | 70% | 49,000 | Model fitting |
| Validation | 15% | 10,500 | Hyperparameter monitoring, overfitting check |
| Test | 15% | 10,500 | Final unbiased evaluation |

Stratified by `scenario_type` to preserve class balance across attack/normal scenarios.

## Per-Agent Training Pipeline

```
Load CSV → Filter features → StandardScaler (fit on train)
    → Train model → Predict on train/val/test → Compute metrics → Save plots
```

## Metrics

### Classification Agents (Mobility, Security, Energy, QoS, Policy)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | ✓ | ✓ | ✓ |
| Precision (weighted) | ✓ | ✓ | ✓ |
| Recall (weighted) | ✓ | ✓ | ✓ |
| F1 (weighted) | ✓ | ✓ | ✓ |
| ROC-AUC (binary) | — | — | ✓ |

**Plots:** Confusion matrix (test), ROC curve (test, binary), learning curve

### Regression Agents (Resource, Trust, Beamforming)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| MAE | ✓ | ✓ | ✓ |
| RMSE | ✓ | ✓ | ✓ |
| R² | ✓ | ✓ | ✓ |

**Plots:** Learning curve

## Model Selection

Models defined in `config.yaml`:

| Agent | Algorithm | Rationale |
|-------|-----------|-----------|
| Mobility | Gradient Boosting | Non-linear HO decision boundaries |
| Security | Random Forest | Robust to imbalanced threat classes |
| Resource | Gradient Boosting | PRB allocation regression |
| Trust | MLP | Non-linear trust score mapping |
| Beamforming | Gradient Boosting | Beam index from RF features |
| Energy/QoS/Policy | RF/GB | Stable classification |

## Outputs

After `python run_pipeline.py`:

```
outputs/
├── models/           # *.joblib per agent
├── metrics/
│   ├── all_metrics.json
│   └── coordinator_metrics.json
└── plots/
    ├── data/
    ├── agents/
    ├── training/
    ├── digital_twin/
    └── architecture/
```

## Reproducibility

- Random seed: 42 (config.yaml)
- Same split across all agents
- Scaler fit only on training data (no data leakage)

## Extending to MARL / Deep Learning

The architecture supports replacing sklearn models with:
- **LSTM/Transformer** for mobility (PyTorch)
- **Autoencoder** for security anomaly detection
- **MADDPG/MAPPO** for multi-agent coordination (Ray RLlib)

Current implementation uses sklearn for fast, reproducible baseline training on 70k rows.

# Secure Multi-Agent AI Framework for RAN Control Loops

**NBUC Project 03** | Nokia Bangalore University Collaboration  
**Contact:** Swetha Kerahalli, MI MN RAN RD AS Algo Innov  
**Domain:** AI-Native 6G Open RAN | Multi-Agent MARL | Secure RRC Control | Digital Twin

---

## Dashboard

After setup and pipeline run, launch the interactive dashboard:

```bash
cd proj3_code
streamlit run dashboard/app.py
```

**Dashboard URL:** [http://localhost:8501](http://localhost:8501)

The dashboard opens automatically in your browser. If port 8501 is busy, Streamlit will use the next available port (e.g. `http://localhost:8502`) — check the terminal output for the exact URL.

### Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | Architecture KPIs, scenario distribution, consensus metrics |
| Data Exploration | Histograms, correlation heatmaps, CDFs, scenario analysis |
| Agent Evaluation | Per-agent train/val/test metrics, confusion matrices, ROC curves |
| Train/Val/Test | All 8 agents performance comparison |
| Digital Twin | KPI time series, cell state map, gNB heatmap |
| Chatbot | RAG Q&A over project knowledge base (3GPP, O-RAN, security) |

---

## Prerequisites

- **Python:** 3.10 or higher
- **OS:** Windows / Linux / macOS
- **RAM:** 4 GB minimum (8 GB recommended for full pipeline)
- **Disk:** ~500 MB for datasets, models, and plots

---

## Installation

```bash
# 1. Navigate to project directory
cd proj3_code

# 2. (Recommended) Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run — Step by Step

### Step 1: Generate Datasets (first time only)

Creates synthetic RAG corpus (70 chunks) and telemetry CSV (70,000 rows × 42 columns).

```bash
python generate_datasets.py
```

**Outputs:**
- `data/rag_corpus.jsonl`
- `data/ran_multi_agent_telemetry_70k.csv`
- `data/dataset_metadata.json`

### Step 2: Run End-to-End Pipeline

Trains all 8 agents, evaluates on train/val/test, generates plots, runs digital twin.

```bash
python run_pipeline.py
```

**Expected runtime:** ~15–45 minutes (depends on hardware)

**Outputs:**
- `outputs/models/*.joblib` — 8 trained agent models
- `outputs/metrics/all_metrics.json` — train/val/test metrics
- `outputs/plots/` — 27+ plots (data, agents, training, digital twin)
- `outputs/digital_twin/` — twin simulation CSVs
- `outputs/reports/pipeline_summary.json`

### Step 2 (Alternative): Fast Re-Evaluation

If models already exist, skip retraining and regenerate plots/metrics only:

```bash
python run_evaluation.py
```

**Expected runtime:** ~3–5 minutes

### Step 3: Generate Slides (optional)

```bash
python slides/generate_slides.py
```

**Output:** `docs/slides/End_to_End_Implementation.pptx`

### Step 4: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
proj3_code/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml               # Agent configs, paths, consensus thresholds
├── run_pipeline.py           # Full pipeline: train + evaluate + plots + twin
├── run_evaluation.py         # Fast re-evaluation from saved models
├── generate_datasets.py      # Synthetic dataset generator
│
├── data/
│   ├── ran_multi_agent_telemetry_70k.csv   # 70k telemetry records
│   ├── rag_corpus.jsonl                    # RAG knowledge chunks
│   └── dataset_metadata.json
│
├── src/
│   ├── data/                 # Loaders, preprocessing, train/val/test split
│   ├── training/             # Agent model training (8 agents)
│   ├── orchestration/        # Consensus engine, multi-agent coordinator
│   ├── digital_twin/         # RAN digital twin (48 cells, 12 gNBs)
│   ├── chatbot/              # RAG knowledge chatbot
│   └── visualization/        # Plot generation (CDFs, heatmaps, ROC, etc.)
│
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
│
├── outputs/                  # Generated after pipeline run
│   ├── models/               # Trained .joblib models
│   ├── plots/                # All visualization PNGs
│   ├── metrics/              # JSON metrics
│   ├── reports/              # Pipeline summary
│   └── digital_twin/         # Twin simulation data
│
├── docs/
│   ├── END_TO_END_IMPLEMENTATION.md
│   ├── ARCHITECTURE.md
│   ├── TRAINING_GUIDE.md
│   ├── AGENT_REFERENCE.md
│   ├── DASHBOARD_GUIDE.md
│   └── slides/End_to_End_Implementation.pptx
│
└── slides/
    └── generate_slides.py
```

---

## Implementation Overview

### 8 AI Agents

| Agent | Model | Task | Target |
|-------|-------|------|--------|
| Mobility | Gradient Boosting | Classification | `ho_required` |
| Security | Random Forest | Classification | `threat_label` |
| Resource | Gradient Boosting | Regression | `allocated_prb_count` |
| Energy | Random Forest | Classification | `energy_label` |
| Trust | MLP | Regression | `trust_score` |
| Beamforming | Gradient Boosting | Regression | `beam_index` |
| QoS | Random Forest | Classification | `ho_success` |
| Policy | Gradient Boosting | Classification | `rrc_action` |

### Data Split

- **Train:** 70% (49,000 rows)
- **Validation:** 15% (10,500 rows)
- **Test:** 15% (10,501 rows)  
- Stratified by `scenario_type` (6 scenarios)

### Consensus Engine

- Byzantine Fault Tolerance + weighted trust voting
- Thresholds: majority >70%, trust >0.8, confidence >0.85

### Digital Twin

- 48 cells across 12 gNBs
- 200 simulation steps with jamming attack at step 50
- KPIs: SINR, throughput, trust, HO count, mitigations

---

## Generated Plots

| Category | Files |
|----------|-------|
| Data | Dataset overview, correlation heatmap, CDFs |
| Agents | F1/R² comparison, performance heatmap, confusion matrices, ROC curves |
| Training | Learning curves (all 8 agents) |
| Digital Twin | Time series KPIs, cell state map, gNB heatmap |
| Architecture | Consensus accept rate, RRC action distribution |

All plots saved under `outputs/plots/`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` from `proj3_code/` |
| Dashboard shows no metrics | Run `python run_pipeline.py` or `python run_evaluation.py` first |
| Port 8501 in use | Streamlit auto-selects next port; check terminal for URL |
| Pipeline slow / hangs | Use `python run_evaluation.py` if models already exist |
| Matplotlib display errors | Pipeline uses non-interactive `Agg` backend; no action needed |

---

## Documentation

| Document | Description |
|----------|-------------|
| [END_TO_END_IMPLEMENTATION.md](docs/END_TO_END_IMPLEMENTATION.md) | Complete implementation guide |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture & O-RAN mapping |
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Train/val/test methodology |
| [AGENT_REFERENCE.md](docs/AGENT_REFERENCE.md) | Per-agent specifications |
| [DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md) | Dashboard usage |

---

## References

- **3GPP:** TS 38.300, 38.331, 38.215, 38.214, 28.530, 28.105, 23.288, 33.501
- **3GPP TR:** TR 38.817, TR 23.700-80
- **O-RAN:** Near-RT RIC, E2GAP, xApp/rApp specifications
- **IEEE:** MobiLLM, AI-Augmented Predictive Mobility, Jamming-Resilient HO (RL)
- **NIST:** PQC (CRYSTALS-Kyber, Dilithium), SP 800-207 Zero Trust
- **Nokia/NBUC:** Problem Statement — Swetha Kerahalli, MI MN RAN RD AS Algo Innov

---

## Quick Command Reference

```bash
pip install -r requirements.txt          # Install dependencies
python generate_datasets.py              # Generate datasets (first time)
python run_pipeline.py                   # Full train + evaluate + plots
python run_evaluation.py                 # Fast re-eval from saved models
python slides/generate_slides.py         # Generate PPTX slides
streamlit run dashboard/app.py           # Launch dashboard → http://localhost:8501
```

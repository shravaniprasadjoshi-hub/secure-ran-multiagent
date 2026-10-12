# Secure Multi-Agent AI Framework for RAN Control Loops

Nokia Bangalore University Collaboration (NBUC) · PESU

**Mentor:** Swetha Kerahalli (Nokia)

**Faculty:** Rajashekar Mohan (PESU)

**Team:** Shreyashree S Rao, Shravani P Joshi, Shashank, Shloka

---

## Overview

A MAPPO-based multi-agent RL system where 7 agents (one per hex cell, 3 gNBs) make RRC-layer handover decisions. A security stack is added step by step (ablation style): Byzantine fault injection, anomaly detection, trust scoring, consensus, and a post-quantum secure channel (PQC). A live dashboard shows everything.

---

## 1. Observation Space (9 features)

The local observation per agent was expanded from 3 to 9 features (`o_i^t ∈ R^9`):

| # | Feature | What it captures |
|---|---------|------------------|
| 1 | Serving SINR | Link quality of the current serving cell |
| 2 | Best-neighbor RSRP delta (ΔRSRP) | How much stronger the best neighbor is than the serving cell |
| 3 | NACK density proxy | Retransmission pressure, rises as SINR drops |
| 4 | Normalized serving RSRP | Absolute signal strength of the serving cell |
| 5 | Normalized UE speed | Mobility level of the UE |
| 6 | Recent handover count | Recent HO activity, helps avoid ping-pong |
| 7 | Cell load | How busy the serving cell is |
| 8 | Best-neighbor SINR | Link quality available if the UE hands over |
| 9 | SINR variance across cells | Signal instability, a jamming/interference indicator |

`is_handover_zone` (MATLAB label) is deliberately **excluded** from the observation to prevent label leakage.

---

## 2. Repo Layout

```
PESU_MARL_implementation/
├── data/
│   ├── synthetic/                            # read-only  files
|   |   ├── ran_multi_agent_telemetry_70k.csv     # synthetic telemetry (70k x 42)
│   |   ├── dataset_metadata.json
│   |   └── rag_corpus.jsonl
│   └── matlab/rsrp_map.csv, sinr_map.csv     # MATLAB grids (seed=42)
├── env/        cell.py  ue.py  channel.py  replay_loader.py  telemetry_loader.py  ran_env.py
├── agents/     actor.py  critic.py  mappo_agent.py  agent_manager.py
├── security/   byzantine.py  anomaly_detector.py  policy_checker.py  pqc_channel.py
├── coordination/  consensus.py  trust.py
├── training/   train.py  train_secure.py  evaluate.py  stats_analysis.py  stats_plots.py  config.yaml
├── dashboard/  api.py  sim_runner.py  index.html  main.js  style.css
└── docs/       experiment_3_findings.md  ablation_comparison_table.md training_curves_analysis.md
```

---

## 3. Setup

**Python 3.10+ recommended.**

```bash
cd PESU_MARL_implementation
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install torch numpy pandas scipy pyyaml matplotlib seaborn scikit-learn
pip install pettingzoo gymnasium
pip install fastapi "uvicorn[standard]"
pip install cryptography
```

### PQC dependency (liboqs) - needed for Exp 4

`liboqs-python` needs the liboqs C library built first.

```bash
# Linux / macOS (needs cmake, gcc/clang, git, openssl dev headers)
git clone --depth 1 https://github.com/open-quantum-safe/liboqs
cmake -S liboqs -B liboqs/build -DBUILD_SHARED_LIBS=ON
cmake --build liboqs/build --parallel 4
sudo cmake --build liboqs/build --target install
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH     # macOS: DYLD_LIBRARY_PATH

pip install git+https://github.com/open-quantum-safe/liboqs-python
```

Smoke test (should print `smoke test passed`):

```bash
python security/pqc_channel.py
```

Algorithm names: older liboqs = `Kyber768` / `Dilithium3`, newer = `ML-KEM-768` / `ML-DSA-65`. The code tries the legacy name first and falls back automatically. If liboqs is missing, the sim runs without the PQC layer (Exp 3 behavior) instead of crashing.

---

## 4. How to Run

### 4.1 Dashboard

Two terminals.

```bash
# Terminal 1 - backend
cd PESU_MARL_implementation
pip install fastapi uvicorn pandas numpy --break-system-packages
uvicorn dashboard.api:app --reload --port 8000
```

```bash
# Terminal 2 - frontend
cd PESU_MARL_implementation
python -m http.server 5500
```

Open **http://localhost:5500/dashboard/**. Backend health check: **http://localhost:8000/health**

> If you see a directory listing instead of the dashboard, the HTTP server was started from the wrong folder.

## Experiments, training and Evaluations:

All commands from `PESU_MARL_implementation/`.

### 4.2 Exp 2 - MAPPO only (no security)

```bash
python training/train.py --config training/config.yaml
# resume:
python training/train.py --config training/config.yaml --resume training/outputs/checkpoints/ep_500
```
Outputs: `training/outputs/train_log.csv`, `training/outputs/checkpoints/{ep_*,final}`

### 4.3 Exp 3 / Exp 4 - MAPPO + security (+ PQC)

```bash
python training/train_secure.py --config training/config.yaml
```
- Byzantine agents and attack types come from `security:` in `config.yaml` (default: cell 3 `gradual`, cell 6 `random`).
- Exp 3 = PQC off. Exp 4 = set `enable_pqc: true` under `security:`.
- Outputs: `training/outputs/train_log_secure.csv`, `training/outputs/checkpoints_secure/{ep_*,final}`
- Extra logged metrics: `consensus_rate`, `byzantine_detection_rate`, `pqc_detection_rate`, `false_positive_rate`, `avg_trust_score`

### 4.4 Evaluate

```bash
# live analytic model
python training/evaluate.py --config training/config.yaml --checkpoint training/outputs/checkpoints/final

# MATLAB replay (3GPP-traceable, use for final numbers)
python training/evaluate.py --config training/config.yaml --checkpoint training/outputs/checkpoints/final --replay
```

### 4.5 Training statistics + plots

```bash
python training/stats_analysis.py --logs training/outputs/train_log.csv training/outputs/train_log_secure.csv
python training/stats_plots.py    --logs training/outputs/train_log.csv training/outputs/train_log_secure.csv
```
Outputs: `training/outputs/stats_report.json` + 5 PNGs in `training/outputs/plots/`.

---

## 5. Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Headline cards (70k records, 8 sklearn + 7 MARL agents, consensus rate), architecture layers, recent alerts, MARL trust bars, scenario distribution pie |
| **Data Exploration** | RSRP / SINR histograms, feature correlation heatmap, latency CDF, median SINR per scenario (from the 70k Nokia CSV). SINR drops sharply under jamming |
| **Agent Evaluation** | Nokia's 8 sklearn agents (Exp 1): pick an agent, see Train / Val / Test F1 (classification) or R² (regression) + summary table |
| **Train/Val/Test** | All 8 sklearn agents grouped bar chart (70/15/15 split by scenario) + model details table |
| **Digital Twin + Security** | Clickable hex map of 7 cells / 3 gNBs. Buttons: **Run live sim**, **Inject Byzantine fault** (pick attack + cell), **Simulate handover**, **Clear faults**. Below it: Byzantine fault status, live trust scores, and a consensus log (step, agreement %, excluded agents, PQC-rejected agents) |
| **MARL Training** | MAPPO reward curve (+50-ep moving average), actor/critic loss, and the ablation grid (Exp 1–4) with a note on Exp 3 |
| **PQC** | Live PQC channel stats from `/state`: KEM + signature algorithm in use, KEM handshakes, signatures issued, verifications, rejections (tampered/forged messages caught), avg sign / verify time (ms) |
| **Chatbot** | RAG chatbot over Nokia's `rag_corpus.jsonl` (70 chunks, 22 categories). Falls back to cached answers if the API is offline |

### Fault injection flow (important - detection is real, not faked)
1. Pick attack + cell → **Inject**. This only sets *ground truth*; the cell stays green.
2. Click **Run live sim**. Only when the AnomalyDetector or PQC layer actually flags the agent does it turn red, trust drops, and the banner appears.
3. **Clear all faults** resets everything.

---

## 6. Security Pipeline (per step)

```
raw MAPPO action
  → Byzantine corruption (if agent compromised)
  → PQC: sign + encrypt (agent) → verify + decrypt (coordinator)
  → anomaly detection (statistical / voting / drift)
  → trust update + policy check
  → consensus (diagnostic metric)
  → quarantine: trust < 0.3 → force action = defer
  → executed action
```

PQC-flagged agents are merged into the same `flagged` set as anomaly detections (equal severity).

---

## 7. PQC Details

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Key exchange | Kyber768 / ML-KEM-768 | One handshake per agent per episode → shared secret |
| Key derivation | HKDF-SHA256 | Per-agent AES-256 session key (bound to agent_id) |
| Encryption | AES-256-GCM | Confidentiality of each action message |
| Signatures | Dilithium3 / ML-DSA-65 | Authenticity + integrity; long-term identity key per agent, persists across episodes |

Threat covered: **message spoofing / forgery / tampering in transit** (different from behavioral attacks). In the sim, compromised agents also tamper their messages (`simulate_tamper`: `flip_action`, `forge_sig`), so PQC rejection should approach 100% for them.

**Known gaps (future work):**
- No replay protection (a resent valid message still verifies)
- One KEM handshake per episode, not per message
- Cannot catch a compromised agent that signs correctly with its own valid key
- No latency benchmarking vs. the near-RT RIC budget

---

## 8. Ablation / Results

| Exp | Setup | Result | Status |
|-----|-------|--------|--------|
| 1 | Nokia supervised sklearn + static consensus | 6.6% consensus accept rate | Done (reference) |
| 2 | MAPPO only | Final eval reward 962 | Converged |
| 3 | MAPPO + Byzantine + anomaly + trust + consensus | Reward -2,520 → -259 over 1000 eps | Improving |
| 3 | MAPPO + Byzantine injection + anomaly detection + trust + consensus | Reward -2,520 → -259 over 1000 eps | Converging |
| 4 | Exp 3 + PQC secure channel (Kyber768 / Dilithium3 / AES-256-GCM) | 100% PQC detection rate (tampered/forged messages rejected before consensus) | Done |

Why Exp 3 lags: quarantined agents skip PPO updates, so security costs training signal. This exploration–security tradeoff is a finding, not a bug. Details: `docs/experiment_3_findings.md`, `docs/ablation_comparison_table.md`.

---

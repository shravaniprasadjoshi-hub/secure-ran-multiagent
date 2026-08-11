# Data

This folder contains two datasets used in the Secure Multi-Agent AI Framework for RAN Control Loops project.

---

## 1. MATLAB Generated Dataset (`matlab/`)

Channel model data generated using the 3GPP TR 38.901 Urban Macro (UMa) path loss model across a 1km × 1km simulation area with a 7-cell hexagonal layout (ISD = 500m).

### Files

**`rsrp_map.csv`**
Reference Signal Received Power map. Contains RSRP values in dBm from all 7 cells at every position on a 10m resolution grid. Each row is one (x, y) position with 9 columns — x, y, and rsrp_cell0 through rsrp_cell6. Feeds directly into the RL agents' observation space via channel.py.

**`sinr_map.csv`**
Signal to Interference plus Noise Ratio map. Derived from rsrp_map.csv. Contains SINR values per cell at every position, along with serving_cell (dominant cell at each position) and is_handover_zone (1 if top two cells are within 3dB, 0 otherwise). Used as ground truth for reward calculation during agent training.

**`serving_cell_map.png`**
Voronoi coverage visualization showing which of the 7 cells serves each area of the simulation grid. Validates correct hexagonal layout and channel model implementation.

**`handover_zone_map.png`**
Handover decision zone visualization. Red regions indicate positions where RRC handover decisions are required (top two cells within 3dB). Blue regions are stable zones where one cell clearly dominates. Informs where agents are most active and where Byzantine manipulation has maximum impact.

### Generation Parameters

| Parameter | Value |
|---|---|
| Area | 1km × 1km |
| Grid resolution | 10m |
| Number of cells | 7 (hexagonal layout) |
| Inter-site distance | 500m |
| Carrier frequency | 3.5 GHz |
| Transmit power | 46 dBm |
| Path loss model | 3GPP TR 38.901 UMa LOS |
| Noise power | -104 dBm |
| Handover threshold | 3dB |

---

## 2. Nokia Provided Dataset (`nokia_provided/`)

Real-world telemetry dataset provided by Nokia Bangalore as part of the NBUC research collaboration. Contains labeled multi-agent RAN telemetry across multiple security attack scenarios.

### Files

**`ran_multi_agent_telemetry_70k.csv`**
70,000 rows of per-UE per-timestamp RAN telemetry data across 42 columns covering radio metrics, mobility events, network state, and security labels. Includes pre-labeled attack scenarios with trust scores, attack probability, anomaly scores, and consensus vote percentages — directly usable for security module validation without hand-crafting adversarial scenarios.

**`dataset_metadata.json`**
Schema definition and scenario distribution for the telemetry CSV. Contains column descriptions, data types, and scenario labels. Use this first to validate compatibility before ingesting into the pipeline.

**`rag_corpus.jsonl`**
Chunked domain knowledge corpus covering 3GPP standards, O-RAN architecture, MARL concepts, and network security. Provided as supplementary reference material. Not currently used in the RL training pipeline — marked as stretch goal for potential retrieval-augmented agent context in future phases.

---

## How These Datasets Are Used

| Stage | Dataset | Purpose |
|---|---|---|
| RL agent training | Both | MATLAB for 3GPP PHY-layer fidelity, Nokia for scenario diversity and scale |
| Baseline and ablation testing | Nokia | Pre-labeled ho_required and ho_success columns enable fast single-agent baseline |
| Security module validation | Nokia | Built-in trust_score, attack_probability, anomaly_score, consensus_vote_pct columns directly validate byzantine.py and anomaly_detector.py |
| Final results and paper | MATLAB | 3GPP-traceable channel model provides defensible publication claims |

---

## Important Notes

The Nokia telemetry CSV is static historical data, not a live simulation feed. To use it in ran_env.py a replay mode is required — grouping rows by timestamp and cell_id and feeding them as observations per step rather than computing physics live via channel.py. This is distinct from the live simulation mode which uses the MATLAB CSVs.

Reward function for Nokia dataset must be derived from ho_success, latency, and packet_loss columns as there is no explicit reward column.
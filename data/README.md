# Multi-Agent AI Framework for RAN Control Loops

University Connect — NBUC Project 03

## Submissions

| Folder | Author | Description |
|--------|--------|-------------|
| [`Swetha_proj3_code/`](Swetha_proj3_code/) | Swetha Kerahalli (Nokia) | End-to-end implementation: 8 AI agents, consensus engine, digital twin, dashboard, RAG chatbot |

---

# Project 03 Synthetic Datasets

Datasets for **Secure Multi-Agent AI Framework for RAN Control Loops** (NBUC / Nokia).

## Files

| File | Description |
|------|-------------|
| `rag_corpus.jsonl` | RAG knowledge chunks (3GPP, O-RAN, MARL, security, workflows) |
| `ran_multi_agent_telemetry_70k.csv` | 70,000 rows × 42 columns of synthetic RAN telemetry |
| `dataset_metadata.json` | Schema, scenario distribution, generation metadata |
| `sinr_map.csv` | MATLAB-generated (5G Toolbox) SINR per cell, serving cell assignment, handover zone flags: 7-cell hexagonal layout, seed=42 |
| `rsrp_map.csv` | MATLAB-generated (5G Toolbox) RSRP per cell: 7-cell hexagonal layout, seed=42 |
| `serving_cell_map.png` | Visualization of serving cell assignment (from `sinr_map.csv`) |
| `handover_zone_map.png` | Visualization of handover decision zones (from `sinr_map.csv`) |

## Regenerate

```bash
python ../generate_datasets.py
```

## RAG Corpus (`rag_corpus.jsonl`)

JSONL format, one chunk per line:

```json
{
  "doc_id": "uuid",
  "chunk_id": "chunk_0001",
  "title": "...",
  "category": "3gpp_rrc | o_ran | security | marl | ...",
  "source": "3GPP TS 38.331 / O-RAN / Project docs",
  "content": "Full text for embedding and retrieval",
  "keywords": ["handover", "RSRP"],
  "metadata": {}
}
```

**Categories:** project_overview, architecture, 3gpp_*, o_ran, marl, security, quantum_security, workflow, scenarios, agents, faq, simulation, evaluation.

**Usage:** Embed `content` (+ optional `title`) with your vector store; filter by `category` or `keywords` for domain-specific retrieval.

## Telemetry CSV (`ran_multi_agent_telemetry_70k.csv`)

Synthetic multi-agent RAN telemetry aligned with project feature requirements and 3GPP/O-RAN measurement ranges.

### Scenario distribution (~70k rows)

- `normal_operation` (~55%)
- `jamming_attack`, `compromised_ai_agent`, `adversarial_mobility_attack`, `massive_ue_mobility`, `high_congestion`

### Column groups

- **Identity:** timestamp, ue_id, cell_id, gnb_id, neighbor_cell_id, agent_id, agent_type, ric_component, rrc_state, scenario_type
- **Radio (TS 38.215):** rsrp_dbm, rsrq_db, sinr_db, cqi, rssi_dbm, beam_index, neighbor_rsrp_dbm, interference_dbm
- **Mobility:** ue_speed_kmh, ue_direction_deg, ho_history_count
- **Network:** prb_utilization_pct, cell_load_pct, dl/ul_throughput_mbps, latency_ms, packet_loss_pct, spectral_efficiency_bps_hz
- **Security / AI:** trust_score, attack_probability, anomaly_score, agent_confidence, consensus_vote_pct, policy_compliance_score
- **Labels / actions:** ho_required, target_cell_id, threat_type, allocated_prb_count, rrc_action, ho_success, energy_saving_mode, pqc_key_exchange_status

### References

- 3GPP TS 38.300, 38.331, 38.215, 38.214, 28.530, 28.105, 23.288, 33.501
- 3GPP TR 38.817, TR 23.700-80
- O-RAN Near-RT RIC, E2GAP, xApp/rApp
- NBUC problem statement and project documentation

## MATLAB-Generated RAN Maps (`sinr_map.csv`, `rsrp_map.csv`)

3GPP-compliant PHY-layer maps generated via MATLAB 5G Toolbox, 7-cell hexagonal layout, 1000m × 1000m grid. 
Used for 3GPP-traceable validation alongside the synthetic telemetry above.

### `rsrp_map.csv` - 10,201 rows x 9 columns
- `x`, `y` - grid position (m)
- `rsrp_cell0` to `rsrp_cell6` - RSRP (dBm) from each of the 7 cells at that point

### `sinr_map.csv` - 10,201 rows x 11 columns
- `x`, `y` - grid position (m)
- `sinr_cell0` – `sinr_cell6` - SINR (dB) from each of the 7 cells at that point
- `serving_cell` - index (0-6) of the strongest serving cell
- `is_handover_zone` - binary flag marking handover decision zones (approx 25% of grid points)
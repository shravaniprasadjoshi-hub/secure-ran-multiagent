# Project 03 Synthetic Datasets

Datasets for **Secure Multi-Agent AI Framework for RAN Control Loops** (NBUC / Nokia).

## Files

| File | Description |
|------|-------------|
| `rag_corpus.jsonl` | RAG knowledge chunks (3GPP, O-RAN, MARL, security, workflows) |
| `ran_multi_agent_telemetry_70k.csv` | 70,000 rows × 42 columns of synthetic RAN telemetry |
| `dataset_metadata.json` | Schema, scenario distribution, generation metadata |

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

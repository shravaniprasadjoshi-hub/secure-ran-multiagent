# Agent Reference

## Mobility Agent
- **Function:** Handover optimization (TS 38.331 Events A2/A3)
- **Model:** Gradient Boosting Classifier
- **Target:** `ho_required` (0/1)
- **RRC Action:** `handover_trigger` or `none`
- **Latency Target:** <50 ms

## Security Agent
- **Function:** Attack/threat detection
- **Model:** Random Forest Classifier
- **Target:** `threat_label` (benign/malicious)
- **Threats:** jamming, AI poisoning, rogue xApp, adversarial input, Sybil
- **RRC Action:** `security_mitigation`

## Resource Agent
- **Function:** PRB/spectrum allocation (TS 38.214)
- **Model:** Gradient Boosting Regressor
- **Target:** `allocated_prb_count`
- **RRC Action:** `rrc_reconfiguration`

## Energy Agent
- **Function:** Power/energy optimization
- **Model:** Random Forest Classifier
- **Target:** `energy_label` (normal/mimo_reduce/deep_sleep)
- **RRC Action:** `beam_switch` when not normal

## Trust Agent
- **Function:** Dynamic trust scoring (GNN concept → MLP implementation)
- **Model:** MLP Regressor
- **Target:** `trust_score` (0–1)
- **Threshold:** 0.8 for consensus participation

## Beamforming Agent
- **Function:** Beam index prediction (0–63 SSB beams)
- **Model:** Gradient Boosting Regressor
- **Target:** `beam_index`

## QoS Agent
- **Function:** SLA assurance, HO success prediction
- **Model:** Random Forest Classifier
- **Target:** `ho_success`
- **RRC Action:** `bearer_adaptation` on failure

## Policy Agent
- **Function:** 3GPP policy compliance validation
- **Model:** Gradient Boosting Classifier
- **Target:** `rrc_action`
- **Validates:** RRC constraints per TS 38.331

## Consensus Integration

All agents (except Trust as scorer) propose actions → ConsensusEngine validates → Final RRC action executed or rejected.

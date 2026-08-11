# Architecture Reference

## Multi-Agent Layer

Eight distributed agents, each with a dedicated ML model:

```
                    ┌─────────────────┐
                    │  Coordinator    │
                    │  + Consensus    │
                    └────────┬────────┘
         ┌──────────┬────────┼────────┬──────────┐
         ▼          ▼        ▼        ▼          ▼
    Mobility    Security  Resource  Energy    Trust
     Agent       Agent     Agent    Agent     Agent
         │          │        │        │          │
         ▼          ▼        ▼        ▼          ▼
    Beamforming   QoS     Policy
     Agent       Agent    Agent
```

## O-RAN Mapping

| Framework Component | O-RAN Mapping |
|--------------------|---------------|
| AI Agents | xApps / rApps |
| Coordinator + Consensus | Near-RT RIC |
| Policy training | Non-RT RIC |
| Security Layer | SMO Security |
| Telemetry collection | E2 Nodes |

## 3GPP Control Loop Alignment

- **RRC (TS 38.331):** Mobility agent drives HO triggers; policy agent validates RRC constraints
- **Measurements (TS 38.215):** RSRP, RSRQ, SINR, CQI as agent input features
- **AI Management (TS 28.530):** Model lifecycle — train (pipeline) → deploy (joblib) → infer (coordinator)
- **Analytics (TS 23.288):** NWDAF-style accuracy metrics in evaluation

## Security Architecture

1. **Trust Engine** — Bayesian trust scoring per agent
2. **Anomaly Detection** — Autoencoder-based (security agent features)
3. **Consensus** — BFT + weighted trust voting
4. **PQC** — Kyber/Dilithium status tracked in telemetry (`pqc_key_exchange_status`)

## Data Flow

```
UE Measurements → gNB/O-DU → Near-RT RIC → AI Agents → Trust Validation
    → Consensus Engine → RRC Decision → Execution → Feedback → Dashboard
```

## File Mapping

| Component | Source File |
|-----------|-------------|
| Data loading | `src/data/loader.py` |
| Train/val/test split | `src/data/preprocessor.py` |
| Agent training | `src/training/trainer.py` |
| Consensus | `src/orchestration/consensus.py` |
| Coordinator | `src/orchestration/coordinator.py` |
| Digital twin | `src/digital_twin/twin_engine.py` |
| Chatbot | `src/chatbot/rag_chatbot.py` |
| Plots | `src/visualization/plots.py` |
| Dashboard | `dashboard/app.py` |

# Secure Multi-Agent AI Framework for RAN Control Loops

Nokia Bangalore University Collaboration project — secure, distributed multi-agent AI system for autonomous 6G RAN control loops (RRC decisions), resilient to adversarial/Byzantine agents.

## Team
- Shashank — RAN environment & simulation realism
- Shreya — MAPPO agent implementation, security
- Shravani — Agents, coordination/consensus
- Shloka — Coordination, security

## Project Structure

```
env/            # RAN simulation environment (gym-style)
agents/         # MAPPO multi-agent RL implementation
coordination/   # Consensus & trust mechanisms
security/       # Anomaly detection & Byzantine agent handling
training/       # Training loop, config, evaluation
results/        # Logs, plots, checkpoints
tests/          # Unit tests
notebooks/      # Experiments & paper figures
```

## Methodology Phases
1. **Environment Setup** (Weeks 1–3) — RAN scenario simulation, RRC decision space
2. **Multi-Agent Framework** (Weeks 4–7) — Agent observation/action loop, consensus
3. **Adversarial Modeling** (Weeks 8–11) — Byzantine agent injection & resilience testing
4. **Security Layer** (Weeks 12–15) — Anomaly detection, policy arbitration, quantum-safe comms
5. **Evaluation & Paper** (Weeks 16–18) — Metrics, baseline comparison, write-up

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status
🚧 Project kickoff phase — environment and agent skeletons in progress.

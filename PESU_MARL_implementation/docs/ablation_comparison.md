# Ablation Comparison Table

Secure Multi-Agent AI Framework for RAN Control Loops - NBUC / Nokia project.
Full methodology and honest findings: `docs/experiment_3_findings.md`.

## Exp 1 vs Exp 2 vs Exp 3

| | **Exp 1: Nokia's baseline** | **Exp 2: MAPPO only** | **Exp 3: MAPPO + security** |
|---|---|---|---|
| **Approach** | Supervised ML, 8 sklearn agents | MARL (7 agents, PPO) | MARL + Byzantine/anomaly/trust/consensus |
| **Paradigm** | Per-row classification/regression on static labels | Sequential RL, reward-driven | Sequential RL, reward-driven, under active attack |
| **Data** | 70k-row Nokia telemetry (48,999/10,500/10,501 split) | Live analytic RF model, JRHT-derived reward | Same as Exp 2 + Byzantine fault injection |
| **Primary metric** | F1 / R² per agent (not reward-based) | Episode reward | Episode reward |
| **Result** | See per-agent table below | 867 → 1791 avg reward (early → late, 50-ep windows); final eval reward 962, 386 successful handovers, 0 RLF | -2520 → -259 avg reward (early → late); **not converged** within 1000 episodes |
| **Status** | Complete | Complete, converged cleanly | Complete, trending correctly, gap remains |

## Exp 1 detail: per-agent baseline performance

| Agent | Task | Test F1 / R² |
|---|---|---|
| mobility_agent | classification | F1 0.903 |
| security_agent | classification | F1 0.994 |
| resource_agent | regression | R² 0.406 |
| energy_agent | classification | F1 1.000 |
| trust_agent | regression | R² 0.558 |
| beamforming_agent | regression | R² -0.001 (near-unpredictable from given features) |
| qos_agent | classification | F1 0.894 |
| policy_agent | classification | F1 0.837 |

## Exp 3 detail: security metrics (1000 episodes)

| Metric | First 50 episodes | Last 50 episodes |
|---|---|---|
| `total_reward` | avg -2520 | avg -259 |
| `avg_trust_score` | ~0.44 | ~0.58 |
| `false_positive_rate` | ~0.25 | ~0.25 |
| `byzantine_detection_rate` | ~0.14 | ~0.14 |
| `consensus_rate` (diagnostic only) | ~0.68 | ~0.68 |

## Inference from the table

- **Exp 1 and Exp 2/3 are not directly comparable** - different paradigms (static supervised prediction vs. sequential policy optimization). Exp 1's F1/R² and Exp 2/3's reward measure fundamentally different things. Present side by side, not as a single ranked score.
- **Exp 2 is the clean result**: MAPPO converges reliably with zero radio link failures at eval time.
- **Exp 3 shows security has a real, measurable cost**: reward improves 88% relative to its own starting point but hasn't caught up to Exp 2's unsecured performance in the same episode budget. This is because quarantined agents skip PPO updates that step (deliberate, correct behavior - see findings doc), which throttles learning signal during the (currently noisy, exploration-heavy) early-training period.
- **Exp 4 (PQC)**: not started, remains a stretch goal.
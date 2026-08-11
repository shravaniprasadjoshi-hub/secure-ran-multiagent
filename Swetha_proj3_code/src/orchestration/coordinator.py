from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.orchestration.consensus import AgentProposal, ConsensusEngine


class MultiAgentCoordinator:
    """Orchestrates agent inference and consensus-based RRC decisions."""

    def __init__(self, artifacts: dict, consensus_cfg: dict):
        self.artifacts = artifacts
        self.consensus = ConsensusEngine(
            majority_threshold=consensus_cfg.get("majority_threshold", 0.70),
            trust_threshold=consensus_cfg.get("trust_threshold", 0.80),
            confidence_threshold=consensus_cfg.get("confidence_threshold", 0.85),
        )

    def _predict_row(self, agent_name: str, row: pd.Series) -> tuple[str | float, float]:
        art = self.artifacts[agent_name]
        from src.data.preprocessor import encode_features

        X, _ = encode_features(pd.DataFrame([row]), art["features"], fit=False, scaler=art["scaler"])
        pred = art["model"].predict(X)[0]
        if art["task"] == "classification" and art["label_encoder"] is not None:
            label = art["label_encoder"].inverse_transform([int(pred)])[0]
            conf = float(max(art["model"].predict_proba(X)[0])) if hasattr(art["model"], "predict_proba") else 0.85
            return label, conf
        return float(pred), 0.85

    def run_inference(self, row: pd.Series) -> dict:
        proposals = []
        action_map = {
            "mobility_agent": ("ho_required", lambda v: "handover_trigger" if str(v) in ("1", "True", "true") else "none"),
            "security_agent": ("threat_label", lambda v: "security_mitigation" if v == "malicious" else "none"),
            "resource_agent": ("allocated_prb_count", lambda v: "rrc_reconfiguration"),
            "energy_agent": ("energy_label", lambda v: "beam_switch" if v != "normal" else "none"),
            "qos_agent": ("ho_success", lambda v: "bearer_adaptation" if str(v) == "0" else "none"),
            "policy_agent": ("rrc_action", lambda v: str(v)),
        }
        for agent, (target_key, action_fn) in action_map.items():
            if agent not in self.artifacts:
                continue
            pred, conf = self._predict_row(agent, row)
            action = action_fn(pred)
            proposals.append(
                AgentProposal(
                    agent_id=agent,
                    agent_type=agent,
                    action=action,
                    confidence=conf,
                    trust_score=float(row.get("trust_score", 0.8)),
                    metadata={"prediction": pred},
                )
            )
        consensus = self.consensus.validate(proposals)
        return {"proposals": proposals, "consensus": consensus}

    def evaluate_batch(self, df: pd.DataFrame, sample_size: int = 500) -> dict:
        sample = df.sample(min(sample_size, len(df)), random_state=42)
        accepted = 0
        actions = []
        for _, row in sample.iterrows():
            r = self.run_inference(row)
            if r["consensus"].accepted:
                accepted += 1
            actions.append(r["consensus"].final_action)
        return {
            "samples": len(sample),
            "consensus_accept_rate": accepted / len(sample),
            "action_distribution": pd.Series(actions).value_counts().to_dict(),
        }


def save_coordinator_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

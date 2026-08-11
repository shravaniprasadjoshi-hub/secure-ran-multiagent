from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class RANDigitalTwin:
    """Digital twin simulating multi-cell RAN state and agent interventions."""

    def __init__(self, num_cells: int = 48, num_gnbs: int = 12, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_cells = num_cells
        self.num_gnbs = num_gnbs
        self.state: pd.DataFrame | None = None
        self.history: list[dict] = []

    def initialize(self) -> pd.DataFrame:
        cells = np.arange(1, self.num_cells + 1)
        self.state = pd.DataFrame({
            "cell_id": [f"CELL_{c:03d}" for c in cells],
            "gnb_id": [f"GNB_{((c - 1) % self.num_gnbs) + 1:02d}" for c in cells],
            "rsrp_dbm": self.rng.normal(-95, 8, self.num_cells),
            "sinr_db": self.rng.normal(12, 5, self.num_cells),
            "prb_util_pct": self.rng.uniform(20, 80, self.num_cells),
            "active_ues": self.rng.integers(10, 200, self.num_cells),
            "trust_score": self.rng.uniform(0.7, 0.99, self.num_cells),
            "attack_flag": np.zeros(self.num_cells, dtype=int),
            "ho_count": np.zeros(self.num_cells, dtype=int),
            "throughput_mbps": self.rng.uniform(50, 300, self.num_cells),
            "latency_ms": self.rng.uniform(5, 40, self.num_cells),
        })
        return self.state.copy()

    def inject_attack(self, attack_type: str = "jamming", cell_indices: list[int] | None = None) -> None:
        if self.state is None:
            self.initialize()
        idx = cell_indices or self.rng.choice(self.num_cells, size=3, replace=False).tolist()
        for i in idx:
            self.state.loc[i, "attack_flag"] = 1
            if attack_type == "jamming":
                self.state.loc[i, "sinr_db"] -= self.rng.uniform(8, 15)
                self.state.loc[i, "rsrp_dbm"] -= self.rng.uniform(5, 12)
            elif attack_type == "congestion":
                self.state.loc[i, "prb_util_pct"] = self.rng.uniform(85, 99)
                self.state.loc[i, "latency_ms"] = self.rng.uniform(60, 120)

    def step(self, agent_actions: dict | None = None) -> dict:
        if self.state is None:
            self.initialize()
        agent_actions = agent_actions or {}
        ho_triggers = 0
        mitigations = 0
        for i in range(self.num_cells):
            drift = self.rng.normal(0, 1.5)
            self.state.loc[i, "rsrp_dbm"] += drift * 0.3
            self.state.loc[i, "sinr_db"] += drift * 0.2
            self.state.loc[i, "prb_util_pct"] = np.clip(self.state.loc[i, "prb_util_pct"] + drift, 5, 100)
            self.state.loc[i, "throughput_mbps"] = np.clip(
                self.state.loc[i, "throughput_mbps"] + self.state.loc[i, "sinr_db"] * 0.5, 10, 500
            )
            if self.state.loc[i, "attack_flag"] and agent_actions.get("security_mitigation"):
                self.state.loc[i, "sinr_db"] += 3
                self.state.loc[i, "attack_flag"] = 0
                mitigations += 1
            if agent_actions.get("handover_trigger") and self.state.loc[i, "sinr_db"] < 5:
                self.state.loc[i, "ho_count"] += 1
                ho_triggers += 1
        snapshot = {
            "mean_sinr": float(self.state["sinr_db"].mean()),
            "mean_rsrp": float(self.state["rsrp_dbm"].mean()),
            "mean_prb": float(self.state["prb_util_pct"].mean()),
            "mean_trust": float(self.state["trust_score"].mean()),
            "total_ho": int(self.state["ho_count"].sum()),
            "attacks_active": int(self.state["attack_flag"].sum()),
            "ho_triggers": ho_triggers,
            "mitigations": mitigations,
            "mean_throughput": float(self.state["throughput_mbps"].mean()),
            "mean_latency": float(self.state["latency_ms"].mean()),
        }
        self.history.append(snapshot)
        return snapshot

    def run_simulation(self, steps: int = 200, attack_at: int = 50) -> pd.DataFrame:
        self.initialize()
        for t in range(steps):
            if t == attack_at:
                self.inject_attack("jamming")
            actions = {"handover_trigger": True, "security_mitigation": t > attack_at}
            self.step(actions)
        return pd.DataFrame(self.history)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.state is not None:
            self.state.to_csv(output_dir / "twin_cell_state.csv", index=False)
        if self.history:
            pd.DataFrame(self.history).to_csv(output_dir / "twin_simulation_history.csv", index=False)
        meta = {"num_cells": self.num_cells, "num_gnbs": self.num_gnbs, "steps": len(self.history)}
        (output_dir / "twin_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

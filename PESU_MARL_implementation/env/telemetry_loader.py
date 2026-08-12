"""
env/telemetry_loader.py: Loads Nokia's ran_multi_agent_telemetry_70k.csv
Owner: Shreyashree
Depends on: pandas, numpy
Used by:
  - training-scale bootstrap/pretraining for MAPPO (70k rows, scenario diversity)
  - security/anomaly_detector.py, security/byzantine.py via get_security_features() 
  - trust_score/attack_probability/anomaly_score are already labeled in data, no need to hand-craft them

# Schema confirmed against data/nokia_provided/dataset_metadata.json
# (70000 rows x 42 cols). Key facts:
#   - NO episode_id/trace_id column - sequential replay groups by ue_id,
#     sorted by timestamp (see iter_ue_traces)
#   - cell_id/gnb_id here are UNRELATED to our 7-cell hex layout in
#     env/cell.py - this is flat per-(UE,timestamp) telemetry, not a
#     topology. Don't map her cell_id onto our Cell objects.
#   - scenario_type is a real, useful filter: normal_operation, jamming_attack,
#     compromised_ai_agent, adversarial_mobility_attack, massive_ue_mobility,
#     high_congestion (see get_by_scenario)
# Isolated on purpose - does NOT modify ran_env.py, matches the two-dataset
# strategy (hers = scale + security features, MATLAB = 3GPP fidelity).
"""

import json
import os

import numpy as np
import pandas as pd

EXPECTED_SECURITY_COLS = ["trust_score", "attack_probability", "anomaly_score", "consensus_vote_pct"]
EXPECTED_HANDOVER_COLS = ["ho_required", "ho_success", "threat_type"]
SCENARIO_TYPES = [
    "normal_operation", "jamming_attack", "compromised_ai_agent",
    "adversarial_mobility_attack", "massive_ue_mobility", "high_congestion",
]


class TelemetryLoader:
    """Read-only wrapper around Nokia's telemetry CSV. data/nokia_provided/ stays read-only."""

    def __init__(self, csv_path: str, metadata_path: str = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"telemetry CSV not found at {csv_path}")
        self.df = pd.read_csv(csv_path)

        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        self._missing_security_cols = [c for c in EXPECTED_SECURITY_COLS if c not in self.df.columns]
        self._missing_handover_cols = [c for c in EXPECTED_HANDOVER_COLS if c not in self.df.columns]
        if self._missing_security_cols:
            print(f"[telemetry_loader] WARNING: missing expected security cols: {self._missing_security_cols}")
        if self._missing_handover_cols:
            print(f"[telemetry_loader] WARNING: missing expected handover cols: {self._missing_handover_cols}")

    def __len__(self):
        return len(self.df)

    @property
    def columns(self):
        return self.df.columns.tolist()

    # training-scale bootstrap

    def sample_batch(self, batch_size: int, seed: int = None):
        """Random rows - for augmenting/pretraining, not a sequential rollout."""
        return self.df.sample(n=batch_size, random_state=seed)

    def iter_ue_traces(self):
        """
        Groups rows by ue_id, sorted by timestamp - the real sequential
        replay unit in this dataset (there's no episode_id).
        Yields: (ue_id, DataFrame sorted by timestamp)
        """
        for ue_id, group in self.df.groupby("ue_id"):
            yield ue_id, group.sort_values("timestamp")

    def get_by_scenario(self, scenario_type: str):
        """Filter to one scenario, e.g. 'jamming_attack' or 'compromised_ai_agent' - useful for Byzantine/security eval."""
        if scenario_type not in SCENARIO_TYPES:
            print(f"[telemetry_loader] WARNING: '{scenario_type}' not in known SCENARIO_TYPES={SCENARIO_TYPES}")
        return self.df[self.df["scenario_type"] == scenario_type]

    def to_env_obs(self, df_subset: pd.DataFrame = None) -> np.ndarray:
        """
        Maps her columns onto our OBS_DIM=3 semantics (env/ran_env.py):
        [sinr_db, delta_RSRP_to_best_neighbor, NACK_density]
        Uses her real packet_loss_pct as the NACK proxy instead of our
        synthetic sigmoid in ran_env.py - real signal, prefer this wherever
        her data is available.
        """
        d = df_subset if df_subset is not None else self.df
        sinr = d["sinr_db"].to_numpy()
        delta_rsrp = (d["neighbor_rsrp_dbm"] - d["rsrp_dbm"]).to_numpy()
        nack_density = (d["packet_loss_pct"] / 100.0).to_numpy()
        return np.stack([sinr, delta_rsrp, nack_density], axis=-1).astype(np.float32)

    # security features

    def get_security_features(self, idx=None):
        """trust_score / attack_probability / anomaly_score / consensus_vote_pct, full df or one row."""
        available = [c for c in EXPECTED_SECURITY_COLS if c in self.df.columns]
        if not available:
            raise KeyError(f"none of {EXPECTED_SECURITY_COLS} present in telemetry CSV")
        subset = self.df[available]
        return subset.iloc[idx] if idx is not None else subset

    def get_handover_labels(self):
        """Ground-truth ho_required/ho_success/threat_type - for baseline/ablation comparison."""
        available = [c for c in EXPECTED_HANDOVER_COLS if c in self.df.columns]
        if not available:
            raise KeyError(f"none of {EXPECTED_HANDOVER_COLS} present in telemetry CSV")
        return self.df[available]

    # schema check

    def validate_schema(self, metadata_path: str = None) -> dict:
        """Run this against the real dataset_metadata.json - prints a diff report."""
        meta = self.metadata
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)

        report = {
            "n_rows": len(self.df),
            "n_cols": len(self.df.columns),
            "missing_security_cols": self._missing_security_cols,
            "missing_handover_cols": self._missing_handover_cols,
        }
        if meta and "telemetry_csv" in meta and "columns" in meta["telemetry_csv"]:
            expected = set(meta["telemetry_csv"]["columns"])
            actual = set(self.df.columns)
            report["extra_in_csv_not_in_metadata"] = list(actual - expected)
            report["in_metadata_missing_from_csv"] = list(expected - actual)
        return report
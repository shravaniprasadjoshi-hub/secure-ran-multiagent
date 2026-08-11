"""
env/replay_loader.py: Replay mode - RSRP/SINR from MATLAB-generated grids
Owner: Shreyashree
Depends on: none (pandas, numpy)
Used by: ran_env.py can swap ChannelModel for this at eval/validation time

# CRITICAL: this reads pre-generated MATLAB 5G-Toolbox output CSVs
# (rsrp_map.csv, sinr_map.csv) - 7-cell hex layout, 1000x1000m grid, 10m spacing (101x101 = 10201 points), seed=42
# 3GPP-traceable, used for # final results/publication claims
# Does NOT touch the live analytic channel.py model - that stays independent for training-time speed
# sinr_map.csv also carries MATLAB's own serving_cell + is_handover_zone # labels - exposed here for validating our agents' decisions against the # MATLAB-computed ground truth.
"""

import numpy as np
import pandas as pd

GRID_SPACING_M = 10.0 # DONT TOUCH - must match MATLAB export
GRID_MIN_M = 0.0
GRID_MAX_M = 1000.0
NUM_CELLS = 7


class ReplayChannelModel:
    """
    Drop-in replacement for channel.ChannelModel's query surface, backed by
    static MATLAB grids instead of live path-loss computation. No RNG, no
    jamming injection (MATLAB grids are pre-baked) - purely a lookup table.
    """

    def __init__(self, rsrp_path: str = "rsrp_map.csv", sinr_path: str = "sinr_map.csv"):
        rsrp_df = pd.read_csv(rsrp_path)
        sinr_df = pd.read_csv(sinr_path)

        self._validate(rsrp_df, sinr_df)

        # index by (x_idx, y_idx) for O(1) nearest-grid-point lookup
        n = int((GRID_MAX_M - GRID_MIN_M) / GRID_SPACING_M) + 1  # 101
        self.grid_n = n

        rsrp_cols = [f"rsrp_cell{i}" for i in range(NUM_CELLS)]
        sinr_cols = [f"sinr_cell{i}" for i in range(NUM_CELLS)]

        self._rsrp_grid = np.zeros((n, n, NUM_CELLS), dtype=np.float32)
        self._sinr_grid = np.zeros((n, n, NUM_CELLS), dtype=np.float32)
        self._serving_grid = np.zeros((n, n), dtype=np.int32)
        self._handover_zone_grid = np.zeros((n, n), dtype=np.int32)

        xi = (rsrp_df["x"].to_numpy() / GRID_SPACING_M).astype(int)
        yi = (rsrp_df["y"].to_numpy() / GRID_SPACING_M).astype(int)
        self._rsrp_grid[xi, yi] = rsrp_df[rsrp_cols].to_numpy()

        xi2 = (sinr_df["x"].to_numpy() / GRID_SPACING_M).astype(int)
        yi2 = (sinr_df["y"].to_numpy() / GRID_SPACING_M).astype(int)
        self._sinr_grid[xi2, yi2] = sinr_df[sinr_cols].to_numpy()
        self._serving_grid[xi2, yi2] = sinr_df["serving_cell"].to_numpy()
        self._handover_zone_grid[xi2, yi2] = sinr_df["is_handover_zone"].to_numpy()

    @staticmethod
    def _validate(rsrp_df, sinr_df):
        expected_rows = 101 * 101
        if len(rsrp_df) != expected_rows or len(sinr_df) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows (101x101 @ 10m spacing), "
                f"got rsrp={len(rsrp_df)}, sinr={len(sinr_df)} - grid mismatch, "
                f"check the MATLAB export config before trusting this replay data."
            )

    def _nearest_idx(self, pos):
        x, y = pos
        xi = int(round(np.clip(x, GRID_MIN_M, GRID_MAX_M) / GRID_SPACING_M))
        yi = int(round(np.clip(y, GRID_MIN_M, GRID_MAX_M) / GRID_SPACING_M))
        return xi, yi

    # same-shaped query surface as ChannelModel

    def rsrp_all_cells_dbm(self, ue_pos, cell_positions=None):
        """cell_positions arg accepted for interface parity, unused - grid already encodes cell layout."""
        xi, yi = self._nearest_idx(ue_pos)
        return self._rsrp_grid[xi, yi].tolist()

    def sinr_all_cells_db(self, ue_pos, rsrp_list_dbm=None, t: float = 0.0):
        """rsrp_list_dbm/t accepted for interface parity, unused - MATLAB SINR is precomputed, not live."""
        xi, yi = self._nearest_idx(ue_pos)
        return self._sinr_grid[xi, yi].tolist()

    def sinr_db(self, ue_pos, serving_idx: int, rsrp_list_dbm=None, t: float = 0.0):
        xi, yi = self._nearest_idx(ue_pos)
        return float(self._sinr_grid[xi, yi, serving_idx])

    # MATLAB ground-truth labels, for validating agent decisions

    def matlab_serving_cell(self, ue_pos) -> int:
        xi, yi = self._nearest_idx(ue_pos)
        return int(self._serving_grid[xi, yi])

    def is_matlab_handover_zone(self, ue_pos) -> bool:
        xi, yi = self._nearest_idx(ue_pos)
        return bool(self._handover_zone_grid[xi, yi])
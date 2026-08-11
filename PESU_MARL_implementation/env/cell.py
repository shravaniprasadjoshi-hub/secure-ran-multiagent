"""
env/cell.py: RAN Cell - hex-layout position + serving state
Owner: Shreyashree
Depends on: none
Used by: ran_env.py, ue.py (position lookups only)
"""

import numpy as np

# hex layout
GRID_SIZE_M = 1000.0 # DONT TOUCH - must match channel.py / MATLAB grid
NUM_CELLS = 7 # DONT TOUCH - 1 center + 6 ring, matches MATLAB layout
INTER_SITE_DISTANCE_M = 250.0  # tune for coverage overlap vs GRID_SIZE_M


def hex_cell_positions(center=(GRID_SIZE_M / 2, GRID_SIZE_M / 2),
                        isd: float = INTER_SITE_DISTANCE_M):
    """
    7-cell hex layout: 1 center cell + 6 surrounding at 60-degree spacing.
    Returns list[(x, y)], index 0 = center cell (matches MATLAB serving_cell
    indexing convention in sinr_map.csv).
    """
    positions = [center]
    for k in range(6):
        angle = np.deg2rad(60 * k)
        x = center[0] + isd * np.cos(angle)
        y = center[1] + isd * np.sin(angle)
        positions.append((x, y))
    return positions


class Cell:
    """
    One RAN cell/sector. Holds its own position + tracks which UEs it is
    currently serving. Capacity/load used for reward shaping (overload penalty).
    """

    def __init__(self, cell_id: int, x: float, y: float,
                 max_capacity: int = 32, tx_power_dbm: float = 43.0):
        self.cell_id = cell_id
        self.x = x
        self.y = y
        self.max_capacity = max_capacity
        self.tx_power_dbm = tx_power_dbm
        self.connected_ue_ids = set()

        # security/anomaly_detector.py reads this - so DONT rename
        self.is_compromised = False

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def load(self) -> int:
        return len(self.connected_ue_ids)

    @property
    def load_ratio(self) -> float:
        return self.load / self.max_capacity if self.max_capacity > 0 else 0.0

    @property
    def is_overloaded(self) -> bool:
        return self.load >= self.max_capacity

    def attach(self, ue_id: int) -> bool:
        """Attach a UE if capacity allows. Returns success."""
        if self.is_overloaded:
            return False
        self.connected_ue_ids.add(ue_id)
        return True

    def detach(self, ue_id: int):
        self.connected_ue_ids.discard(ue_id)

    def reset(self):
        self.connected_ue_ids = set()
        self.is_compromised = False

    def __repr__(self):
        return f"Cell(id={self.cell_id}, pos=({self.x:.0f},{self.y:.0f}), load={self.load}/{self.max_capacity})"


def build_hex_layout(max_capacity: int = 32) -> list:
    """Convenience factory: returns list[Cell] for the standard 7-cell layout."""
    positions = hex_cell_positions()
    return [
        Cell(cell_id=i, x=pos[0], y=pos[1], max_capacity=max_capacity)
        for i, pos in enumerate(positions)
    ]
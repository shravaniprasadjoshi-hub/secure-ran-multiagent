"""
env/ue.py: User Equipment - mobility model + measurement history
Owner: Shreyashree
Depends on: channel.py (for RF queries), cell.py (for position/attach)
Used by: ran_env.py
"""

from collections import deque

import numpy as np

from env.cell import GRID_SIZE_M

# mobility
DEFAULT_SPEED_MPS = 1.4 # ~ pedestrian walking speed
MEASUREMENT_HISTORY_LEN = 10  # window used for TTT-style handover hysteresis (JRHT baseline)


class UE:
    """
    One mobile user. Random-waypoint mobility within the grid, plus a rolling
    measurement history (RSRP/SINR per cell) that agents/env use for handover
    decisions - mirrors the JRHT baseline's time-to-trigger windowing.
    """

    def __init__(self, ue_id: int, x: float, y: float,
                 speed_mps: float = DEFAULT_SPEED_MPS, rng: np.random.Generator = None):
        self.ue_id = ue_id
        self.x = x
        self.y = y
        self.speed_mps = speed_mps
        self.rng = rng if rng is not None else np.random.default_rng(ue_id)

        self._heading = self.rng.uniform(0, 2 * np.pi)
        self._waypoint_ticks_left = 0

        self.serving_cell_id = None
        self.rsrp_history = deque(maxlen=MEASUREMENT_HISTORY_LEN)  # list[float] per step
        self.sinr_history = deque(maxlen=MEASUREMENT_HISTORY_LEN)

        # security/anomaly_detector.py flags this - don't rename
        self.is_under_attack = False

    @property
    def position(self):
        return (self.x, self.y)

    # mobility

    def _pick_new_waypoint(self):
        self._heading = self.rng.uniform(0, 2 * np.pi)
        self._waypoint_ticks_left = self.rng.integers(5, 20)

    def step_mobility(self, dt: float = 1.0, grid_size: float = GRID_SIZE_M):
        """Random-waypoint move, clamped + reflected at grid boundary."""
        if self._waypoint_ticks_left <= 0:
            self._pick_new_waypoint()
        self._waypoint_ticks_left -= 1

        dx = self.speed_mps * dt * np.cos(self._heading)
        dy = self.speed_mps * dt * np.sin(self._heading)
        new_x = self.x + dx
        new_y = self.y + dy

        # reflect off boundary instead of clamping, so UE keeps moving
        if new_x < 0 or new_x > grid_size:
            self._heading = np.pi - self._heading
            new_x = np.clip(new_x, 0, grid_size)
        if new_y < 0 or new_y > grid_size:
            self._heading = -self._heading
            new_y = np.clip(new_y, 0, grid_size)

        self.x, self.y = new_x, new_y

    # measurement

    def measure(self, channel_model, cell_positions, t: float = 0.0):
        """
        Queries the channel model for RSRP/SINR against every cell,
        pushes onto rolling history. Returns (rsrp_list, sinr_list).
        """
        rsrp_list = channel_model.rsrp_all_cells_dbm(self.position, cell_positions)
        sinr_list = channel_model.sinr_all_cells_db(self.position, rsrp_list, t=t)
        self.rsrp_history.append(rsrp_list)
        self.sinr_history.append(sinr_list)
        return rsrp_list, sinr_list

    def avg_rsrp_per_cell(self):
        """Mean RSRP per cell over the measurement window - smooths jamming spikes."""
        if not self.rsrp_history:
            return None
        return np.mean(np.array(self.rsrp_history), axis=0).tolist()

    def avg_sinr_per_cell(self):
        if not self.sinr_history:
            return None
        return np.mean(np.array(self.sinr_history), axis=0).tolist()

    # serving cell

    def attach_to(self, cell_id: int):
        self.serving_cell_id = cell_id

    def reset(self, x: float, y: float):
        self.x, self.y = x, y
        self.serving_cell_id = None
        self.rsrp_history.clear()
        self.sinr_history.clear()
        self.is_under_attack = False

    def __repr__(self):
        return f"UE(id={self.ue_id}, pos=({self.x:.0f},{self.y:.0f}), serving={self.serving_cell_id})"
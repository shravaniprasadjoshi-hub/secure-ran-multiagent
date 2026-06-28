"""
Cell/gNB model.
Owner: Shashank
TODO: Define cell state, coverage, capacity.
"""

import numpy as np

# Hexagonal layout constants
INTER_SITE_DISTANCE = 500.0  # meters between cell centers
DEFAULT_TX_POWER_DBM = 43.0  # typical macro cell ~20W
DEFAULT_FREQ_GHZ = 3.5       # 5G n78 band
DEFAULT_BANDWIDTH_MHZ = 100
MAX_UES_PER_CELL = 20        # capacity for load calculation


class Cell:
    """
    Represents a 5G gNB (base station) in the network.
    Cells are arranged in a 7-cell hexagonal layout:
    
              [1]
          [2]     [3]
              [0]
          [4]     [5]
              [6]
    """

    def __init__(self, cell_id,
                 tx_power_dbm=DEFAULT_TX_POWER_DBM,
                 freq_ghz=DEFAULT_FREQ_GHZ,
                 bandwidth_mhz=DEFAULT_BANDWIDTH_MHZ):
        self.cell_id = cell_id
        self.tx_power = tx_power_dbm           # dBm
        self.freq = freq_ghz                   # GHz
        self.bandwidth = bandwidth_mhz         # MHz
        self.position = self._compute_position(cell_id)
        self.load = 0.0                        # 0.0 to 1.0
        self.connected_ues = []                # list of UE ids

    def _compute_position(self, cell_id):
        """
        Returns (x, y) for the cell in a 7-cell hex layout.
        Cell 0 = center; cells 1-6 surround it at INTER_SITE_DISTANCE.
        """
        if cell_id == 0:
            return np.array([0.0, 0.0])

        # 6 surrounding cells at 60° intervals
        angle_deg = 60 * (cell_id - 1)
        angle_rad = np.deg2rad(angle_deg)
        x = INTER_SITE_DISTANCE * np.cos(angle_rad)
        y = INTER_SITE_DISTANCE * np.sin(angle_rad)
        return np.array([x, y])

    def update_load(self, all_ues):
        """Compute load based on connected UEs / capacity."""
        self.connected_ues = [ue.ue_id for ue in all_ues
                              if ue.serving_cell == self.cell_id]
        self.load = min(1.0, len(self.connected_ues) / MAX_UES_PER_CELL)

    def __repr__(self):
        return (f"Cell({self.cell_id}, pos={self.position.round(1)}, "
                f"load={self.load:.2f}, UEs={len(self.connected_ues)})")

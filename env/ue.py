"""
UE mobility + channel model.
Owner: Shashank
TODO: Implement UE movement and signal model.
"""

import numpy as np

# UE parameters
DEFAULT_SPEED_MS = 3.0   # m/s ~ pedestrian
MAX_SPEED_MS = 30.0      # m/s ~ vehicular
AREA_RADIUS = 800.0      # meters - UEs spawn within this radius of origin


class UE:
    """
    Represents a User Equipment (phone / IoT device).
    Uses a simple random-direction mobility model with occasional turns.
    """

    def __init__(self, ue_id, num_cells, speed=None):
        self.ue_id = ue_id
        self.num_cells = num_cells

        # spawn at random location inside coverage area
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, AREA_RADIUS)
        self.position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
        ])

        # mobility
        self.velocity = speed if speed is not None else \
                        np.random.uniform(1.0, DEFAULT_SPEED_MS)
        self.direction = np.random.uniform(0, 2 * np.pi)

        # network state (populated by env)
        self.serving_cell = np.random.randint(0, num_cells)
        self.rsrp = np.zeros(num_cells)   # dBm per cell
        self.interference = 0.0           # mW

    def move(self, dt=1.0):
        """
        Move UE forward; small chance to change direction.
        dt = timestep in seconds (default 1s).
        """
        # 10% chance to turn each step
        if np.random.rand() < 0.1:
            self.direction += np.random.uniform(-np.pi / 4, np.pi / 4)

        dx = self.velocity * dt * np.cos(self.direction)
        dy = self.velocity * dt * np.sin(self.direction)
        self.position += np.array([dx, dy])

        # bounce off boundary
        dist_from_origin = np.linalg.norm(self.position)
        if dist_from_origin > AREA_RADIUS:
            # reflect direction back toward center
            self.direction += np.pi
            self.position = self.position * (AREA_RADIUS / dist_from_origin)

    def __repr__(self):
        return (f"UE({self.ue_id}, pos={self.position.round(1)}, "
                f"v={self.velocity:.1f} m/s, cell={self.serving_cell})")

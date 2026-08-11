"""
env/channel.py: Analytic RF channel model (path loss + shadow fading + jamming -> RSRP/SINR)
Owner: Shreyashree
Depends on: none
Used by: cell.py, ue.py, ran_env.py (live simulation path)

# NOTE: MATLAB-generated rsrp_map.csv / sinr_map.csv are NOT used here.
# They stay isolated in replay_loader.py for 3GPP-traceable validation/results.
# This file is the fast, differentiable-friendly model used during RL rollouts.
# Same seed=42 convention as the MATLAB grids, same 7-cell hex / 1000x1000m layout.
"""

import numpy as np

# RF constants
CARRIER_FREQ_GHZ = 3.5 # mid-band placeholder, matches MATLAB 5G Toolbox config
TX_POWER_DBM = 43.0 # macro cell tx power
NOISE_FLOOR_DBM = -104.0 # thermal noise + NF, ~10 MHz BW
SHADOW_FADING_STD_DB = 8.0 # log-normal shadowing std dev
PATH_LOSS_D0_M = 1.0 # reference distance
PATH_LOSS_PL0_DB = 32.4 + 20 * np.log10(CARRIER_FREQ_GHZ)  # free-space ref @ d0, simplified 3GPP UMa-ish
PATH_LOSS_EXPONENT = 3.5 # NLOS urban macro exponent

GRID_SIZE_M = 1000.0 # DONT TOUCH - must match MATLAB grid (1000m x 1000m)
DEFAULT_SEED = 42 # DONT TOUCH - reproducibility convention across the project


# jamming

class JammingSource:
    """A single jammer: fixed position, active window, transmit power."""

    def __init__(self, x: float, y: float, power_dbm: float,
                 active_from: float = 0.0, active_to: float = float("inf")):
        self.x = x
        self.y = y
        self.power_dbm = power_dbm
        self.active_from = active_from
        self.active_to = active_to

    def is_active(self, t: float) -> bool:
        return self.active_from <= t <= self.active_to

    def interference_dbm(self, ue_x: float, ue_y: float, t: float, path_loss_fn):
        """Returns linear-mW-convertible dBm interference contribution at (ue_x, ue_y), or None if inactive."""
        if not self.is_active(t):
            return None
        d = max(np.hypot(ue_x - self.x, ue_y - self.y), PATH_LOSS_D0_M)
        pl_db = path_loss_fn(d)
        return self.power_dbm - pl_db


# channel model

class ChannelModel:
    """
    Stateless-ish RF model shared by all cells/UEs in one env instance.
    Holds its own RNG for shadow fading so rollouts are reproducible per-seed.
    """

    def __init__(self, seed: int = DEFAULT_SEED,
                 shadow_std_db: float = SHADOW_FADING_STD_DB):
        self.rng = np.random.default_rng(seed)
        self.shadow_std_db = shadow_std_db
        self.jammers = []  # list[JammingSource] - security/byzantine.py can inject these

    # jamming registry

    def add_jammer(self, jammer: JammingSource):
        self.jammers.append(jammer)

    def clear_jammers(self):
        self.jammers = []

    # core RF math

    @staticmethod
    def path_loss_db(distance_m: float) -> float:
        """Log-distance path loss model, floor at d0 to avoid log(0)."""
        d = max(distance_m, PATH_LOSS_D0_M)
        return PATH_LOSS_PL0_DB + 10 * PATH_LOSS_EXPONENT * np.log10(d / PATH_LOSS_D0_M)

    def shadow_fading_db(self) -> float:
        return self.rng.normal(0.0, self.shadow_std_db)

    def rsrp_dbm(self, ue_pos, cell_pos, tx_power_dbm: float = TX_POWER_DBM,
                 apply_shadow: bool = True) -> float:
        """RSRP at ue_pos from a single cell at cell_pos."""
        d = np.hypot(ue_pos[0] - cell_pos[0], ue_pos[1] - cell_pos[1])
        pl_db = self.path_loss_db(d)
        shadow = self.shadow_fading_db() if apply_shadow else 0.0
        return tx_power_dbm - pl_db + shadow

    def rsrp_all_cells_dbm(self, ue_pos, cell_positions, apply_shadow: bool = True):
        """RSRP from every cell, as a list aligned with cell_positions."""
        return [self.rsrp_dbm(ue_pos, cp, apply_shadow=apply_shadow) for cp in cell_positions]

    def jamming_interference_mw(self, ue_pos, t: float) -> float:
        """Sum of active jammer interference at ue_pos, in linear mW."""
        total_mw = 0.0
        for jammer in self.jammers:
            contrib_dbm = jammer.interference_dbm(ue_pos[0], ue_pos[1], t, self.path_loss_db)
            if contrib_dbm is not None:
                total_mw += 10 ** (contrib_dbm / 10)
        return total_mw

    def sinr_db(self, ue_pos, serving_idx: int, rsrp_list_dbm, t: float = 0.0) -> float:
        """
        SINR for the serving cell, treating all other cells' RSRP as interference,
        plus noise floor and any active jamming.
        rsrp_list_dbm: output of rsrp_all_cells_dbm, aligned by cell index.
        """
        signal_mw = 10 ** (rsrp_list_dbm[serving_idx] / 10)
        interference_mw = sum(
            10 ** (r / 10) for i, r in enumerate(rsrp_list_dbm) if i != serving_idx
        )
        noise_mw = 10 ** (NOISE_FLOOR_DBM / 10)
        jam_mw = self.jamming_interference_mw(ue_pos, t)

        denom = interference_mw + noise_mw + jam_mw
        sinr_linear = signal_mw / denom if denom > 0 else float("inf")
        return 10 * np.log10(sinr_linear) if sinr_linear > 0 else -float("inf")

    def sinr_all_cells_db(self, ue_pos, rsrp_list_dbm, t: float = 0.0):
        """SINR as if each cell were serving - used for handover candidate scoring."""
        return [
            self.sinr_db(ue_pos, i, rsrp_list_dbm, t) for i in range(len(rsrp_list_dbm))
        ]
"""
RSRP, interference calculations.
Owner: Shashank
TODO: Implement channel/signal quality calculations.
"""

import numpy as np

# Channel constants
NOISE_FLOOR_DBM = -104.0      # thermal noise for 100 MHz BW
SHADOWING_STD_DB = 4.0        # log-normal shadowing std deviation
MIN_DISTANCE_M = 1.0          # avoid log(0)


def path_loss_3gpp_uma_los(distance_m, freq_ghz):
    """
    3GPP TR 38.901 Urban Macro Line-of-Sight path loss model (simplified).
    Returns path loss in dB.
    
    Formula: PL = 28.0 + 22*log10(d) + 20*log10(f_GHz)
    """
    d = max(distance_m, MIN_DISTANCE_M)
    return 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(freq_ghz)


def compute_rsrp(ue, cell, add_shadowing=True):
    """
    Reference Signal Received Power (dBm) from `cell` to `ue`.
    
    RSRP = Tx_power - PathLoss - Shadowing
    """
    distance = np.linalg.norm(ue.position - cell.position)
    pl = path_loss_3gpp_uma_los(distance, cell.freq)

    shadowing = np.random.normal(0, SHADOWING_STD_DB) if add_shadowing else 0.0
    rsrp = cell.tx_power - pl - shadowing
    return float(rsrp)


def compute_interference(ue, all_cells):
    """
    Total interference (in mW) at UE from all non-serving cells.
    Sums power from all cells except the serving one.
    """
    interference_mw = 0.0
    for cell in all_cells:
        if cell.cell_id == ue.serving_cell:
            continue
        rsrp_dbm = compute_rsrp(ue, cell, add_shadowing=False)
        interference_mw += 10 ** (rsrp_dbm / 10.0)
    return float(interference_mw)


def compute_sinr(ue, all_cells):
    """
    Signal to Interference + Noise Ratio in dB at UE.
    
    SINR = Signal / (Interference + Noise)
    """
    serving_cell = all_cells[ue.serving_cell]
    signal_dbm = compute_rsrp(ue, serving_cell, add_shadowing=False)
    signal_mw = 10 ** (signal_dbm / 10.0)

    interference_mw = compute_interference(ue, all_cells)
    noise_mw = 10 ** (NOISE_FLOOR_DBM / 10.0)

    sinr_linear = signal_mw / (interference_mw + noise_mw)
    return float(10.0 * np.log10(sinr_linear))


def compute_throughput_mbps(sinr_db, bandwidth_mhz=100):
    """
    Shannon capacity estimate: C = BW * log2(1 + SINR_linear)
    """
    sinr_linear = 10 ** (sinr_db / 10.0)
    return bandwidth_mhz * np.log2(1 + sinr_linear)

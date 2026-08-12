"""
env/ran_env.py: PettingZoo ParallelEnv for multi-cell RRC handover control
Owner: Shreyashree
Depends on: cell.py, ue.py, channel.py
Used by: training/train.py (both hers and shared)

# State/action/reward design ported from JRHT baseline (Jamming-Resilient
# Handover Triggering paper, Section III-B):
#   - State: 3-feature minimalist vector (SINR, delta-RSRP-to-best-neighbor,
#     NACK density) - paper shows this is sufficient, vendor-agnostic, and
#     maps to real 3GPP MeasReport fields.
#   - Reward: RLF=-10, ping-pong=-5, successful HO=+10, healthy defer=+1
#     (paper's exact weights - shown robust to +/-30% scaling in their grid search)
#
# Extension beyond JRHT for our multi-agent setting:
#   - JRHT action space is binary {defer, trigger}. Ours is 3-way:
#     {defer, trigger->best neighbor, trigger->2nd-best neighbor}
#     This gives Byzantine/security research room later: a compromised
#     agent can pick a bad target even when triggering itself is justified.
#   - One agent per hex cell (7 agents total), each agent controls the
#     handover decision for the single UE currently attached to its cell.
"""

from functools import lru_cache

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from env.cell import build_hex_layout, GRID_SIZE_M, NUM_CELLS
from env.ue import UE
from env.channel import ChannelModel, DEFAULT_SEED
from env.replay_loader import ReplayChannelModel

# JRHT-derived reward weights
REWARD_RLF = -10.0
REWARD_PING_PONG = -5.0
REWARD_SUCCESSFUL_HO = 10.0
REWARD_HEALTHY_DEFER = 1.0

# thresholds
RLF_SINR_THRESHOLD_DB = -5.0 # below this while deferring => radio link failure
PING_PONG_WINDOW_STEPS = 5 # handover back to a cell visited within this window = ping-pong
NACK_SINR_PIVOT_DB = 0.0 # synthetic NACK proxy pivot (no live HARQ sim yet - see note below)
MAX_EPISODE_STEPS = 200

OBS_DIM = 3 # DONT TOUCH - SINR, delta_RSRP, NACK_density (matches JRHT's minimal state)
ACTION_DIM = 3 # DONT TOUCH - agent_manager.py / mappo_agent.py default action_dim


class RANEnv(ParallelEnv):
    """
    7 agents (one per hex cell). Each agent's UE is the single UE currently
    attached to that cell. Actions: 0=defer, 1=trigger to best neighbor,
    2=trigger to 2nd-best neighbor (by instantaneous SINR).

    # IMPORTANT: cell.py/ue.py/channel.py are the RF/mobility
    # source of truth. Extend those, don't hardcode RF math here.
    """

    metadata = {"name": "ran_env_v0"}

    def __init__(self, num_cells: int = NUM_CELLS, seed: int = DEFAULT_SEED,
                 max_steps: int = MAX_EPISODE_STEPS,
                 replay_rsrp_path: str = None, replay_sinr_path: str = None):
        """
        replay_rsrp_path / replay_sinr_path: if both given, reset() builds a
        ReplayChannelModel from the MATLAB grids instead of the live analytic
        ChannelModel - use this for final 3GPP-traceable evaluation only,
        not for training (no jamming injection, no RNG shadow fading).
        """
        super().__init__()
        self.num_cells = num_cells
        self.seed_value = seed
        self.max_steps = max_steps
        self.use_replay = bool(replay_rsrp_path and replay_sinr_path)
        self.replay_rsrp_path = replay_rsrp_path
        self.replay_sinr_path = replay_sinr_path

        self.possible_agents = [f"cell_{i}" for i in range(num_cells)]
        self.agents = list(self.possible_agents)

        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._act_space = spaces.Discrete(ACTION_DIM)

        self.cells = None
        self.ues = None
        self.channel = None
        self._t = 0
        self._rng = np.random.default_rng(seed)

        # per-UE recent-serving-cell history for ping-pong detection
        self._serving_history = {}

    # PettingZoo required API

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed
        self._rng = np.random.default_rng(self.seed_value)

        self.cells = build_hex_layout()
        if self.use_replay:
            self.channel = ReplayChannelModel(self.replay_rsrp_path, self.replay_sinr_path)
        else:
            self.channel = ChannelModel(seed=self.seed_value)
        self._t = 0

        # one UE per cell, spawned near that cell's position with jitter
        self.ues = []
        for cell in self.cells:
            jitter = self._rng.uniform(-60, 60, size=2)
            x = float(np.clip(cell.x + jitter[0], 0, GRID_SIZE_M))
            y = float(np.clip(cell.y + jitter[1], 0, GRID_SIZE_M))
            ue = UE(ue_id=cell.cell_id, x=x, y=y, rng=self._rng)
            ue.attach_to(cell.cell_id)
            cell.attach(ue.ue_id)
            self.ues.append(ue)

        self._serving_history = {ue.ue_id: [ue.serving_cell_id] for ue in self.ues}

        self.agents = list(self.possible_agents)
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict):
        self._t += 1
        cell_positions = [c.position for c in self.cells]

        rewards, terminations, truncations, infos = {}, {}, {}, {}

        for i, agent in enumerate(self.agents):
            ue = self.ues[i]
            ue.step_mobility()

        for i, agent in enumerate(self.agents):
            ue = self.ues[i]
            rsrp_list, sinr_list = ue.measure(self.channel, cell_positions, t=self._t)
            action = actions.get(agent, 0)
            reward, done = self._apply_action(ue, action, sinr_list)

            rewards[agent] = reward
            terminations[agent] = done
            truncations[agent] = self._t >= self.max_steps
            infos[agent] = {"serving_cell": ue.serving_cell_id, "sinr_db": sinr_list[ue.serving_cell_id]}

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}

        if all(truncations.values()) or all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        for cell in self.cells:
            print(cell)

    def close(self):
        pass

    # internals

    def _get_obs(self, agent_idx):
        """[SINR_serving_dB, delta_RSRP_to_best_neighbor_dB, NACK_density_proxy]"""
        ue = self.ues[agent_idx]
        avg_rsrp = ue.avg_rsrp_per_cell()
        avg_sinr = ue.avg_sinr_per_cell()
        if avg_rsrp is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        serving = ue.serving_cell_id
        sinr_serving = avg_sinr[serving]

        neighbor_rsrp = [r for i, r in enumerate(avg_rsrp) if i != serving]
        best_neighbor_rsrp = max(neighbor_rsrp) if neighbor_rsrp else avg_rsrp[serving]
        delta_rsrp = best_neighbor_rsrp - avg_rsrp[serving]

        # NACK proxy: no live HARQ simulation yet 
        # (swap this for real HARQ/BLER stats if/when the PHY sim exposes them)
        # Modeled as rising sharply as SINR drops below NACK_SINR_PIVOT_DB, clipped [0,1]
        nack_density = float(np.clip(1.0 / (1.0 + np.exp((sinr_serving - NACK_SINR_PIVOT_DB) / 3.0)), 0.0, 1.0))

        return np.array([sinr_serving, delta_rsrp, nack_density], dtype=np.float32)

    def _best_neighbors(self, ue, sinr_list, k=2):
        """Returns up to k neighbor cell indices ranked by current SINR, descending."""
        serving = ue.serving_cell_id
        ranked = sorted(
            (i for i in range(len(sinr_list)) if i != serving),
            key=lambda i: sinr_list[i], reverse=True,
        )
        return ranked[:k]

    def _apply_action(self, ue, action: int, sinr_list):
        """
        Returns (reward, done). Reward weights ported directly from JRHT
        Section III-B (see constants above).
        """
        serving = ue.serving_cell_id
        sinr_serving = sinr_list[serving]

        if action == 0:  # defer
            if sinr_serving < RLF_SINR_THRESHOLD_DB:
                return REWARD_RLF, True
            return REWARD_HEALTHY_DEFER, False

        # trigger -> action 1 = best neighbor, action 2 = 2nd-best neighbor
        neighbors = self._best_neighbors(ue, sinr_list, k=2)
        if not neighbors:
            return REWARD_HEALTHY_DEFER, False  # no viable target, treat as defer
        target_idx = neighbors[0] if action == 1 else (neighbors[1] if len(neighbors) > 1 else neighbors[0])

        history = self._serving_history.setdefault(ue.ue_id, [])
        is_ping_pong = target_idx in history[-PING_PONG_WINDOW_STEPS:]

        # execute handover
        self.cells[serving].detach(ue.ue_id)
        attached = self.cells[target_idx].attach(ue.ue_id)
        if not attached:  # target overloaded - handover fails, stay put
            self.cells[serving].attach(ue.ue_id)
            return REWARD_RLF, False

        ue.attach_to(target_idx)
        history.append(target_idx)

        if is_ping_pong:
            return REWARD_PING_PONG, False
        return REWARD_SUCCESSFUL_HO, False
"""
Main gym-style RAN environment.
Owner: Shashank
RAN simulation as a PettingZoo/Gym-style multi-agent env
Depends on: cell.py, ue.py, channel.py
Shreya-Shravani-Shloka: read output of this file
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from env.cell import Cell
from env.ue import UE
from env.channel import compute_rsrp, compute_interference

# constants
NUM_CELLS = 7
NUM_UES = 10
MAX_STEPS = 200
HO_RSRP_THRESHOLD = -90 # dBm - below this, HO should trigger
PING_PONG_WINDOW = 5 # steps - HO back within this = ping-pong
CELL_LOAD_THRESHOLD = 0.8 # 80% load = congested


def env(**kwargs):
    """Wraps RANEnv in PettingZoo recommended wrappers"""
    raw_env = RANEnv(**kwargs)
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env


def raw_env(**kwargs):
    return RANEnv(**kwargs)


class RANEnv(ParallelEnv):
    """
    Multi-agent RAN environment simulating 7-cell hexagonal layout.
    Follows PettingZoo ParallelEnv API  all agents act simultaneously.

    Agent IDs : "cell_0" ... "cell_6"
    Observation : [rsrp x num_cells, load x num_cells, ue_velocity, interference]
    Action : 0=STAY, 1=PREPARE_HO, 2=TRIGGER_HO
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ran_env_v0",
        "is_parallelizable": True,
    }

    def __init__(self, num_cells=NUM_CELLS, num_ues=NUM_UES, render_mode=None):
        super().__init__()
        self.num_cells = num_cells
        self.num_ues = num_ues
        self.render_mode = render_mode

        # PettingZoo requires agents as list of string IDs
        self.possible_agents = [f"cell_{i}" for i in range(num_cells)]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        # observation dimension per agent
        self._obs_dim = num_cells + num_cells + 1 + 1 # rsrp + load + velocity + interference

        # Cells and UEs (initialized properly in reset())
        self.cells = None
        self.ues = None
        self.step_count = 0
        self.ho_history = {}
        self.metrics = {}

    # PettingZoo required properties

    @property
    def observation_spaces(self):
        return {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    @property
    def action_spaces(self):
        return {
            agent: spaces.Discrete(3)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # core API

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.metrics = {
            "ho_triggered": 0,
            "ho_success": 0,
            "ping_pong": 0,
            "missed_ho": 0,
        }

        self.cells = [Cell(cell_id=i) for i in range(self.num_cells)]
        self.ues = [UE(ue_id=i, num_cells=self.num_cells) for i in range(self.num_ues)]
        self.ho_history = {ue.ue_id: [] for ue in self.ues}

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict):
        """
        actions: dict {"cell_0": int, "cell_1": int, ...}
        Returns: obs, rewards, terminations, truncations, infos
        """
        self.step_count += 1
        rewards = {agent: 0.0 for agent in self.agents}

        # move UEs
        for ue in self.ues:
            ue.move()

        # update channel
        for ue in self.ues:
            ue.rsrp = [compute_rsrp(ue, cell) for cell in self.cells]
            ue.interference = compute_interference(ue, self.cells)

        # execute actions + compute rewards
        for agent, action in actions.items():
            cell_id = self.agent_name_mapping[agent]
            cell = self.cells[cell_id]
            cell_ues = [ue for ue in self.ues if ue.serving_cell == cell_id]

            for ue in cell_ues:
                rewards[agent] += self._execute_action(action, ue, cell_id)

        # update cell loads
        for cell in self.cells:
            cell.update_load(self.ues)

        # termination / truncation
        truncated = self.step_count >= MAX_STEPS
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        if truncated:
            self.agents = [] # PettingZoo convention: clear agents on done

        observations = self._get_observations()
        infos = {agent: {"metrics": self.metrics} for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    # internal helpers

    def _execute_action(self, action, ue, cell_id):
        """
        Executes HO decision for one UE, returns scalar reward.
        action: 0=STAY, 1=PREPARE_HO, 2=TRIGGER_HO
        """
        reward = 0.0
        best_cell = int(np.argmax(ue.rsrp))
        current_rsrp = ue.rsrp[cell_id]

        if action == 0:  # STAY
            if current_rsrp < HO_RSRP_THRESHOLD:
                reward -= 1.0 # missed HO penalty
                self.metrics["missed_ho"] += 1

        elif action == 2:  # TRIGGER_HO
            self.metrics["ho_triggered"] += 1
            if best_cell != cell_id:
                ue.serving_cell = best_cell
                reward += 1.0 # HO success
                self.metrics["ho_success"] += 1

                # Ping-pong check
                history = self.ho_history[ue.ue_id]
                if len(history) >= 2:
                    if history[-1] == cell_id and \
                       (self.step_count - history[-2]) < PING_PONG_WINDOW:
                        reward -= 1.0 # ping-pong penalty
                        self.metrics["ping_pong"] += 1

                self.ho_history[ue.ue_id].append(self.step_count)
            else:
                reward -= 0.5 # unnecessary HO

        elif action == 1:  # PREPARE_HO
            reward += 0.1

        # load balance bonus
        if self.cells[best_cell].load < CELL_LOAD_THRESHOLD:
            reward += 0.5

        return reward

    def _get_observations(self):
        """
        Returns dict {"cell_0": np.array, ...}  one obs vector per agent.
        Each obs: [rsrp x num_cells, load x num_cells, avg_velocity, avg_interference]
        """
        obs = {}
        for agent in self.possible_agents:
            cell_id = self.agent_name_mapping[agent]
            cell_ues = [ue for ue in self.ues if ue.serving_cell == cell_id]

            if cell_ues:
                avg_rsrp = np.mean([ue.rsrp for ue in cell_ues], axis=0)
                avg_velocity = float(np.mean([ue.velocity for ue in cell_ues]))
                avg_interference = float(np.mean([ue.interference for ue in cell_ues]))
            else:
                avg_rsrp = np.zeros(self.num_cells)
                avg_velocity = 0.0
                avg_interference = 0.0

            cell_loads = np.array([c.load for c in self.cells], dtype=np.float32)

            obs[agent] = np.concatenate([
                avg_rsrp.astype(np.float32),
                cell_loads,
                [np.float32(avg_velocity)],
                [np.float32(avg_interference)]
            ])

        return obs

    def render(self):
        if self.render_mode == "human":
            print(f"\nStep: {self.step_count}")
            for ue in self.ues:
                print(f"  UE {ue.ue_id} → Cell {ue.serving_cell} | RSRP: {max(ue.rsrp):.1f} dBm")
            print(f"  Metrics: {self.metrics}")

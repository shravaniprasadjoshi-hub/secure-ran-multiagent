import numpy as np
from collections import deque
import json

class StateSharing:
    """
    Manages how agents share their local observations with each other.
    In a real distributed RAN system, each agent only sees its own
    cell/gNB — state sharing lets agents build a global picture
    without centralizing all control.
    
    Features:
    - agents broadcast their local state each step
    - agents can query neighbors' states
    - stale/missing states are handled gracefully
    - state history is maintained for replay/training
    """

    def __init__(self, n_agents, state_dim, history_len=10,
                 staleness_threshold=5):
        """
        n_agents            : total number of agents
        state_dim           : dimension of each agent's local state vector
        history_len         : how many past states to keep per agent
        staleness_threshold : steps after which a state is considered stale
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.history_len = history_len
        self.staleness_threshold = staleness_threshold

        # current shared states — what each agent has broadcast
        self.shared_states = {
            i: None for i in range(n_agents)
        }

        # tracks when each agent last updated its state
        self.last_update_step = {
            i: -1 for i in range(n_agents)
        }

        # state history per agent
        self.state_history = {
            i: deque(maxlen=history_len) for i in range(n_agents)
        }

        # neighbor map — which agents each agent can communicate with
        # default: all agents can see all others (fully connected)
        self.neighbor_map = {
            i: [j for j in range(n_agents) if j != i]
            for i in range(n_agents)
        }

        # current simulation step
        self.current_step = 0

        # communication log
        self.comm_log = []

    def set_topology(self, neighbor_map: dict):
        """
        Override default fully-connected topology.
        Use this to simulate limited communication range.
        
        neighbor_map: {agent_id: [list of neighbor agent_ids]}
        Example (ring topology):
            {0: [1, 7], 1: [0, 2], 2: [1, 3], ...}
        """
        self.neighbor_map = neighbor_map
        print(f"[StateSharing] Topology updated — "
              f"avg neighbors: "
              f"{np.mean([len(v) for v in neighbor_map.values()]):.1f}")

    def broadcast(self, agent_id: int, local_state: np.ndarray):
        """
        Agent broadcasts its current local state to the shared space.
        Call this every step for each agent.
        
        agent_id   : which agent is broadcasting
        local_state: numpy array of shape (state_dim,)
        """
        assert len(local_state) == self.state_dim, \
            f"State dim mismatch: expected {self.state_dim}, " \
            f"got {len(local_state)}"

        self.shared_states[agent_id] = local_state.copy()
        self.last_update_step[agent_id] = self.current_step
        self.state_history[agent_id].append(local_state.copy())

        self.comm_log.append({
            "step": self.current_step,
            "agent_id": agent_id,
            "event": "broadcast"
        })

    def broadcast_all(self, states: dict):
        """
        Broadcast all agents' states at once.
        states: {agent_id: np.ndarray}
        """
        for agent_id, state in states.items():
            self.broadcast(agent_id, state)

    def get_neighbor_states(self, agent_id: int,
                            exclude_stale: bool = True):
        """
        Returns states of all neighbors for a given agent.
        
        agent_id      : the agent requesting neighbor states
        exclude_stale : if True, skips neighbors that haven't
                        updated recently
        Returns:
            neighbor_states: {neighbor_id: state_array}
            stale_neighbors: list of neighbor_ids with stale states
        """
        neighbors = self.neighbor_map.get(agent_id, [])
        neighbor_states = {}
        stale_neighbors = []

        for neighbor_id in neighbors:
            state = self.shared_states[neighbor_id]
            last_update = self.last_update_step[neighbor_id]
            is_stale = (self.current_step - last_update) \
                        > self.staleness_threshold

            if state is None or (exclude_stale and is_stale):
                stale_neighbors.append(neighbor_id)
                continue

            neighbor_states[neighbor_id] = state

        return neighbor_states, stale_neighbors

    def get_global_state(self, exclude_stale: bool = True):
        """
        Aggregates all agents' states into one global state vector.
        Used by centralized critic in MAPPO (CTDE).
        
        Returns: np.ndarray of shape (n_agents * state_dim,)
        """
        global_state = []

        for agent_id in range(self.n_agents):
            state = self.shared_states[agent_id]
            last_update = self.last_update_step[agent_id]
            is_stale = (self.current_step - last_update) \
                        > self.staleness_threshold

            if state is None or (exclude_stale and is_stale):
                # use zeros for missing/stale agents
                global_state.append(np.zeros(self.state_dim))
            else:
                global_state.append(state)

        return np.concatenate(global_state)

    def get_augmented_state(self, agent_id: int):
        """
        Returns agent's own state + averaged neighbor states combined.
        This is what each agent gets as input to its actor network.
        
        Returns: np.ndarray of shape (state_dim * 2,)
                 [own_state | mean_neighbor_state]
        """
        own_state = self.shared_states[agent_id]
        if own_state is None:
            own_state = np.zeros(self.state_dim)

        neighbor_states, _ = self.get_neighbor_states(agent_id)

        if neighbor_states:
            neighbor_mean = np.mean(
                list(neighbor_states.values()), axis=0
            )
        else:
            neighbor_mean = np.zeros(self.state_dim)

        return np.concatenate([own_state, neighbor_mean])

    def get_state_history(self, agent_id: int):
        """Returns recent state history for an agent."""
        return list(self.state_history[agent_id])

    def is_stale(self, agent_id: int):
        """Checks if an agent's state is stale."""
        last = self.last_update_step[agent_id]
        return (self.current_step - last) > self.staleness_threshold

    def get_stale_agents(self):
        """Returns list of agents with stale states."""
        return [i for i in range(self.n_agents) if self.is_stale(i)]

    def step(self):
        """
        Advance the simulation step counter.
        Call this at the end of each environment step.
        """
        self.current_step += 1

    def reset(self):
        """Resets all shared states — call between episodes."""
        for i in range(self.n_agents):
            self.shared_states[i] = None
            self.last_update_step[i] = -1
            self.state_history[i].clear()
        self.current_step = 0
        self.comm_log.clear()

    def summary(self):
        """Prints current state of the sharing system."""
        print("\n=== State Sharing Summary ===")
        print(f"Step           : {self.current_step}")
        print(f"Agents online  : "
              f"{sum(1 for s in self.shared_states.values() if s is not None)}"
              f"/{self.n_agents}")
        stale = self.get_stale_agents()
        if stale:
            print(f"Stale agents   : {stale}")
        print(f"Global state dim: {self.n_agents * self.state_dim}")
        print(f"=============================\n")

    def export_history(self, filepath: str):
        """
        Saves communication log to JSON.
        Useful for debugging and paper analysis.
        """
        with open(filepath, "w") as f:
            json.dump(self.comm_log, f, indent=2)
        print(f"[StateSharing] History saved to {filepath}")
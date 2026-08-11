import numpy as np
import random

class ByzantineAgent:
    """
    Simulates a compromised or faulty agent in the RAN multi-agent system.
    Used to stress-test the consensus and anomaly detection modules.
    
    Attack types:
    - random:    agent sends completely random actions
    - opposite:  agent always does the opposite of what it should
    - replay:    agent keeps repeating an old action regardless of current state
    - gradual:   agent slowly drifts from correct behavior (harder to detect)
    """

    ATTACK_TYPES = ["random", "opposite", "replay", "gradual"]

    def __init__(self, agent_id, attack_type="random", drift_rate=0.1):
        """
        agent_id   : which agent is compromised (e.g. "agent_3")
        attack_type: one of the 4 attack types above
        drift_rate : how fast gradual attack drifts (only used if attack_type="gradual")
        """
        assert attack_type in self.ATTACK_TYPES, f"Unknown attack type: {attack_type}"
        
        self.agent_id = agent_id
        self.attack_type = attack_type
        self.drift_rate = drift_rate
        
        self._replay_action = None   # stores action for replay attack
        self._drift_step = 0         # tracks how far gradual drift has gone
        self.is_compromised = True

    def corrupt_action(self, correct_action, action_space_size):
        """
        Takes what the agent SHOULD do and returns a corrupted version.
        
        correct_action   : the legitimate action the agent would take (int)
        action_space_size: total number of possible actions
        
        Returns corrupted action (int)
        """
        if self.attack_type == "random":
            return self._random_attack(action_space_size)
        
        elif self.attack_type == "opposite":
            return self._opposite_attack(correct_action, action_space_size)
        
        elif self.attack_type == "replay":
            return self._replay_attack(correct_action, action_space_size)
        
        elif self.attack_type == "gradual":
            return self._gradual_attack(correct_action, action_space_size)

    def _random_attack(self, action_space_size):
        """Sends a completely random action."""
        return random.randint(0, action_space_size - 1)

    def _opposite_attack(self, correct_action, action_space_size):
        """Sends the opposite of the correct action."""
        return (action_space_size - 1) - correct_action

    def _replay_attack(self, correct_action, action_space_size):
        """
        First call: saves the correct action as the replay buffer.
        All future calls: keeps sending that same old action.
        """
        if self._replay_action is None:
            self._replay_action = correct_action
        return self._replay_action

    def _gradual_attack(self, correct_action, action_space_size):
        """
        Slowly drifts away from correct behavior.
        Early on looks normal, gets worse over time — hardest to detect.
        """
        self._drift_step += 1
        # probability of sending wrong action increases with each step
        corrupt_probability = min(self._drift_step * self.drift_rate, 1.0)
        
        if random.random() < corrupt_probability:
            return self._random_attack(action_space_size)
        else:
            return correct_action

    def reset(self):
        """Resets the agent state (use between episodes)."""
        self._replay_action = None
        self._drift_step = 0

    def __repr__(self):
        return f"ByzantineAgent(id={self.agent_id}, attack={self.attack_type})"


class ByzantineFaultInjector:
    """
    Manages multiple Byzantine agents in the system.
    You tell it which agents to compromise and how,
    and it handles corrupting their actions during simulation.
    """

    def __init__(self, total_agents):
        """
        total_agents: total number of agents in your system (e.g. 8)
        """
        self.total_agents = total_agents
        self.compromised_agents = {}  # agent_id -> ByzantineAgent

    def inject(self, agent_id, attack_type="random", drift_rate=0.1):
        """
        Mark an agent as compromised.
        Call this before running simulation.
        """
        self.compromised_agents[agent_id] = ByzantineAgent(
            agent_id, attack_type, drift_rate
        )
        print(f"[ByzantineInjector] Agent {agent_id} compromised — attack: {attack_type}")

    def is_compromised(self, agent_id):
        return agent_id in self.compromised_agents

    def get_action(self, agent_id, correct_action, action_space_size):
        """
        If agent is compromised, returns corrupted action.
        If agent is clean, returns correct action unchanged.
        """
        if self.is_compromised(agent_id):
            return self.compromised_agents[agent_id].corrupt_action(
                correct_action, action_space_size
            )
        return correct_action

    def reset_all(self):
        for agent in self.compromised_agents.values():
            agent.reset()

    def status(self):
        """Prints current compromise status of all agents."""
        print(f"\n=== Byzantine Fault Status ===")
        print(f"Total agents : {self.total_agents}")
        print(f"Compromised  : {list(self.compromised_agents.keys())}")
        print(f"Clean        : {[i for i in range(self.total_agents) if i not in self.compromised_agents]}")
        print(f"==============================\n")
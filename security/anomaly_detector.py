"""
security/anomaly_detector.py - statistical anomaly detection on agent behavior
Owner: Shreya
Reads from: mappo_agent.py (action history), consensus.py (consensus decisions)
Outputs to: policy_checker.py (flagged agents), trust.py (trust score updates)
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple

# constants
WINDOW_SIZE = 20 # steps to keep in rolling history per agent
ZSCORE_THRESHOLD = 2.5 # flag if action rate deviates this many std devs
MIN_HISTORY = 10 # minimum steps before detection kicks in
HO_RATE_UPPER = 0.85 # CRITICAL: agent triggering HO >85% = suspicious
HO_RATE_LOWER = 0.02 # CRITICAL: agent triggering HO <2% = suspicious (but too passive)
CONSENSUS_DEVIATION_THRESH = 0.6 # if agent disagrees with consensus >60% = flag


class AgentActionHistory:
    """
    Tracks rolling action history for one agent.
    """

    def __init__(self, agent_id: str, window: int = WINDOW_SIZE):
        self.agent_id = agent_id
        self.window = window

        # rolling buffers
        self.actions = deque(maxlen=window) # raw actions taken
        self.consensus_match = deque(maxlen=window) # 1=matched consensus, 0=deviated
        self.rewards = deque(maxlen=window) # reward received per step

    def record(self, action: int, matched_consensus: bool, reward: float):
        self.actions.append(action)
        self.consensus_match.append(int(matched_consensus))
        self.rewards.append(reward)

    @property
    def ho_trigger_rate(self) -> float:
        # fraction of steps where agent chose TRIGGER_HO (action=2)
        if not self.actions:
            return 0.0
        return sum(1 for a in self.actions if a == 2) / len(self.actions)

    @property
    def consensus_agreement_rate(self) -> float:
        if not self.consensus_match:
            return 1.0
        return sum(self.consensus_match) / len(self.consensus_match)

    @property
    def avg_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards))

    def has_enough_history(self) -> bool:
        return len(self.actions) >= MIN_HISTORY


class AnomalyDetector:
    """
    Detects Byzantine or faulty agents by monitoring:
    - abnormal HO trigger rates (too high or too low)
    - consistent deviation from consensus decisions
    - reward z-score outliers across agents

    IMPORTANT: call record_step() every env step
               call detect() every N steps or after each consensus round
    
    NOTE: 
    flagged_agents from detect() feeds into policy_checker.py
    trust_deltas from detect() feeds into trust.py
    """

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids

        # one history tracker per agent
        self.histories: Dict[str, AgentActionHistory] = {
            agent_id: AgentActionHistory(agent_id)
            for agent_id in agent_ids
        }

        # anomaly log for debugging + paper results
        self.anomaly_log: List[dict] = []

    def record_step(
        self,
        actions: Dict[str, int],
        consensus_decisions: Dict[str, int],
        rewards: Dict[str, float]
    ):
        """
        Call every env step.
        actions : {agent_id: action taken by agent}
        consensus_decisions: {agent_id: action decided by consensus module}
        rewards : {agent_id: reward received}
        consensus_decisions comes from consensus.py output
        """
        for agent_id in self.agent_ids:
            action = actions.get(agent_id, 0)
            consensus = consensus_decisions.get(agent_id, 0)
            reward = rewards.get(agent_id, 0.0)
            matched = (action == consensus)
            self.histories[agent_id].record(action, matched, reward)

    def detect(self, step: int) -> Tuple[List[str], Dict[str, float]]:
        """
        Runs anomaly detection across all agents.
        Returns:
          flagged_agents : list of agent_ids flagged as anomalous
          trust_deltas   : {agent_id: delta to apply to trust score}

        IMPORTANT: only runs if agents have MIN_HISTORY steps recorded
        consume flagged_agents in policy_checker.py
        consume trust_deltas in trust.py -> update_trust_score()
        """
        flagged_agents = []
        trust_deltas = {agent_id: 0.0 for agent_id in self.agent_ids}

        # collect reward stats across agents for z-score
        avg_rewards = {
            agent_id: self.histories[agent_id].avg_reward
            for agent_id in self.agent_ids
            if self.histories[agent_id].has_enough_history()
        }

        if len(avg_rewards) < 2:
            # - not enough data yet
            return flagged_agents, trust_deltas

        reward_values = np.array(list(avg_rewards.values()))
        reward_mean = reward_values.mean()
        reward_std = reward_values.std() + 1e-8

        for agent_id in self.agent_ids:
            history = self.histories[agent_id]

            if not history.has_enough_history():
                continue

            reasons = []

            # Check 1: HO trigger rate bounds
            ho_rate = history.ho_trigger_rate
            if ho_rate > HO_RATE_UPPER:
                reasons.append(f"ho_rate_high:{ho_rate:.2f}")
                trust_deltas[agent_id] -= 0.2

            elif ho_rate < HO_RATE_LOWER:
                reasons.append(f"ho_rate_low:{ho_rate:.2f}")
                trust_deltas[agent_id] -= 0.1

            # Check 2: consensus deviation
            agreement = history.consensus_agreement_rate
            if agreement < (1 - CONSENSUS_DEVIATION_THRESH):
                reasons.append(f"consensus_deviation:{agreement:.2f}")
                trust_deltas[agent_id] -= 0.3

            # Check 3: reward z-score outlier
            # CRITICAL: low reward alone doesnt mean byzantine
            # only flag if ALSO deviating on checks 1 or 2
            if agent_id in avg_rewards:
                z = (avg_rewards[agent_id] - reward_mean) / reward_std
                if z < -ZSCORE_THRESHOLD and len(reasons) > 0:
                    reasons.append(f"reward_outlier:z={z:.2f}")
                    trust_deltas[agent_id] -= 0.2

            # flag agent if any reason found 
            if reasons:
                flagged_agents.append(agent_id)
                self.anomaly_log.append({
                    "step": step,
                    "agent": agent_id,
                    "reasons": reasons,
                    "ho_rate": history.ho_trigger_rate,
                    "consensus_agreement": history.consensus_agreement_rate,
                    "avg_reward": history.avg_reward,
                })

            else:
                # well-behaved agent: small positive trust delta
                trust_deltas[agent_id] += 0.05

        # clip all trust deltas
        for agent_id in trust_deltas:
            trust_deltas[agent_id] = float(np.clip(trust_deltas[agent_id], -0.5, 0.1))

        return flagged_agents, trust_deltas

    def get_agent_summary(self, agent_id: str) -> dict:
        """Quick summary for logging / paper results"""
        h = self.histories[agent_id]
        return {
            "agent_id": agent_id,
            "ho_trigger_rate": h.ho_trigger_rate,
            "consensus_agreement": h.consensus_agreement_rate,
            "avg_reward": h.avg_reward,
            "history_length": len(h.actions),
        }

    def get_full_log(self) -> List[dict]:
        # returns full anomaly event log for results/plots
        return self.anomaly_log
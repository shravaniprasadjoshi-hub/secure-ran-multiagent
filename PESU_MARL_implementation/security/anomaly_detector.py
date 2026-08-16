import numpy as np
from collections import deque

class AnomalyDetector:
    """
    Detects Byzantine/faulty agents by watching their actions over time.
    Flags agents whose behavior deviates significantly from the group.
    
    Detection methods:
    - statistical : flags agents whose actions are statistical outliers
    - voting      : flags agents who consistently disagree with the majority
    - drift       : flags agents whose behavior changes suspiciously over time
    """

    def __init__(self, n_agents, window_size=20, threshold=2.0):
        """
        n_agents    : total number of agents in the system
        window_size : how many recent actions to look at (sliding window)
        threshold   : how many std deviations before flagging as anomaly
        """
        self.n_agents = n_agents
        self.window_size = window_size
        self.threshold = threshold

        # stores recent actions for each agent
        self.action_history = {
            i: deque(maxlen=window_size) for i in range(n_agents)
        }
        
        # suspicion score per agent — goes up when flagged, down when clean
        self.suspicion_scores = {i: 0.0 for i in range(n_agents)}
        
        # full log of all detections
        self.detection_log = []

    def record_actions(self, actions: dict):
        """
        Call this every step with each agent's action.
        actions: dict like {0: 2, 1: 1, 2: 2, 3: 4, ...}
        """
        for agent_id, action in actions.items():
            self.action_history[agent_id].append(action)

    def detect_statistical_outliers(self):
        """
        Compares each agent's average action to the group average.
        Agents far from the group mean get flagged.
        Returns list of flagged agent ids.
        """
        flagged = []
        
        # need enough history first
        if not all(len(h) >= 5 for h in self.action_history.values()):
            return flagged

        # compute mean action per agent
        agent_means = {
            i: np.mean(list(self.action_history[i]))
            for i in range(self.n_agents)
        }

        all_means = list(agent_means.values())
        group_mean = np.mean(all_means)
        group_std = np.std(all_means)

        if group_std == 0:
            return flagged

        for agent_id, mean in agent_means.items():
            z_score = abs(mean - group_mean) / group_std
            if z_score > self.threshold:
                flagged.append(agent_id)
                self.suspicion_scores[agent_id] += 1.0
                self.detection_log.append({
                    "method": "statistical",
                    "agent_id": agent_id,
                    "z_score": round(z_score, 3)
                })

        return flagged

    def detect_voting_outliers(self, actions: dict):
        """
        Checks who disagrees with the majority vote most often.
        If an agent keeps voting differently from everyone else, suspicious.
        actions: current step's actions dict
        Returns list of flagged agent ids.

        # FIX (by me shreyashree, this is a post-integration-test): 
        # suspicion_scores only ever grew before this - an honest agent that legitimately disagreed
        # with the group ~6 times (0.5 * 6 = 3.0) got flagged EVERY remaining step of the episode, forever, with no way to recover.
        # In our env agents legitimately disagree often (each observes a different UE's signal), so this was flagging ~45% of clean
        # agents. Added symmetric decay on agreement: an agent alternating disagree/agree roughly 50/50 now nets to ~0 and stays unflagged;
        # only agents disagreeing MORE than they agree accumulate toward the 3.0 threshold. See training/train_secure.py docstring for
        # the false_positive_rate numbers that motivated this.
        """
        flagged = []
        
        if not actions:
            return flagged

        # find majority action
        action_values = list(actions.values())
        majority_action = max(set(action_values), key=action_values.count)

        for agent_id, action in actions.items():
            if action != majority_action:
                self.suspicion_scores[agent_id] += 0.5
                
                # only flag if suspicion has built up enough
                if self.suspicion_scores[agent_id] >= 3.0:
                    flagged.append(agent_id)
                    self.detection_log.append({
                        "method": "voting",
                        "agent_id": agent_id,
                        "their_action": action,
                        "majority_action": majority_action
                    })
            else:
                # decay - agreeing with the group lets suspicion recover,
                # symmetric with the 0.5 increment above
                self.suspicion_scores[agent_id] = max(0.0, self.suspicion_scores[agent_id] - 0.5)

        return flagged

    def detect_drift(self):
        """
        Checks if any agent's behavior has changed significantly
        between the first half and second half of their history window.
        Catches gradual Byzantine attacks.
        Returns list of flagged agent ids.
        """
        flagged = []

        for agent_id in range(self.n_agents):
            history = list(self.action_history[agent_id])
            
            # need enough data
            if len(history) < self.window_size:
                continue

            mid = len(history) // 2
            first_half = np.mean(history[:mid])
            second_half = np.mean(history[mid:])

            drift = abs(second_half - first_half)

            if drift > self.threshold:
                flagged.append(agent_id)
                self.suspicion_scores[agent_id] += 1.5
                self.detection_log.append({
                    "method": "drift",
                    "agent_id": agent_id,
                    "drift_magnitude": round(drift, 3)
                })

        return flagged

    def run_all_detectors(self, current_actions: dict):
        """
        Runs all three detection methods at once.
        Returns combined list of flagged agent ids (no duplicates).
        """
        self.record_actions(current_actions)

        flagged = set()
        flagged.update(self.detect_statistical_outliers())
        flagged.update(self.detect_voting_outliers(current_actions))
        flagged.update(self.detect_drift())

        if flagged:
            print(f"[AnomalyDetector] Flagged agents: {flagged}")

        return list(flagged)

    def get_suspicion_scores(self):
        """Returns current suspicion score for all agents."""
        return self.suspicion_scores

    def reset(self):
        """Resets all history and scores — use between episodes."""
        for i in range(self.n_agents):
            self.action_history[i].clear()
            self.suspicion_scores[i] = 0.0
        self.detection_log.clear()

    def summary(self):
        """Prints a summary of suspicion scores."""
        print("\n=== Anomaly Detector Summary ===")
        for agent_id, score in self.suspicion_scores.items():
            status = "⚠ SUSPICIOUS" if score >= 3.0 else "✓ clean"
            print(f"  Agent {agent_id}: score={score:.1f}  {status}")
        print(f"================================\n")
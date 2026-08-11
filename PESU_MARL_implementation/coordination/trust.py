import numpy as np
from collections import deque

class TrustManager:
    """
    Tracks and updates trust scores for each agent over time.
    Trust scores influence how much weight each agent gets in consensus.
    
    Trust goes UP when agent:
    - agrees with consensus
    - passes policy checks
    - behaves consistently over time
    
    Trust goes DOWN when agent:
    - is flagged by anomaly detector
    - violates policy checks
    - disagrees with consensus repeatedly
    
    Trust score range: 0.0 (completely untrusted) to 1.0 (fully trusted)
    """

    def __init__(self, n_agents, initial_trust=1.0, 
                 decay_rate=0.05, recovery_rate=0.02,
                 min_trust=0.1, max_trust=1.0):
        """
        n_agents      : total number of agents
        initial_trust : starting trust score for all agents (default full trust)
        decay_rate    : how much trust drops when agent misbehaves
        recovery_rate : how much trust recovers per good step
        min_trust     : floor — even bad agents keep a tiny score
        max_trust     : ceiling — perfect trust
        """
        self.n_agents = n_agents
        self.decay_rate = decay_rate
        self.recovery_rate = recovery_rate
        self.min_trust = min_trust
        self.max_trust = max_trust

        # initialize all agents with full trust
        self.trust_scores = {
            i: initial_trust for i in range(n_agents)
        }

        # history of trust scores over time (for plotting)
        self.trust_history = {
            i: deque(maxlen=100) for i in range(n_agents)
        }
        for i in range(n_agents):
            self.trust_history[i].append(initial_trust)

        # track consecutive good/bad steps per agent
        self.good_streak = {i: 0 for i in range(n_agents)}
        self.bad_streak = {i: 0 for i in range(n_agents)}

        # full event log
        self.event_log = []

    def update_on_consensus(self, actions: dict, final_action: int):
        """
        After consensus is reached, reward agents that agreed
        and penalize agents that disagreed.
        
        actions      : {agent_id: action} what each agent voted
        final_action : the consensus decision
        """
        for agent_id, action in actions.items():
            if action == final_action:
                # agreed with consensus — small trust boost
                self._increase_trust(agent_id, self.recovery_rate,
                                    reason="agreed_with_consensus")
                self.good_streak[agent_id] += 1
                self.bad_streak[agent_id] = 0
            else:
                # disagreed — small trust penalty
                self._decrease_trust(agent_id, self.decay_rate * 0.5,
                                    reason="disagreed_with_consensus")
                self.bad_streak[agent_id] += 1
                self.good_streak[agent_id] = 0

    def update_on_anomaly(self, flagged_agents: list, 
                          clean_agents: list = None):
        """
        Penalizes flagged agents, rewards clean ones.
        Called after AnomalyDetector runs.
        
        flagged_agents: agent ids flagged as suspicious
        clean_agents  : agent ids confirmed clean (optional)
        """
        for agent_id in flagged_agents:
            self._decrease_trust(agent_id, self.decay_rate * 2,
                                reason="flagged_by_anomaly_detector")
            self.bad_streak[agent_id] += 1

        if clean_agents:
            for agent_id in clean_agents:
                if agent_id not in flagged_agents:
                    self._increase_trust(agent_id, self.recovery_rate * 0.5,
                                        reason="confirmed_clean")

    def update_on_policy(self, policy_results: dict):
        """
        Updates trust based on policy checker results.
        
        policy_results: {agent_id: {"passed": bool, "violations": list}}
                        (output of PolicyChecker.validate_all())
        """
        for agent_id, result in policy_results.items():
            if result["passed"]:
                self._increase_trust(agent_id, self.recovery_rate,
                                    reason="passed_policy_check")
            else:
                # penalize more for each violation
                penalty = self.decay_rate * len(result["violations"])
                self._decrease_trust(agent_id, penalty,
                                    reason=f"policy_violations: "
                                           f"{result['violations']}")

    def apply_streak_bonus(self):
        """
        Agents on a long good streak get a bonus trust boost.
        Agents on a long bad streak get extra penalty.
        Call this every N steps (e.g. every 10 steps).
        """
        for agent_id in range(self.n_agents):
            if self.good_streak[agent_id] >= 10:
                self._increase_trust(agent_id, self.recovery_rate * 2,
                                    reason="good_streak_bonus")
            elif self.bad_streak[agent_id] >= 5:
                self._decrease_trust(agent_id, self.decay_rate,
                                    reason="bad_streak_penalty")

    def get_trust_scores(self):
        """Returns current trust scores for all agents."""
        return dict(self.trust_scores)

    def get_trust_weights(self):
        """
        Returns normalized trust scores for use in weighted voting.
        Scores sum to n_agents so average weight stays 1.0.
        """
        scores = self.trust_scores
        total = sum(scores.values())
        if total == 0:
            return {i: 1.0 for i in range(self.n_agents)}
        
        # normalize so weights sum to n_agents
        return {
            agent_id: (score / total) * self.n_agents
            for agent_id, score in scores.items()
        }

    def is_trusted(self, agent_id, threshold=0.3):
        """
        Quick check — is this agent trusted enough to participate?
        Agents below threshold are effectively quarantined.
        """
        return self.trust_scores[agent_id] >= threshold

    def get_quarantined_agents(self, threshold=0.3):
        """
        Returns list of agents whose trust is too low to participate.
        These should be excluded from consensus entirely.
        """
        return [
            agent_id for agent_id, score in self.trust_scores.items()
            if score < threshold
        ]

    def _increase_trust(self, agent_id, amount, reason=""):
        old = self.trust_scores[agent_id]
        new = min(old + amount, self.max_trust)
        self.trust_scores[agent_id] = new
        self.trust_history[agent_id].append(new)
        self.event_log.append({
            "agent_id": agent_id,
            "event": "increase",
            "old": round(old, 3),
            "new": round(new, 3),
            "reason": reason
        })

    def _decrease_trust(self, agent_id, amount, reason=""):
        old = self.trust_scores[agent_id]
        new = max(old - amount, self.min_trust)
        self.trust_scores[agent_id] = new
        self.trust_history[agent_id].append(new)
        self.event_log.append({
            "agent_id": agent_id,
            "event": "decrease",
            "old": round(old, 3),
            "new": round(new, 3),
            "reason": reason
        })

    def reset(self):
        """Resets all trust scores to initial state."""
        for i in range(self.n_agents):
            self.trust_scores[i] = self.max_trust
            self.trust_history[i].clear()
            self.trust_history[i].append(self.max_trust)
            self.good_streak[i] = 0
            self.bad_streak[i] = 0
        self.event_log.clear()

    def summary(self):
        """Prints current trust scores for all agents."""
        print("\n=== Trust Manager Summary ===")
        for agent_id, score in self.trust_scores.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            status = "🔴 LOW" if score < 0.3 else \
                     "🟡 MED" if score < 0.7 else "🟢 HIGH"
            print(f"  Agent {agent_id}: [{bar}] {score:.2f}  {status}")
        quarantined = self.get_quarantined_agents()
        if quarantined:
            print(f"\n  ⚠ Quarantined agents: {quarantined}")
        print(f"=============================\n")
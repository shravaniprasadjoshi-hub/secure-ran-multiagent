import numpy as np
from collections import Counter

class ConsensusEngine:
    """
    Manages collective decision-making across all RAN agents.
    Instead of one agent making the final call, all agents vote
    and the consensus engine produces a final agreed action.
    
    Methods:
    - majority_vote    : simple majority wins
    - weighted_vote    : agents with higher trust get more say
    - byzantine_robust : excludes flagged agents before voting
    """

    def __init__(self, n_agents, action_space_size, min_agreement=0.6):
        """
        n_agents         : total number of agents
        action_space_size: number of possible actions
        min_agreement    : minimum fraction of agents that must agree
                           for consensus to be reached (default 60%)
        """
        self.n_agents = n_agents
        self.action_space_size = action_space_size
        self.min_agreement = min_agreement

        # tracks how many times consensus was reached vs failed
        self.consensus_reached = 0
        self.consensus_failed = 0

        # full history of decisions
        self.decision_log = []

    def majority_vote(self, actions: dict):
        """
        Simplest method — whichever action most agents pick wins.
        
        actions: {agent_id: action}
        Returns: (final_action, agreement_fraction)
        """
        if not actions:
            return None, 0.0

        action_counts = Counter(actions.values())
        final_action = action_counts.most_common(1)[0][0]
        agreement = action_counts[final_action] / len(actions)

        self._log_decision("majority_vote", actions, final_action, agreement)
        self._update_counts(agreement)

        return final_action, agreement

    def weighted_vote(self, actions: dict, trust_scores: dict):
        """
        Agents with higher trust scores have more influence on the decision.
        
        actions      : {agent_id: action}
        trust_scores : {agent_id: float} — higher = more trusted
        Returns: (final_action, agreement_fraction)
        """
        if not actions:
            return None, 0.0

        # accumulate weighted votes per action
        weighted_counts = {}
        total_weight = 0.0

        for agent_id, action in actions.items():
            weight = trust_scores.get(agent_id, 1.0)  # default weight = 1
            weighted_counts[action] = weighted_counts.get(action, 0) + weight
            total_weight += weight

        if total_weight == 0:
            return None, 0.0

        final_action = max(weighted_counts, key=weighted_counts.get)
        agreement = weighted_counts[final_action] / total_weight

        self._log_decision("weighted_vote", actions, final_action, agreement, 
                          trust_scores=trust_scores)
        self._update_counts(agreement)

        return final_action, agreement

    def byzantine_robust_vote(self, actions: dict, flagged_agents: list,
                               trust_scores: dict = None):
        """
        Excludes flagged (suspicious/compromised) agents before voting.
        This is the most secure method — combines anomaly detection
        with consensus.
        
        actions        : {agent_id: action}
        flagged_agents : list of agent_ids to exclude (from AnomalyDetector)
        trust_scores   : optional, if provided uses weighted vote on clean agents
        Returns: (final_action, agreement_fraction, clean_agents_used)
        """
        # filter out flagged agents
        clean_actions = {
            agent_id: action 
            for agent_id, action in actions.items() 
            if agent_id not in flagged_agents
        }

        if not clean_actions:
            print("[ConsensusEngine] ⚠ No clean agents left — cannot reach consensus")
            self.consensus_failed += 1
            return None, 0.0, []

        print(f"[ConsensusEngine] Using {len(clean_actions)}/{len(actions)} "
              f"agents (excluded: {flagged_agents})")

        # use weighted if trust scores provided, else majority
        if trust_scores:
            clean_trust = {
                k: v for k, v in trust_scores.items() 
                if k in clean_actions
            }
            final_action, agreement = self.weighted_vote(
                clean_actions, clean_trust
            )
        else:
            final_action, agreement = self.majority_vote(clean_actions)

        return final_action, agreement, list(clean_actions.keys())

    def reach_consensus(self, actions: dict, flagged_agents: list = None,
                        trust_scores: dict = None):
        """
        Main method to call each step.
        Automatically picks the best voting method based on 
        what information is available.
        
        actions        : {agent_id: action}
        flagged_agents : from AnomalyDetector (optional)
        trust_scores   : from TrustManager (optional)
        
        Returns:
            final_action (int)  : the agreed upon action
            agreement (float)   : fraction of agents that agreed
            consensus_ok (bool) : True if agreement >= min_agreement
        """
        # pick method based on available info
        if flagged_agents:
            final_action, agreement, _ = self.byzantine_robust_vote(
                actions, flagged_agents, trust_scores
            )
        elif trust_scores:
            final_action, agreement = self.weighted_vote(
                actions, trust_scores
            )
        else:
            final_action, agreement = self.majority_vote(actions)

        consensus_ok = (agreement >= self.min_agreement 
                       if agreement else False)

        if consensus_ok:
            print(f"[ConsensusEngine] ✓ Consensus reached — "
                  f"action={final_action}, agreement={agreement:.1%}")
        else:
            print(f"[ConsensusEngine] ✗ Consensus failed — "
                  f"agreement={agreement:.1%} < {self.min_agreement:.1%}")

        return final_action, agreement, consensus_ok

    def _log_decision(self, method, actions, final_action, 
                      agreement, trust_scores=None):
        self.decision_log.append({
            "method": method,
            "actions": actions,
            "final_action": final_action,
            "agreement": round(agreement, 3),
            "trust_scores": trust_scores
        })

    def _update_counts(self, agreement):
        if agreement >= self.min_agreement:
            self.consensus_reached += 1
        else:
            self.consensus_failed += 1

    def reset(self):
        self.consensus_reached = 0
        self.consensus_failed = 0
        self.decision_log.clear()

    def summary(self):
        total = self.consensus_reached + self.consensus_failed
        print("\n=== Consensus Engine Summary ===")
        print(f"Total decisions : {total}")
        print(f"Consensus reached: {self.consensus_reached} "
              f"({self.consensus_reached/total:.1%})" if total > 0 else "")
        print(f"Consensus failed : {self.consensus_failed}")
        print(f"================================\n")
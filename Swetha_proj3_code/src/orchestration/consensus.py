from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentProposal:
    agent_id: str
    agent_type: str
    action: str
    confidence: float
    trust_score: float
    metadata: dict


@dataclass
class ConsensusResult:
    accepted: bool
    final_action: str
    vote_pct: float
    avg_trust: float
    avg_confidence: float
    participating_agents: int
    rejected_reason: str = ""


class ConsensusEngine:
    def __init__(self, majority_threshold: float = 0.70, trust_threshold: float = 0.80, confidence_threshold: float = 0.85):
        self.majority_threshold = majority_threshold
        self.trust_threshold = trust_threshold
        self.confidence_threshold = confidence_threshold

    def validate(self, proposals: list[AgentProposal]) -> ConsensusResult:
        if not proposals:
            return ConsensusResult(False, "none", 0, 0, 0, 0, "No proposals")

        trusted = [p for p in proposals if p.trust_score >= self.trust_threshold]
        if not trusted:
            return ConsensusResult(False, "none", 0, 0, 0, len(proposals), "Trust threshold not met")

        votes: dict[str, float] = {}
        for p in trusted:
            weight = p.trust_score * p.confidence
            votes[p.action] = votes.get(p.action, 0) + weight

        total = sum(votes.values())
        winner = max(votes, key=votes.get)
        vote_pct = votes[winner] / total if total else 0
        avg_trust = sum(p.trust_score for p in trusted) / len(trusted)
        avg_conf = sum(p.confidence for p in trusted) / len(trusted)

        accepted = (
            vote_pct >= self.majority_threshold
            and avg_trust >= self.trust_threshold
            and avg_conf >= self.confidence_threshold
        )
        reason = "" if accepted else "Consensus thresholds not satisfied"
        return ConsensusResult(accepted, winner, vote_pct, avg_trust, avg_conf, len(trusted), reason)

"""
agents/actor.py: Per-agent policy network
Owner: Shreyashree
Depends on: none
Used by: mappo_agent.py
"""

import torch.nn as nn

HIDDEN_DIM = 64


class Actor(nn.Module):
    """Per-agent policy net -> discrete action logits. Decentralized execution."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)  # raw logits
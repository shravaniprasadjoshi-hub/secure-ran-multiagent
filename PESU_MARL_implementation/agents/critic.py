"""
agents/critic.py: Shared centralized value network (CTDE)
Owner: Shreyashree
Depends on: none
Used by: mappo_agent.py (re-exported from there for agent_manager.py's import)
"""

import torch.nn as nn

HIDDEN_DIM = 64


class Critic(nn.Module):
    """Centralized value net -> shared across all agents (CTDE)."""

    def __init__(self, global_obs_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs):
        return self.net(global_obs).squeeze(-1)
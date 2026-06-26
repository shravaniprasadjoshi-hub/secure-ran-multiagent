"""
agents/mappo_agent.py: MAPPO agent implementation
Owners: Shreya
Depends on: actor.py, critic.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# constants
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99              # discount factor
GAE_LAMBDA = 0.95         # for advantage estimation
CLIP_EPS = 0.2            # PPO clip epsilon
ENTROPY_COEF = 0.01       # encourages exploration
VALUE_COEF = 0.5          # critic loss weight
MAX_GRAD_NORM = 0.5       # gradient clipping
UPDATE_EPOCHS = 4         # PPO update epochs per rollout
MINIBATCH_SIZE = 64


# actor network
class Actor(nn.Module):
    """
    Policy network: obs → action probabilities
    Input:  observation vector (see ran_env.py obs_dim)
    Output: softmax over [STAY, PREPARE_HO, TRIGGER_HO]
    """
    def __init__(self, obs_dim, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs):
        """Sample action + return log prob for PPO update"""
        probs = self.forward(obs)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy


# critic network
class Critic(nn.Module):
    """
    Value network: global state → state value estimate
    Input:  concatenated observations from all agents (centralized critic)
    Output: scalar value V(s)

    Note: Centralized training, decentralized execution (CTDE) - standard MAPPO
    """
    def __init__(self, global_obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)


# rollout buffer
class RolloutBuffer:
    """Stores experience tuples for one PPO update cycle"""

    def __init__(self):
        self.obs = []
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, obs, global_obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """GAE-Lambda advantage estimation"""
        advantages = []
        gae = 0
        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32)
        )


# MAPPO Agent
class MAPPOAgent:
    """
    One agent per cell. Each has its own actor.
    Critic is shared (centralized) - managed by AgentManager.

    Usage:
        agent = MAPPOAgent(agent_id=0, obs_dim=16)
        action, log_prob, entropy = agent.select_action(obs_tensor)
        agent.store_transition(...)
        agent.update(critic, global_obs)
    """

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int = 3):
        self.agent_id = agent_id
        self.obs_dim = obs_dim

        self.actor = Actor(obs_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.buffer = RolloutBuffer()
        self.trust_score = 1.0          # used by consensus module

    def select_action(self, obs: torch.Tensor):
        """
        obs: torch.Tensor of shape (obs_dim,)
        Returns: action (int), log_prob (tensor), entropy (tensor)
        """
        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(obs)
        return action, log_prob, entropy

    def store_transition(self, obs, global_obs, action, log_prob, reward, done, value):
        self.buffer.store(obs, global_obs, action, log_prob, reward, done, value)

    def update(self, critic: Critic, last_global_obs: torch.Tensor):
        """
        PPO update step for this agent's actor.
        Critic is shared - passed in from AgentManager.
        """
        last_value = critic(last_global_obs).detach()
        advantages, returns = self.buffer.compute_returns_and_advantages(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        actions_batch = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)

        actor_losses = []

        for _ in range(UPDATE_EPOCHS):
            # recompute log probs under current policy
            probs = self.actor(obs_batch)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

        self.buffer.clear()
        return np.mean(actor_losses)

    def update_trust_score(self, consensus_agreement: float):
        """
        Called by consensus module after each decision round.
        consensus_agreement: 0.0 to 1.0 - how much this agent agreed with consensus
        Trust decays if agent consistently disagrees with majority.
        """
        alpha = 0.1                            # update rate
        self.trust_score = (1 - alpha) * self.trust_score + alpha * consensus_agreement
        self.trust_score = np.clip(self.trust_score, 0.1, 1.0)

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(path))
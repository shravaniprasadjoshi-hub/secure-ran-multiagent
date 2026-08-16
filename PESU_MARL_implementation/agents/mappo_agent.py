"""
agents/mappo_agent.py: MAPPOAgent (rollout + PPO update) - wires up Actor/Critic
Owner: Shreyashree
Depends on: actor.py, critic.py
Used by: agent_manager.py

# IMPORTANT: agent_manager.py does `from agents.mappo_agent import MAPPOAgent, Critic, LR_CRITIC, MAX_GRAD_NORM`
# Critic is imported here from critic.py and re-exported at module level - dont remove the import even though it looks unused by this file directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.actor import Actor
from agents.critic import Critic  # noqa: F401 

# hyperparams
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
MAX_GRAD_NORM = 0.5
TRUST_EMA_ALPHA = 0.1
HIDDEN_DIM = 64


def safe_categorical(logits):
    """
    Clamps and sanitizes logits before creating Categorical distribution.
    Prevents NaN/inf crashes during long training runs.
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    logits = torch.clamp(logits, min=-10, max=10)
    return Categorical(logits=logits)


class RolloutBuffer:
    """
    Holds one rollout's worth of transitions for a single agent.
    # CRITICAL: cleared at the end of MAPPOAgent.update() - don't reuse across updates
    """

    def __init__(self):
        self.obs = []
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, global_obs, action, log_prob, reward, done, value):
        self.obs.append(torch.as_tensor(obs, dtype=torch.float32))
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value):
        """
        GAE-Lambda. last_value: scalar float/tensor bootstrap for the state
        after the final stored step.
        Returns: (advantages, returns) both as 1D tensors, len == len(self.rewards)
        """
        last_value = float(last_value)
        values = self.values + [last_value]
        advantages = [0.0] * len(self.rewards)
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + GAMMA * values[t + 1] * not_done - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * not_done * gae
            advantages[t] = gae

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        return advantages, returns

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class MAPPOAgent:
    """
    One per RAN cell. Decentralized actor, trained against the shared
    centralized critic owned by AgentManager.
    """

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int = 3,
                 hidden_dim: int = HIDDEN_DIM):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.buffer = RolloutBuffer()

        self.trust_score = 1.0

    def select_action(self, obs_t: torch.Tensor, deterministic: bool = False):
        """
        obs_t: tensor (obs_dim,)
        deterministic: if True, take argmax action (eval mode) instead of
                        sampling (training mode, default - needed for PPO's
                        stochastic exploration).
        Returns: action (int), log_prob (tensor, detached), entropy (tensor, detached)
        """
        with torch.no_grad():
            logits = self.actor(obs_t)
            dist = safe_categorical(logits)  # fixed: was Categorical(logits=logits)
            action = torch.argmax(logits) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob.detach(), entropy.detach()

    def store_transition(self, obs, global_obs, action, log_prob, reward, done, value):
        self.buffer.add(obs, global_obs, action, log_prob, reward, done, value)

    def update(self, critic: Critic, last_global_obs: torch.Tensor):
        """
        PPO-clip update for this agent's actor only.
        critic: shared centralized critic (already updated by AgentManager
                BEFORE this call - see agent_manager.update())
        last_global_obs: bootstrap state for GAE
        Returns: float, mean actor loss across PPO epochs (for logging)
        """
        with torch.no_grad():
            last_value = critic(last_global_obs)

        advantages, returns = self.buffer.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.stack(self.buffer.obs)
        actions_batch = torch.as_tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()

        epoch_losses = []
        for _ in range(PPO_EPOCHS):
            logits = self.actor(obs_batch)
            dist = safe_categorical(logits)  # fixed: was Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()

            epoch_losses.append(actor_loss.item())

        self.buffer.clear()
        return sum(epoch_losses) / len(epoch_losses)

    def update_trust_score(self, agreement: float):
        """EMA of consensus agreement in [0,1]. Called by AgentManager.update_trust_scores()."""
        agreement = max(0.0, min(1.0, float(agreement)))
        self.trust_score = (1 - TRUST_EMA_ALPHA) * self.trust_score + TRUST_EMA_ALPHA * agreement
        self.trust_score = max(0.0, min(1.0, self.trust_score))

    def save(self, path: str):
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(path))
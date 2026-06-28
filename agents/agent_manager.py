"""
agents/agent_manager.py: Manages all N agents
Owners: Shreya, Shravani
Depends on: mappo_agent.py (Actor, Critic, MAPPOAgent)
TODO: Coordinate agent lifecycle, observations, actions.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.mappo_agent import MAPPOAgent, Critic, LR_CRITIC, MAX_GRAD_NORM


class AgentManager:
    """
    Owns and coordinates all per-cell MAPPO agents plus the single
    shared (centralized) critic used for CTDE training.

    Usage:
        manager = AgentManager(num_agents=4, obs_dim=16)
        actions, log_probs, _ = manager.select_actions(obs_list)
        value = manager.get_value(obs_list)
        manager.store_transitions(obs_list, actions, log_probs, rewards, dones, value)
        ... repeat for full rollout ...
        losses = manager.update(last_obs_list)
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int = 3):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # one actor per agent -> decentralized execution
        self.agents = [
            MAPPOAgent(agent_id=i, obs_dim=obs_dim, action_dim=action_dim)
            for i in range(num_agents)
        ]

        # single shared critic -> centralized training (CTDE)
        self.global_obs_dim = obs_dim * num_agents
        self.critic = Critic(self.global_obs_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    # ---------- observation helpers ----------

    def build_global_obs(self, obs_list):
        """
        obs_list: list of N tensors/arrays, each (obs_dim,)
        Returns: tensor (global_obs_dim,) - concatenation across agents.
        This is the centralized critic's input.
        """
        return torch.cat(
            [torch.as_tensor(o, dtype=torch.float32) for o in obs_list], dim=-1
        )

    # ---------- rollout ----------

    def select_actions(self, obs_list):
        """
        obs_list: list of N per-agent observations
        Returns: actions (list[int]), log_probs (list[tensor]), entropies (list[tensor])
        """
        actions, log_probs, entropies = [], [], []
        for agent, obs in zip(self.agents, obs_list):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            action, log_prob, entropy = agent.select_action(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        return actions, log_probs, entropies

    def get_value(self, obs_list):
        """Centralized critic's value estimate V(s) for the current global state."""
        global_obs = self.build_global_obs(obs_list)
        with torch.no_grad():
            value = self.critic(global_obs)
        return value.item()

    def store_transitions(self, obs_list, actions, log_probs, rewards, dones, value):
        """
        Stores one timestep for every agent.
        obs_list, actions, log_probs, rewards, dones: lists, length num_agents
        value: scalar - shared centralized value estimate for this timestep
        """
        global_obs = self.build_global_obs(obs_list)
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                obs=obs_list[i],
                global_obs=global_obs,
                action=actions[i],
                log_prob=log_probs[i],
                reward=rewards[i],
                done=dones[i],
                value=value,
            )

    # ---------- training ----------

    def update(self, last_obs_list):
        """
        Runs one PPO update cycle: shared critic first, then each agent's actor.
        last_obs_list: observations from the final rollout step, used to
                        bootstrap the value of the next state.
        Returns: dict with mean actor loss and critic loss, for logging.
        """
        last_global_obs = self.build_global_obs(last_obs_list)
        last_value = self.critic(last_global_obs).detach()

        # NOTE: critic update must run BEFORE agent.update(), since each
        # agent's buffer gets cleared at the end of its own update() call.
        critic_loss = self._update_critic(last_value)
        actor_losses = [agent.update(self.critic, last_global_obs) for agent in self.agents]

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": critic_loss,
        }

    def _update_critic(self, last_value):
        """
        Centralized critic, trained on returns pooled from every agent's
        buffer (they all share one value function under CTDE).
        """
        all_global_obs, all_returns = [], []
        for agent in self.agents:
            _, returns = agent.buffer.compute_returns_and_advantages(last_value)
            all_global_obs.append(torch.stack(agent.buffer.global_obs))
            all_returns.append(returns)

        global_obs_batch = torch.cat(all_global_obs, dim=0)
        returns_batch = torch.cat(all_returns, dim=0)

        values_pred = self.critic(global_obs_batch).squeeze(-1)
        critic_loss = F.mse_loss(values_pred, returns_batch)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_optimizer.step()

        return critic_loss.item()

    # ---------- trust (used by coordination/consensus.py) ----------

    def update_trust_scores(self, consensus_agreements):
        """
        consensus_agreements: list[float] in [0,1], one per agent - how much
        each agent's proposed action matched the consensus module's final call.
        """
        for agent, agreement in zip(self.agents, consensus_agreements):
            agent.update_trust_score(agreement)

    def get_trust_scores(self):
        return [agent.trust_score for agent in self.agents]

    # ---------- checkpointing ----------

    def save_checkpoint(self, dir_path: str):
        """Saves every agent's actor + the shared critic."""
        os.makedirs(dir_path, exist_ok=True)
        for agent in self.agents:
            agent.save(os.path.join(dir_path, f"actor_{agent.agent_id}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(dir_path, "critic.pt"))

    def load_checkpoint(self, dir_path: str):
        for agent in self.agents:
            agent.load(os.path.join(dir_path, f"actor_{agent.agent_id}.pt"))
        self.critic.load_state_dict(torch.load(os.path.join(dir_path, "critic.pt")))
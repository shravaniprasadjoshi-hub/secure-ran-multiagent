"""
training/train.py - Main training loop for secure multi-agent RAN system
Owner: Shreya
Reads from: ran_env.py (Shashank), agent_manager.py (Shravani)
Outputs to: results/logs/, results/checkpoints/
Shloka + Shravani: consensus.py + trust.py hooked in after baseline training works
"""

import os
import yaml
import numpy as np
import torch
from datetime import datetime

from env.ran_env import RANEnv
from agents.agent_manager import AgentManager

# IMPORTANT
# consensus + trust imports - ill uncomment once Shravani + Shloka finish
# from coordination.consensus import ConsensusModule
# from coordination.trust import TrustModule
from security.anomaly_detector import AnomalyDetector

# load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# constants from config
NUM_CELLS       = cfg["env"]["num_cells"]
NUM_UES         = cfg["env"]["num_ues"]
OBS_DIM         = cfg["env"]["obs_dim"]
ACTION_DIM      = cfg["env"]["action_dim"]
MAX_EPISODES    = cfg["training"]["max_episodes"]
MAX_STEPS       = cfg["training"]["max_steps_per_episode"]
UPDATE_INTERVAL = cfg["training"]["update_interval"] # steps between PPO updates
CHECKPOINT_DIR  = cfg["training"]["checkpoint_dir"]
LOG_DIR         = cfg["training"]["log_dir"]
SAVE_EVERY      = cfg["training"]["save_every"] # save checkpoint every N episodes
SEED            = cfg["training"]["seed"]

# reproducibility
# DONT TOUCH: seed must be set before env + agent init
torch.manual_seed(SEED)
np.random.seed(SEED)

# logging setup
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"run_{run_id}.csv")

def init_log(path: str):
    with open(path, "w") as f:
        f.write("episode,step,actor_loss,critic_loss,ho_success,ping_pong,missed_ho,mean_reward\n")

def write_log(path: str, row: dict):
    with open(path, "a") as f:
        f.write(",".join(str(row[k]) for k in [
            "episode", "step", "actor_loss", "critic_loss",
            "ho_success", "ping_pong", "missed_ho", "mean_reward"
        ]) + "\n")


def obs_dict_to_list(obs_dict: dict, agent_ids: list) -> list:
    """
    Converts PettingZoo obs dict to ordered list for AgentManager.
    - agent_ids order must match AgentManager.agents order
    - AgentManager expects list not dict
    """
    return [obs_dict[agent_id] for agent_id in agent_ids]


def rewards_dict_to_list(rewards_dict: dict, agent_ids: list) -> list:
    return [rewards_dict.get(agent_id, 0.0) for agent_id in agent_ids]


def dones_dict_to_list(truncations: dict, terminations: dict, agent_ids: list) -> list:
    return [
        float(truncations.get(a, False) or terminations.get(a, False))
        for a in agent_ids
    ]


# main training loop

def train():
    # env + agent setup
    env = RANEnv(num_cells=NUM_CELLS, num_ues=NUM_UES)
    manager = AgentManager(
        num_agents=NUM_CELLS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    )

    # agent string IDs from PettingZoo env: ["cell_0", ..., "cell_6"]
    agent_ids = env.possible_agents

    init_log(log_path)
    print(f"Training started | run_id: {run_id} | episodes: {MAX_EPISODES}")

    global_step = 0

    for episode in range(1, MAX_EPISODES + 1):
        obs_dict, _ = env.reset(seed=SEED + episode)
        obs_list = obs_dict_to_list(obs_dict, agent_ids)

        episode_rewards = []
        step = 0

        while step < MAX_STEPS:
            # rollout phase
            actions, log_probs, _ = manager.select_actions(obs_list)
            value = manager.get_value(obs_list)

            # convert actions list to PettingZoo dict format
            actions_dict = {agent_id: actions[i] for i, agent_id in enumerate(agent_ids)}

            # step environment
            next_obs_dict, rewards_dict, terminations, truncations, info = env.step(actions_dict)

            rewards_list = rewards_dict_to_list(rewards_dict, agent_ids)
            dones_list = dones_dict_to_list(truncations, terminations, agent_ids)

            # store transitions
            manager.store_transitions(
                obs_list, actions, log_probs, rewards_list, dones_list, value
            )

            episode_rewards.append(np.mean(rewards_list))
            obs_list = obs_dict_to_list(next_obs_dict, agent_ids)
            step += 1
            global_step += 1

            # PPO update
            # CRITICAL: update interval controls how often PPO runs
            # too frequent = unstable
            # too rare = stale policy
            if global_step % UPDATE_INTERVAL == 0:
                losses = manager.update(obs_list)

                metrics = info.get("metrics", {})
                log_row = {
                    "episode": episode,
                    "step": global_step,
                    "actor_loss": round(losses["actor_loss"], 4),
                    "critic_loss": round(losses["critic_loss"], 4),
                    "ho_success": metrics.get("ho_success", 0),
                    "ping_pong": metrics.get("ping_pong", 0),
                    "missed_ho": metrics.get("missed_ho", 0),
                    "mean_reward": round(float(np.mean(episode_rewards)), 4),
                }
                write_log(log_path, log_row)

            # check if all agents done
            if all(truncations.get(a, False) or terminations.get(a, False) for a in agent_ids):
                break

        mean_ep_reward = np.mean(episode_rewards)
        if episode % 10 == 0:
            print(
                f"Ep {episode:04d} | "
                f"reward: {mean_ep_reward:.3f} | "
                f"steps: {step} | "
                f"HO success: {info.get('metrics', {}).get('ho_success', 0)}"
            )

        # checkpoint
        # IMPORTANT: checkpoints saved every SAVE_EVERY episodes
        # results/checkpoints/ - shashank DO NOT touch this folder
        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ep_{episode:04d}")
            manager.save_checkpoint(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\nTraining complete. Log: {log_path}")
    manager.save_checkpoint(os.path.join(CHECKPOINT_DIR, "final"))


if __name__ == "__main__":
    train()
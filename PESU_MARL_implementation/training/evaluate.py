"""
training/evaluate.py: Rollout + metric aggregation for eval (live model or MATLAB replay)
Owners: Shreya
Depends on: env/ran_env.py, agents/agent_manager.py
Usage: python training/evaluate.py --config training/config.yaml --checkpoint <dir> [--replay]
"""

import argparse
import os

import yaml

from env.ran_env import (
    RANEnv, OBS_DIM, ACTION_DIM,
    REWARD_SUCCESSFUL_HO, REWARD_PING_PONG, REWARD_RLF, REWARD_HEALTHY_DEFER,
)
from agents.agent_manager import AgentManager


def run_episode(env: RANEnv, manager: AgentManager, agent_order, deterministic: bool):
    """
    Rolls out one episode. Does NOT store transitions / call manager.update()
    - pure inference, safe to call mid-training for periodic eval without
    disturbing the PPO buffer.
    Returns: dict of per-episode aggregate metrics.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    counts = {"successful_ho": 0, "ping_pong": 0, "rlf": 0, "defer": 0}
    steps = 0

    while True:
        obs_list = [obs[a] for a in agent_order]
        actions, _, _ = manager.select_actions(obs_list, deterministic=deterministic)
        action_dict = {a: act for a, act in zip(agent_order, actions)}
        next_obs, rewards, terms, truncs, infos = env.step(action_dict)

        for a in agent_order:
            r = rewards[a]
            total_reward += r
            if r == REWARD_SUCCESSFUL_HO:
                counts["successful_ho"] += 1
            elif r == REWARD_PING_PONG:
                counts["ping_pong"] += 1
            elif r == REWARD_RLF:
                counts["rlf"] += 1
            elif r == REWARD_HEALTHY_DEFER:
                counts["defer"] += 1

        steps += 1
        if not env.agents:
            break
        obs = next_obs

    return {
        "total_reward": total_reward,
        "avg_reward_per_agent": total_reward / (len(agent_order) * steps),
        "steps": steps,
        **counts,
    }


def evaluate(env: RANEnv, manager: AgentManager, agent_order, num_episodes: int = 5,
             deterministic: bool = True):
    """Averages run_episode metrics over num_episodes. Returns dict of means."""
    episode_metrics = [run_episode(env, manager, agent_order, deterministic) for _ in range(num_episodes)]
    keys = episode_metrics[0].keys()
    return {k: sum(m[k] for m in episode_metrics) / num_episodes for k in keys}


def _print_metrics(tag: str, metrics: dict):
    print(f"[{tag}] " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--checkpoint", required=True, help="dir containing actor_*.pt + critic.pt")
    parser.add_argument("--replay", action="store_true",
                         help="use MATLAB grid replay mode instead of the live analytic model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.replay:
        env = RANEnv(
            num_cells=cfg["env"]["num_cells"],
            max_steps=cfg["env"]["max_steps"],
            replay_rsrp_path=cfg["eval"]["replay_rsrp_path"],
            replay_sinr_path=cfg["eval"]["replay_sinr_path"],
        )
        num_episodes = cfg["eval"]["replay_episodes"]
        tag = "REPLAY (MATLAB 3GPP-traceable)"
    else:
        env = RANEnv(
            num_cells=cfg["env"]["num_cells"],
            max_steps=cfg["env"]["max_steps"],
            seed=cfg["env"]["seed"],
        )
        num_episodes = cfg["eval"]["eval_episodes"]
        tag = "LIVE MODEL"

    agent_order = [f"cell_{i}" for i in range(cfg["env"]["num_cells"])]
    manager = AgentManager(
        num_agents=cfg["env"]["num_cells"],
        obs_dim=cfg["model"]["obs_dim"],
        action_dim=cfg["model"]["action_dim"],
    )
    manager.load_checkpoint(args.checkpoint)

    metrics = evaluate(env, manager, agent_order, num_episodes=num_episodes,
                        deterministic=cfg["eval"]["deterministic"])
    _print_metrics(tag, metrics)


if __name__ == "__main__":
    main()
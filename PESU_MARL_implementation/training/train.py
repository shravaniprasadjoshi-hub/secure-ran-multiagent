"""
training/train.py: Main training loop for MAPPO on RANEnv
Owners: Shreyashree (primary), Shravani
Depends on: env/ran_env.py, agents/agent_manager.py, training/evaluate.py, training/config.yaml
Usage: python training/train.py --config training/config.yaml
"""

import argparse
import csv
import os

import yaml

from env.ran_env import RANEnv
from agents.agent_manager import AgentManager
from training.evaluate import evaluate, run_episode as eval_run_episode  # noqa: F401 - eval_run_episode kept for notebook/debug use


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_training_episode(env: RANEnv, manager: AgentManager, agent_order):
    """
    Rolls out one full episode, storing every transition into each agent's
    PPO buffer, then runs the update. Returns (episode_reward_total, update_losses).
    """
    obs, _ = env.reset()
    episode_reward_total = 0.0
    last_obs_list = None

    while True:
        obs_list = [obs[a] for a in agent_order]
        actions, log_probs, _ = manager.select_actions(obs_list, deterministic=False)
        value = manager.get_value(obs_list)
        action_dict = {a: act for a, act in zip(agent_order, actions)}

        next_obs, rewards, terms, truncs, infos = env.step(action_dict)
        reward_list = [rewards[a] for a in agent_order]
        done_list = [terms[a] or truncs[a] for a in agent_order]
        manager.store_transitions(obs_list, actions, log_probs, reward_list, done_list, value)

        episode_reward_total += sum(reward_list)

        if not env.agents:
            # env clears self.agents once every agent is done/truncated -
            # next_obs still has all keys from before the clear, safe to index
            last_obs_list = [next_obs[a] for a in agent_order]
            break
        obs = next_obs
        last_obs_list = obs_list

    losses = manager.update(last_obs_list)
    return episode_reward_total, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, model_cfg, train_cfg, eval_cfg = cfg["env"], cfg["model"], cfg["training"], cfg["eval"]

    agent_order = [f"cell_{i}" for i in range(env_cfg["num_cells"])]

    env = RANEnv(num_cells=env_cfg["num_cells"], max_steps=env_cfg["max_steps"], seed=env_cfg["seed"])
    manager = AgentManager(num_agents=env_cfg["num_cells"], obs_dim=model_cfg["obs_dim"],
                            action_dim=model_cfg["action_dim"])

    if args.resume:
        manager.load_checkpoint(args.resume)
        print(f"resumed from {args.resume}")

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(train_cfg["log_csv_path"]), exist_ok=True)

    log_fields = ["episode", "total_reward", "actor_loss", "critic_loss"]
    with open(train_cfg["log_csv_path"], "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    for episode in range(1, train_cfg["num_episodes"] + 1):
        total_reward, losses = run_training_episode(env, manager, agent_order)

        with open(train_cfg["log_csv_path"], "a", newline="") as f:
            csv.writer(f).writerow([episode, total_reward, losses["actor_loss"], losses["critic_loss"]])

        if episode % train_cfg["log_every"] == 0:
            print(f"ep {episode:5d} | reward={total_reward:8.2f} | "
                  f"actor_loss={losses['actor_loss']:.4f} | critic_loss={losses['critic_loss']:.4f} | "
                  f"trust={[round(t, 2) for t in manager.get_trust_scores()]}")

        if episode % train_cfg["checkpoint_every"] == 0:
            ckpt_path = os.path.join(train_cfg["checkpoint_dir"], f"ep_{episode}")
            manager.save_checkpoint(ckpt_path)

        if episode % eval_cfg["eval_every"] == 0:
            eval_metrics = evaluate(env, manager, agent_order, num_episodes=eval_cfg["eval_episodes"],
                                     deterministic=eval_cfg["deterministic"])
            print(f"  [eval @ ep {episode}] " + " | ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))

    final_ckpt = os.path.join(train_cfg["checkpoint_dir"], "final")
    manager.save_checkpoint(final_ckpt)
    print(f"training done, final checkpoint at {final_ckpt}")


if __name__ == "__main__":
    main()
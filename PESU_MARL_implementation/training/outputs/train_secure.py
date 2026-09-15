"""
training/train_secure.py: MAPPO training with full security stack + PQC (Exp 3 + Exp 4 combined)
Owner: Shreyashree
Depends on: env/ran_env.py, agents/agent_manager.py,
            security/{byzantine,anomaly_detector,trust,consensus,policy_checker,pqc_channel}.py
Usage: python training/train_secure.py --config training/config.yaml
#
# Pipeline per step, per pqc_channel.py's own integration note:
#   agent policy -> byzantine corruption (behavioral attack sim, only for
#   compromised agent ids) -> PQC send/receive (transit attack sim - tamper/
#   forge detection) -> anomaly detection on the PQC-verified actions ->
#   trust update -> consensus (DIAGNOSTIC ONLY - config.yaml says
#   min_agreement is diagnostic, doesn't override actions) -> policy check ->
#   effective action applied to env: quarantined OR PQC-failed agents get
#   their action forced to defer (0), everyone else's verified action is used.
#
# IMPORTANT: PPO training signal (manager.store_transitions) uses each
# agent's own RAW sampled action + log_prob, not the byzantine-corrupted or
# quarantine-forced one - byzantine/PQC attacks corrupt what reaches the
# coordinator, not what the policy is trained on. If your Exp 3 script did
# this differently, that's a real divergence, not a typo - flag it.
"""

import argparse
import csv
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.ran_env import RANEnv
from agents.agent_manager import AgentManager
from security.anomaly_detector import AnomalyDetector
from torch.distributions import Categorical
from coordination.trust import TrustManager
from coordination.consensus import ConsensusEngine
from security.policy_checker import PolicyChecker
from security.pqc_channel import PQCChannel, simulate_tamper
from security.byzantine import ByzantineFaultInjector 



def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_security_stack(sec_cfg: dict, num_agents: int, action_dim: int):
    injector = ByzantineFaultInjector(total_agents=num_agents)
    for agent_id in sec_cfg.get("byzantine_agents", []):
        attack_type = sec_cfg.get("attack_types", {}).get(agent_id, "random")
        injector.inject(agent_id, attack_type=attack_type, drift_rate=sec_cfg.get("drift_rate", 0.1))

    anomaly = AnomalyDetector(n_agents=num_agents, window_size=sec_cfg["anomaly_window_size"], threshold=sec_cfg["anomaly_threshold"])
    policy = PolicyChecker(n_agents=num_agents, action_space_size=action_dim, max_resource=sec_cfg["max_resource"])
    trust = TrustManager(n_agents=num_agents, initial_trust=sec_cfg["initial_trust"], decay_rate=sec_cfg["decay_rate"], recovery_rate=sec_cfg["recovery_rate"], min_trust=sec_cfg["min_trust"], max_trust=sec_cfg["max_trust"])
    consensus = ConsensusEngine(n_agents=num_agents, action_space_size=action_dim, min_agreement=sec_cfg["min_agreement"])
    return injector, anomaly, policy, trust, consensus


def build_pqc_channel(sec_cfg: dict, num_agents: int):
    if not sec_cfg.get("enable_pqc", False):
        return None
    try:
        channel = PQCChannel()
        for i in range(num_agents):
            channel.register_agent(i)
        print(f"[PQC] channel ready — {channel.kem_alg} / {channel.sig_alg}")
        return channel
    except Exception as e:
        print(f"[PQC] unavailable ({e}) — training falls back to Exp 3 (no PQC layer)")
        return None


def run_secure_episode(env, manager, agent_order,
                        injector, anomaly, policy, trust, consensus,
                        quarantine_threshold, rsrp_threshold, sinr_threshold,
                        channel=None):
    obs, _ = env.reset()
    episode_reward_total = 0.0
    injector.reset_all(); anomaly.reset(); policy.reset(); trust.reset(); consensus.reset()

    n = len(agent_order)
    compromised_ids = set(injector.compromised_agents.keys())

    if channel is not None:
        for i in range(n):
            channel.establish_session(i)

    steps = 0
    consensus_ok_count = 0
    byzantine_true_positive = 0
    byzantine_total_steps = 0
    false_positive_count = 0
    clean_total_steps = 0
    pqc_true_positive = 0
    pqc_total_steps = 0

    while True:
        obs_list = [obs[a] for a in agent_order]
        proposed_actions, _, _ = manager.select_actions(obs_list, deterministic=False)

        corrupted_actions = {
            i: injector.get_action(i, proposed_actions[i], manager.action_dim) for i in range(n)
        }

        pqc_flagged = []
        if channel is not None:
            for i in range(n):
                msg = channel.send_action(agent_id=i, step=steps, action=corrupted_actions[i])
                if i in compromised_ids:
                    # compromised agent also spoofs the transmitted message,
                    # not just its decision — separate attack surface from byzantine.py
                    msg = simulate_tamper(msg, tamper_type="flip_action")
                _, verified = channel.receive_action(msg)
                if not verified:
                    pqc_flagged.append(i)

        flagged = anomaly.run_all_detectors(corrupted_actions)
        all_flagged = list(set(flagged) | set(pqc_flagged))   # merged severity

        rsrp_map, sinr_map, binarized = {}, {}, {}
        for i in range(n):
            ue = env.ues[i]
            avg_rsrp, avg_sinr = ue.avg_rsrp_per_cell(), ue.avg_sinr_per_cell()
            serving = ue.serving_cell_id
            rsrp_map[i] = avg_rsrp[serving] if avg_rsrp else 0.0
            sinr_map[i] = avg_sinr[serving] if avg_sinr else 0.0
            binarized[i] = 1 if corrupted_actions[i] in (1, 2) else 0
        policy_results = policy.validate_all(binarized, rsrp_map=rsrp_map, sinr_map=sinr_map)

        # trust now sees the MERGED flagged set (behavioral + message-layer)
        trust.update_on_anomaly(all_flagged, clean_agents=[i for i in range(n) if i not in all_flagged])
        trust.update_on_policy(policy_results)

        trust_scores = trust.get_trust_scores()
        _, agreement, consensus_ok = consensus.reach_consensus(
            corrupted_actions, flagged_agents=all_flagged, trust_scores=trust_scores
        )

        quarantined = set(trust.get_quarantined_agents(threshold=quarantine_threshold))
        final_actions = {i: (0 if i in quarantined else corrupted_actions[i]) for i in range(n)}

        steps += 1
        consensus_ok_count += int(bool(consensus_ok))
        for i in range(n):
            if i in compromised_ids:
                byzantine_total_steps += 1
                byzantine_true_positive += int(i in flagged)
                if channel is not None:
                    pqc_total_steps += 1
                    pqc_true_positive += int(i in pqc_flagged)
            else:
                clean_total_steps += 1
                false_positive_count += int(i in flagged)

        action_dict = {a: final_actions[i] for i, a in enumerate(agent_order)}
        value = manager.get_value(obs_list)
        next_obs, rewards, terms, truncs, infos = env.step(action_dict)
        reward_list = [rewards[a] for a in agent_order]
        done_list = [terms[a] or truncs[a] for a in agent_order]
        episode_reward_total += sum(reward_list)

        for i in range(n):
            if i in quarantined:
                continue
            obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32)
            with torch.no_grad():
                logits = manager.agents[i].actor(obs_t)
                log_prob = Categorical(logits=logits).log_prob(torch.tensor(final_actions[i]))
            manager.agents[i].store_transition(
                obs=obs_list[i], global_obs=manager.build_global_obs(obs_list),
                action=final_actions[i], log_prob=log_prob,
                reward=reward_list[i], done=done_list[i], value=value,
            )

        if not env.agents:
            last_obs_list = [next_obs[a] for a in agent_order]
            break
        obs = next_obs
        last_obs_list = obs_list

    empty_agents = [i for i, a in enumerate(manager.agents) if len(a.buffer) == 0]
    if empty_agents:
        print(f"  [WARN] agents {empty_agents} fully quarantined this episode - skipping PPO update")
        losses = {"actor_loss": float("nan"), "critic_loss": float("nan")}
        for a in manager.agents:
            a.buffer.clear()
    else:
        losses = manager.update(last_obs_list)

    episode_metrics = {
        "consensus_rate": consensus_ok_count / steps if steps else 0.0,
        "byzantine_detection_rate": (byzantine_true_positive / byzantine_total_steps) if byzantine_total_steps else float("nan"),
        "false_positive_rate": (false_positive_count / clean_total_steps) if clean_total_steps else 0.0,
        "avg_trust_score": float(np.mean(list(trust.get_trust_scores().values()))),
        "pqc_detection_rate": (pqc_true_positive / pqc_total_steps) if pqc_total_steps else float("nan"),  # NEW
    }
    return episode_reward_total, losses, episode_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, model_cfg, train_cfg, sec_cfg = cfg["env"], cfg["model"], cfg["training"], cfg["security"]

    agent_order = [f"cell_{i}" for i in range(env_cfg["num_cells"])]
    env = RANEnv(num_cells=env_cfg["num_cells"], max_steps=env_cfg["max_steps"], seed=env_cfg["seed"])
    manager = AgentManager(num_agents=env_cfg["num_cells"], obs_dim=model_cfg["obs_dim"],
                            action_dim=model_cfg["action_dim"])

    if args.resume:
        manager.load_checkpoint(args.resume)
        print(f"resumed from {args.resume}")

    injector, anomaly, policy, trust, consensus = build_security_stack(
        sec_cfg, num_agents=env_cfg["num_cells"], action_dim=model_cfg["action_dim"]
    )
    channel = build_pqc_channel(sec_cfg, num_agents=env_cfg["num_cells"])

    ckpt_dir = train_cfg.get("checkpoint_dir_secure", "training/outputs/checkpoints_secure")
    log_path = train_cfg.get("log_csv_path_secure", "training/outputs/train_log_secure.csv")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_fields = ["episode", "total_reward", "actor_loss", "critic_loss",
                  "consensus_rate", "byzantine_detection_rate", "false_positive_rate",
                  "avg_trust_score", "pqc_detection_rate"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    for episode in range(1, train_cfg["num_episodes"] + 1):
        total_reward, losses, metrics = run_secure_episode(
            env, manager, agent_order, injector, anomaly, policy, trust, consensus,
            quarantine_threshold=sec_cfg["quarantine_threshold"],
            rsrp_threshold=sec_cfg["rsrp_threshold"], sinr_threshold=sec_cfg["sinr_threshold"],
            channel=channel,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([episode, total_reward, losses["actor_loss"], losses["critic_loss"],
                                     metrics["consensus_rate"], metrics["byzantine_detection_rate"],
                                     metrics["false_positive_rate"], metrics["avg_trust_score"],
                                     metrics["pqc_detection_rate"]])

        if episode % train_cfg["log_every"] == 0:
            print(f"ep {episode:5d} | reward={total_reward:8.2f} | "
                  f"byz_detect={metrics['byzantine_detection_rate']:.2f} | "
                  f"pqc_detect={metrics['pqc_detection_rate']:.2f} | "
                  f"fp_rate={metrics['false_positive_rate']:.2f} | "
                  f"avg_trust={metrics['avg_trust_score']:.2f}")

        if episode % train_cfg["checkpoint_every"] == 0:
            manager.save_checkpoint(os.path.join(ckpt_dir, f"ep_{episode}"))

    manager.save_checkpoint(os.path.join(ckpt_dir, "final"))
    if channel is not None:
        channel.summary()
        channel.close()
    print(f"secure training done, log at {log_path}")


if __name__ == "__main__":
    main()
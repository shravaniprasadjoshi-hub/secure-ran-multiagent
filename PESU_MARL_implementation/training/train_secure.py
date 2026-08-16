"""
training/train_secure.py: 
MAPPO training loop with the full security stack wired in.

Owners: Shreyashree

Depends on: env/ran_env.py, agents/agent_manager.py,
            security/byzantine.py, security/anomaly_detector.py, security/policy_checker.py,
            coordination/trust.py, coordination/consensus.py

Usage: python training/train_secure.py --config training/config.yaml

train.py (no security) stays untouched

# IMPORTANT ARCHITECTURE NOTE - read before touching consensus wiring:
# ConsensusEngine.reach_consensus() is built for ONE shared decision across all agents (majority/weighted vote picks a single final_action)
# Our agents each independently control handover for their OWN UE - there's no single "correct" action all 7 should agree on
# Feeding all 7 agents' per-cell actions into reach_consensus() and using its final_action to drive every
# agent's execution would be semantically wrong (agent 3's UE would execute agent 0's decision)
#
# Resolution used here:
#   - consensus.reach_consensus() IS still called every step, but its (agreement, consensus_ok) output is used ONLY as a diagnostic/logging
#     signal (consensus_rate metric) 
#   - TrustManager.update_on_consensus() is deliberately NOT called - it also needs one "correct" final_action to compare against, same mismatch.
#     Trust updates come from update_on_anomaly + update_on_policy only, both genuinely per-agent.
#   - What actually executes per agent, each step:
#       1. Byzantine-compromised agents (per config) - ByzantineFaultInjector corrupts their action BEFORE execution (the attacker's real effect)
#       2. Quarantined agents (trust < quarantine_threshold) - action is force-overridden to defer(0), the RLF-safe fallback, regardless of
#          what was proposed - this is what "excluded from consensus" means in practice per trust.py's docstring
#       3. Everyone else - their own MAPPO action executes unchanged 
#   - PPO training: log_prob is recomputed under each agent's current policy for whatever action ACTUALLY executed (so credit assignment stays
#     internally consistent - see MAPPOAgent.update()'s ratio calc), EXCEPT quarantined agents, whose transition is skipped entirely that step -
#     they didn't choose defer, we forced it, so we don't want the policy gradient crediting/blaming them for an outcome their own action didn't cause.
#
# KNOWN LIMITATION: AnomalyDetector's voting/statistical detectors assume agents *should* behave similarly - flags agents who differ from the group. 
# That's not strictly true here (agents legitimately differ because they observe different UEs/local conditions). 
"""

import argparse
import csv
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
import torch
from torch.distributions import Categorical
import yaml
 
from env.ran_env import RANEnv
from agents.agent_manager import AgentManager
from security.byzantine import ByzantineFaultInjector
from security.anomaly_detector import AnomalyDetector
from security.policy_checker import PolicyChecker
from coordination.trust import TrustManager
from coordination.consensus import ConsensusEngine
 
 
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def build_security_stack(sec_cfg: dict, num_agents: int, action_dim: int):
    injector = ByzantineFaultInjector(total_agents=num_agents)
    for agent_id in sec_cfg.get("byzantine_agents", []):
        attack_type = sec_cfg.get("attack_types", {}).get(agent_id, "random")
        injector.inject(agent_id, attack_type=attack_type, drift_rate=sec_cfg.get("drift_rate", 0.1))
 
    anomaly = AnomalyDetector(
        n_agents=num_agents,
        window_size=sec_cfg.get("anomaly_window_size", 20),
        threshold=sec_cfg.get("anomaly_threshold", 2.0),
    )
    policy = PolicyChecker(
        n_agents=num_agents,
        action_space_size=action_dim,
        max_resource=sec_cfg.get("max_resource", 100),
    )
    trust = TrustManager(
        n_agents=num_agents,
        initial_trust=sec_cfg.get("initial_trust", 1.0),
        decay_rate=sec_cfg.get("decay_rate", 0.05),
        recovery_rate=sec_cfg.get("recovery_rate", 0.02),
        min_trust=sec_cfg.get("min_trust", 0.1),
        max_trust=sec_cfg.get("max_trust", 1.0),
    )
    consensus = ConsensusEngine(
        n_agents=num_agents,
        action_space_size=action_dim,
        min_agreement=sec_cfg.get("min_agreement", 0.6),
    )
    return injector, anomaly, policy, trust, consensus
 
 
def run_secure_episode(env: RANEnv, manager: AgentManager, agent_order,
                        injector, anomaly, policy, trust, consensus,
                        quarantine_threshold: float, rsrp_threshold: float, sinr_threshold: float):
    """Rolls out one episode with the full security stack wired in. Returns (reward, losses, metrics)."""
    obs, _ = env.reset()
    injector.reset_all()
    anomaly.reset()
    policy.reset()
    trust.reset()
    consensus.reset()
 
    episode_reward_total = 0.0
    last_obs_list = None
    n = len(agent_order)
    compromised_ids = set(injector.compromised_agents.keys())
 
    steps = 0
    consensus_ok_count = 0
    byzantine_true_positive = 0
    byzantine_total_steps = 0
    false_positive_count = 0
    clean_total_steps = 0
 
    while True:
        obs_list = [obs[a] for a in agent_order]
        proposed_actions, _, _ = manager.select_actions(obs_list, deterministic=False)
 
        # 1. Byzantine fault injection
        corrupted_actions = {
            i: injector.get_action(i, proposed_actions[i], manager.action_dim) for i in range(n)
        }
 
        # 2. anomaly detection on what was actually proposed (post-corruption)
        flagged = anomaly.run_all_detectors(corrupted_actions)
 
        # 3. policy checking - per agent, using its own rsrp/sinr.
        # handover_policy check is binary (1=handover); our action space is
        # 3-way {defer, trigger-best, trigger-2nd} - binarize: any trigger
        # counts as "handover triggered".
        rsrp_map, sinr_map, binarized = {}, {}, {}
        for i in range(n):
            ue = env.ues[i]
            avg_rsrp, avg_sinr = ue.avg_rsrp_per_cell(), ue.avg_sinr_per_cell()
            serving = ue.serving_cell_id
            rsrp_map[i] = avg_rsrp[serving] if avg_rsrp else 0.0
            sinr_map[i] = avg_sinr[serving] if avg_sinr else 0.0
            binarized[i] = 1 if corrupted_actions[i] in (1, 2) else 0
        policy_results = policy.validate_all(binarized, rsrp_map=rsrp_map, sinr_map=sinr_map)
 
        # 4. trust updates - anomaly + policy only, see module docstring for why
        # consensus-based trust update is skipped
        trust.update_on_anomaly(flagged, clean_agents=[i for i in range(n) if i not in flagged])
        trust.update_on_policy(policy_results)
 
        # 5. consensus - DIAGNOSTIC METRIC ONLY, not applied to execution
        trust_scores = trust.get_trust_scores()
        _, agreement, consensus_ok = consensus.reach_consensus(
            corrupted_actions, flagged_agents=flagged, trust_scores=trust_scores
        )
 
        # 6. quarantine - force low-trust agents to the safe default (defer)
        quarantined = set(trust.get_quarantined_agents(threshold=quarantine_threshold))
        final_actions = {i: (0 if i in quarantined else corrupted_actions[i]) for i in range(n)}
 
        # ---- metrics bookkeeping ----
        steps += 1
        consensus_ok_count += int(bool(consensus_ok))
        for i in range(n):
            if i in compromised_ids:
                byzantine_total_steps += 1
                byzantine_true_positive += int(i in flagged)
            else:
                clean_total_steps += 1
                false_positive_count += int(i in flagged)
 
        # ---- execute ----
        action_dict = {a: final_actions[i] for i, a in enumerate(agent_order)}
        value = manager.get_value(obs_list)
        next_obs, rewards, terms, truncs, infos = env.step(action_dict)
        reward_list = [rewards[a] for a in agent_order]
        done_list = [terms[a] or truncs[a] for a in agent_order]
        episode_reward_total += sum(reward_list)
 
        # 7. PPO storage - recompute log_prob under current policy for the
        # action that ACTUALLY executed (byzantine-corrupted actions DO get
        # trained on - that's the attack's real consequence). Quarantined
        # agents are skipped - forced action, not the policy's own choice.
        for i in range(n):
            if i in quarantined:
                continue
            obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32)
            with torch.no_grad():
                logits = manager.agents[i].actor(obs_t)
                log_prob = Categorical(logits=logits).log_prob(torch.tensor(final_actions[i]))
            manager.agents[i].store_transition(
                obs=obs_list[i],
                global_obs=manager.build_global_obs(obs_list),
                action=final_actions[i],
                log_prob=log_prob,
                reward=reward_list[i],
                done=done_list[i],
                value=value,
            )
 
        if not env.agents:
            last_obs_list = [next_obs[a] for a in agent_order]
            break
        obs = next_obs
        last_obs_list = obs_list
 
    # guard: if any agent was quarantined for the ENTIRE episode, its buffer
    # is empty and agent_manager.update()'s torch.stack(...) would crash -
    # deliberately not touching agent_manager.py, so skip the update instead
    # and warn, rather than patch the shared/frozen file.
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
    }
    return episode_reward_total, losses, episode_metrics
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    env_cfg, model_cfg, train_cfg, sec_cfg = cfg["env"], cfg["model"], cfg["training"], cfg["security"]
 
    agent_order = [f"cell_{i}" for i in range(env_cfg["num_cells"])]
    env = RANEnv(num_cells=env_cfg["num_cells"], max_steps=env_cfg["max_steps"], seed=env_cfg["seed"])
    manager = AgentManager(num_agents=env_cfg["num_cells"], obs_dim=model_cfg["obs_dim"],
                            action_dim=model_cfg["action_dim"])
    injector, anomaly, policy, trust, consensus = build_security_stack(
        sec_cfg, num_agents=env_cfg["num_cells"], action_dim=model_cfg["action_dim"]
    )
 
    ckpt_dir = train_cfg["checkpoint_dir"] + "_secure"
    log_path = train_cfg["log_csv_path"].replace(".csv", "_secure.csv")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
 
    log_fields = ["episode", "total_reward", "actor_loss", "critic_loss",
                  "consensus_rate", "byzantine_detection_rate", "false_positive_rate", "avg_trust_score"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)
 
    for episode in range(1, train_cfg["num_episodes"] + 1):
        total_reward, losses, metrics = run_secure_episode(
            env, manager, agent_order, injector, anomaly, policy, trust, consensus,
            quarantine_threshold=sec_cfg["quarantine_threshold"],
            rsrp_threshold=sec_cfg["rsrp_threshold"], sinr_threshold=sec_cfg["sinr_threshold"],
        )
 
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([episode, total_reward, losses["actor_loss"], losses["critic_loss"],
                                     metrics["consensus_rate"], metrics["byzantine_detection_rate"],
                                     metrics["false_positive_rate"], metrics["avg_trust_score"]])
 
        if episode % train_cfg["log_every"] == 0:
            print(f"ep {episode:5d} | reward={total_reward:8.2f} | "
                  f"consensus_rate={metrics['consensus_rate']:.2f} | "
                  f"byz_detect={metrics['byzantine_detection_rate']:.2f} | "
                  f"fp_rate={metrics['false_positive_rate']:.2f} | "
                  f"avg_trust={metrics['avg_trust_score']:.2f}")
 
        if episode % train_cfg["checkpoint_every"] == 0:
            manager.save_checkpoint(os.path.join(ckpt_dir, f"ep_{episode}"))
 
    manager.save_checkpoint(os.path.join(ckpt_dir, "final"))
    print(f"secure training done, log at {log_path}")
 
 
if __name__ == "__main__":
    main()
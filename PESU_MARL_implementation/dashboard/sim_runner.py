"""
dashboard/sim_runner.py — background MAPPO eval-episode runner + shared live-sim state
Extracted out of api.py so the sim/env logic isn't coupled to the HTTP layer.
api.py imports `sim_state` and `run_eval_episode` from here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# same layout convention as api.py
PESU_DIR = Path(__file__).resolve().parent.parent

# train.py -> training/outputs/checkpoints/final (MAPPO only)
# train_secure.py -> training/outputs/checkpoints_secure/final (MAPPO + security)
BASE_CHECKPOINT_DIR = PESU_DIR / "training" / "outputs" / "checkpoints" / "final"
SECURE_CHECKPOINT_DIR = PESU_DIR / "training" / "outputs" / "checkpoints_secure" / "final"

# Shared simulation state, polled by the frontend via GET /state
sim_state: dict = {
    "agents": [
        {"id": i, "name": f"Cell {i}", "trust": 1.0,
         "status": "healthy", "load": 50, "ho_rate": 60}
        for i in range(7)
    ],
    "consensus_rate": 0.72,
    "byzantine_count": 0,
    "step": 0,
    "consensus_log": [],
    "alerts": [],
    "running": False,
}


def run_eval_episode(use_secure: bool = True):
    """
    Loads trained MAPPO checkpoint and runs one eval episode.
    Updates sim_state every step so the frontend sees live trust/status changes.
    Imports are inside the function so the API can start even if torch isn't installed.
    Intended to be run in a background thread (see api.py: POST /start-sim).

    use_secure: prefer training/outputs/checkpoints_secure/final (MAPPO + security)
                falls back to training/outputs/checkpoints/final if secure doesn't
                exist yet, and to an untrained/random policy if neither exists.
    """
    sys.path.insert(0, str(PESU_DIR))

    checkpoint_dir = SECURE_CHECKPOINT_DIR if (use_secure and SECURE_CHECKPOINT_DIR.exists()) else BASE_CHECKPOINT_DIR

    try:
        from env.ran_env import RANEnv
        from agents.agent_manager import AgentManager
        from security.byzantine import ByzantineFaultInjector
        from security.anomaly_detector import AnomalyDetector
        from security.policy_checker import PolicyChecker
        from coordination.consensus import ConsensusEngine
        from coordination.trust import TrustManager

        sim_state["running"] = True

        env = RANEnv()
        obs, _ = env.reset()
        agent_list = env.agents
        n_agents = len(agent_list)

        manager = AgentManager(num_agents=n_agents, obs_dim=3, action_dim=3)
        if checkpoint_dir.exists():
            manager.load_checkpoint(str(checkpoint_dir))
            sim_state["alerts"].insert(0, {
                "type": "System",
                "msg": f"Loaded checkpoint: {checkpoint_dir.relative_to(PESU_DIR)}",
                "time": "Just now"
            })
        else:
            sim_state["alerts"].insert(0, {
                "type": "System",
                "msg": "No checkpoint found — running with untrained/random policy",
                "time": "Just now"
            })

        # preserve any byzantine agents injected from the frontend before this run started
        injector = ByzantineFaultInjector(total_agents=n_agents)
        for ag in sim_state["agents"]:
            if ag["status"] == "byzantine":
                injector.inject(agent_id=ag["id"], attack_type="random")

        detector = AnomalyDetector(n_agents=n_agents, window_size=20, threshold=3.0)
        checker = PolicyChecker(n_agents=n_agents, action_space_size=3)
        consensus = ConsensusEngine(n_agents=n_agents, action_space_size=3)
        trust = TrustManager(n_agents=n_agents)

        # seed trust scores from current sim_state
        for ag in sim_state["agents"]:
            trust.trust_scores[ag["id"]] = ag["trust"]

        for step in range(200):
            obs_array = np.array(
                [obs[agent] for agent in agent_list], dtype=np.float32
            )
            actions_tuple = manager.select_actions(obs_array)
            actions_raw = actions_tuple[0]

            actions = {
                i: injector.get_action(i, int(actions_raw[i]), 3)
                for i in range(n_agents)
            }

            flagged = detector.run_all_detectors(actions)
            policy_results = checker.validate_all(actions)
            trust.update_on_anomaly(flagged)
            trust.update_on_policy(policy_results)

            trust_weights = trust.get_trust_weights()
            final_action, agreement, ok = consensus.reach_consensus(
                actions, flagged_agents=flagged, trust_scores=trust_weights
            )

            # push results into shared state
            scores = trust.get_trust_scores()
            quarantined = trust.get_quarantined_agents()

            for i, ag in enumerate(sim_state["agents"]):
                ag["trust"] = round(scores[i], 3)
                if i in flagged:
                    ag["status"] = "byzantine" if injector.is_compromised(i) else "degraded"
                elif i in quarantined:
                    ag["status"] = "degraded"
                elif ag["status"] not in ("byzantine",):
                    ag["status"] = "healthy"

            sim_state["consensus_rate"] = round(agreement or 0.0, 3)
            sim_state["step"] = step
            sim_state["byzantine_count"] = sum(
                1 for ag in sim_state["agents"] if ag["status"] == "byzantine"
            )

            sim_state["consensus_log"].insert(0, {
                "step": step + 1,
                "agreement": round((agreement or 0) * 100, 1),
                "ok": ok,
                "final_action": final_action,
                "excluded": flagged,
            })
            sim_state["consensus_log"] = sim_state["consensus_log"][:10]

            env_actions = {
                agent: int(actions_raw[i])
                for i, agent in enumerate(agent_list)
            }
            obs, _, terminations, truncations, _ = env.step(env_actions)

            time.sleep(0.05)  # ~20 steps/sec so the frontend can keep up

            if all(terminations.values()) or all(truncations.values()):
                break

    except Exception as e:
        sim_state["alerts"].insert(0, {
            "type": "System",
            "msg": f"Simulation error: {str(e)}",
            "time": "Just now"
        })
    finally:
        sim_state["running"] = False
"""
dashboard/api.py — FastAPI backend for the Secure RAN Multi-Agent Dashboard
Serves data from both Swetha's outputs and our MARL implementation.

Run:
    cd PESU_MARL_implementation
    uvicorn dashboard.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
SWETHA_DIR   = ROOT / "Swetha_proj3_code"
PESU_DIR     = ROOT / "PESU_MARL_implementation"

METRICS_JSON     = SWETHA_DIR / "outputs" / "metrics" / "all_metrics.json"
COORDINATOR_JSON = SWETHA_DIR / "outputs" / "metrics" / "coordinator_metrics.json"
TELEMETRY_CSV    = ROOT / "data" / "ran_multi_agent_telemetry_70k.csv"
TRAINING_LOG     = PESU_DIR / "outputs" / "training_log.csv"
CHECKPOINT_DIR   = PESU_DIR / "results" / "checkpoints" / "final"

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Secure RAN Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared simulation state (updated by sim_runner thread) ─────────────────
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

# ── Helper: safe JSON load ──────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

# ── Telemetry stats (cached on first call) ─────────────────────────────────
_telem_cache: dict = {}

def get_telemetry_stats() -> dict:
    global _telem_cache
    if _telem_cache:
        return _telem_cache

    if not TELEMETRY_CSV.exists():
        return {}

    df = pd.read_csv(TELEMETRY_CSV)

    # RSRP histogram
    rsrp = df["rsrp_dbm"].dropna()
    rsrp_hist, rsrp_edges = np.histogram(rsrp.sample(min(10000, len(rsrp))), bins=30)

    # SINR histogram
    sinr = df["sinr_db"].dropna()
    sinr_hist, sinr_edges = np.histogram(sinr.sample(min(10000, len(sinr))), bins=30)

    # Scenario distribution
    scenario_counts = df["scenario_type"].value_counts().to_dict()

    # Correlation heatmap
    num_cols = ["rsrp_dbm", "rsrq_db", "sinr_db", "cqi",
                "prb_utilization_pct", "trust_score", "attack_probability"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr().round(2).to_dict()

    # CDF — latency
    lat = df["latency_ms"].dropna().sort_values()
    cdf_x = lat.iloc[::max(1, len(lat)//200)].tolist()
    cdf_y = (np.arange(1, len(lat)+1) / len(lat))[::max(1, len(lat)//200)].tolist()

    # SINR by scenario (median)
    sinr_by_scenario = df.groupby("scenario_type")["sinr_db"].median().round(2).to_dict()

    _telem_cache = {
        "total_records": len(df),
        "rsrp": {
            "counts": rsrp_hist.tolist(),
            "edges": rsrp_edges.tolist(),
        },
        "sinr": {
            "counts": sinr_hist.tolist(),
            "edges": sinr_edges.tolist(),
        },
        "scenario_distribution": scenario_counts,
        "correlation": corr,
        "cdf": {"x": cdf_x, "y": cdf_y},
        "sinr_by_scenario": sinr_by_scenario,
    }
    return _telem_cache

# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Live simulation state ───────────────────────────────────────────────
@app.get("/state")
def get_state():
    """Polled every 2s by frontend to update digital twin + security section."""
    return sim_state

# ── 2. Inject Byzantine fault ──────────────────────────────────────────────
@app.post("/inject")
def inject_fault(agent_id: int, attack_type: str = "random"):
    """Mark an agent as Byzantine — frontend calls this when button is pressed."""
    for agent in sim_state["agents"]:
        if agent["id"] == agent_id:
            agent["status"] = "byzantine"
            agent["trust"] = 0.41
            sim_state["byzantine_count"] += 1
            sim_state["alerts"].insert(0, {
                "type": "Byzantine",
                "msg": f"Cell {agent_id} compromised — {attack_type} attack detected",
                "time": "Just now"
            })
            sim_state["alerts"].insert(0, {
                "type": "Consensus",
                "msg": f"Cell {agent_id} excluded from voting",
                "time": "Just now"
            })
            # keep alerts list to last 10
            sim_state["alerts"] = sim_state["alerts"][:10]
            break
    return {"ok": True, "agent_id": agent_id, "attack_type": attack_type}

# ── 3. Clear all faults ────────────────────────────────────────────────────
@app.post("/clear")
def clear_faults():
    """Reset all agents to healthy."""
    for agent in sim_state["agents"]:
        agent["status"] = "healthy"
        agent["trust"] = 1.0
    sim_state["byzantine_count"] = 0
    sim_state["alerts"].insert(0, {
        "type": "Recovered",
        "msg": "All faults cleared — agents restored to healthy",
        "time": "Just now"
    })
    return {"ok": True}

# ── 4. Swetha's agent metrics ──────────────────────────────────────────────
@app.get("/agent-metrics")
def get_agent_metrics():
    """Returns Swetha's sklearn agent F1/R² scores from all_metrics.json."""
    return load_json(METRICS_JSON)

# ── 5. Coordinator metrics ─────────────────────────────────────────────────
@app.get("/coordinator-metrics")
def get_coordinator_metrics():
    """Returns Swetha's consensus accept rate and action distribution."""
    metrics = load_json(METRICS_JSON)
    return metrics.get("coordinator", load_json(COORDINATOR_JSON))

# ── 6. MARL training log ───────────────────────────────────────────────────
@app.get("/training-log")
def get_training_log():
    """Returns our MAPPO training log CSV as JSON for reward curve chart."""
    if not TRAINING_LOG.exists():
        return {"episodes": [], "rewards": [], "actor_loss": [], "critic_loss": []}

    df = pd.read_csv(TRAINING_LOG, header=None,
                     names=["episode", "reward", "actor_loss", "critic_loss"])
    return {
        "episodes":    df["episode"].tolist(),
        "rewards":     df["reward"].tolist(),
        "actor_loss":  df["actor_loss"].tolist(),
        "critic_loss": df["critic_loss"].tolist(),
    }

# ── 7. Telemetry stats ─────────────────────────────────────────────────────
@app.get("/telemetry-stats")
def get_telemetry_stats_endpoint():
    """Returns RSRP/SINR distributions, correlation, CDF from 70k CSV."""
    return get_telemetry_stats()

# ── 8. Consensus log ──────────────────────────────────────────────────────
@app.get("/consensus-log")
def get_consensus_log():
    """Returns last 10 consensus decisions."""
    return {"log": sim_state["consensus_log"]}

# ── 9. Start simulation ───────────────────────────────────────────────────
@app.post("/start-sim")
def start_simulation():
    """Start background eval episode using trained checkpoint."""
    if sim_state["running"]:
        return {"ok": False, "msg": "Already running"}
    thread = threading.Thread(target=run_eval_episode, daemon=True)
    thread.start()
    return {"ok": True, "msg": "Simulation started"}

# ── 10. Health check ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_exists": CHECKPOINT_DIR.exists()}

# ══════════════════════════════════════════════════════════════════════════
# BACKGROUND SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_eval_episode():
    """
    Loads trained MAPPO checkpoint and runs one eval episode.
    Updates sim_state every step so frontend sees live trust/status changes.
    Imports are inside the function so API starts even if torch isn't available.
    """
    import sys
    sys.path.insert(0, str(PESU_DIR))

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
        if CHECKPOINT_DIR.exists():
            manager.load_checkpoint(str(CHECKPOINT_DIR))

        # preserve any injected byzantine agents from frontend
        injector = ByzantineFaultInjector(total_agents=n_agents)
        for ag in sim_state["agents"]:
            if ag["status"] == "byzantine":
                injector.inject(agent_id=ag["id"], attack_type="random")

        detector = AnomalyDetector(n_agents=n_agents, window_size=20, threshold=3.0)
        checker  = PolicyChecker(n_agents=n_agents, action_space_size=3)
        consensus = ConsensusEngine(n_agents=n_agents, action_space_size=3)
        trust    = TrustManager(n_agents=n_agents)

        # set trust scores from current sim_state
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

            # update shared state
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

            # log consensus decision
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

            time.sleep(0.05)  # ~20 steps/sec so frontend can follow

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
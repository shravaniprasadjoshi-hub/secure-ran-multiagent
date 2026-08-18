"""
dashboard/api.py — FastAPI backend for the Secure RAN Multi-Agent Dashboard
Serves data from both Swetha's outputs and our MARL implementation.
Live-sim logic lives in sim_runner.py (see: run_eval_episode, sim_state).

Run:
    cd PESU_MARL_implementation
    uvicorn dashboard.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard.sim_runner import sim_state, run_eval_episode, BASE_CHECKPOINT_DIR, SECURE_CHECKPOINT_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
SWETHA_DIR = ROOT / "Swetha_proj3_code"
PESU_DIR = ROOT / "PESU_MARL_implementation"

METRICS_JSON = SWETHA_DIR / "outputs" / "metrics" / "all_metrics.json"
COORDINATOR_JSON = SWETHA_DIR / "outputs" / "metrics" / "coordinator_metrics.json"
TELEMETRY_CSV = ROOT / "data" / "ran_multi_agent_telemetry_70k.csv"
TRAINING_LOG = PESU_DIR / "outputs" / "training_log.csv"

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Secure RAN Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    rsrp = df["rsrp_dbm"].dropna()
    rsrp_hist, rsrp_edges = np.histogram(rsrp.sample(min(10000, len(rsrp))), bins=30)

    sinr = df["sinr_db"].dropna()
    sinr_hist, sinr_edges = np.histogram(sinr.sample(min(10000, len(sinr))), bins=30)

    scenario_counts = df["scenario_type"].value_counts().to_dict()

    num_cols = ["rsrp_dbm", "rsrq_db", "sinr_db", "cqi",
                "prb_utilization_pct", "trust_score", "attack_probability"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr().round(2).to_dict()

    lat = df["latency_ms"].dropna().sort_values()
    cdf_x = lat.iloc[::max(1, len(lat)//200)].tolist()
    cdf_y = (np.arange(1, len(lat)+1) / len(lat))[::max(1, len(lat)//200)].tolist()

    sinr_by_scenario = df.groupby("scenario_type")["sinr_db"].median().round(2).to_dict()

    _telem_cache = {
        "total_records": len(df),
        "rsrp": {"counts": rsrp_hist.tolist(), "edges": rsrp_edges.tolist()},
        "sinr": {"counts": sinr_hist.tolist(), "edges": sinr_edges.tolist()},
        "scenario_distribution": scenario_counts,
        "correlation": corr,
        "cdf": {"x": cdf_x, "y": cdf_y},
        "sinr_by_scenario": sinr_by_scenario,
    }
    return _telem_cache

# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/state")
def get_state():
    """Polled every 2s by frontend to update digital twin + security section."""
    return sim_state

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
            sim_state["alerts"] = sim_state["alerts"][:10]
            break
    return {"ok": True, "agent_id": agent_id, "attack_type": attack_type}

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

@app.get("/agent-metrics")
def get_agent_metrics():
    """Returns Swetha's sklearn agent F1/R² scores from all_metrics.json."""
    return load_json(METRICS_JSON)

@app.get("/coordinator-metrics")
def get_coordinator_metrics():
    """Returns Swetha's consensus accept rate and action distribution."""
    metrics = load_json(METRICS_JSON)
    return metrics.get("coordinator", load_json(COORDINATOR_JSON))

@app.get("/training-log")
def get_training_log():
    """Returns our MAPPO training log CSV as JSON for the reward curve chart."""
    if not TRAINING_LOG.exists():
        return {"episodes": [], "rewards": [], "actor_loss": [], "critic_loss": []}

    # train.py's csv.writer writes a real header row (episode, total_reward,
    # actor_loss, critic_loss) - don't pass header=None here, that shifts
    # the header itself into row 0 as string data.
    df = pd.read_csv(TRAINING_LOG)
    return {
        "episodes": df["episode"].tolist(),
        "rewards": df["total_reward"].tolist(),
        "actor_loss": df["actor_loss"].tolist(),
        "critic_loss": df["critic_loss"].tolist(),
    }

@app.get("/telemetry-stats")
def get_telemetry_stats_endpoint():
    """Returns RSRP/SINR distributions, correlation, CDF from the 70k CSV."""
    return get_telemetry_stats()

@app.get("/consensus-log")
def get_consensus_log():
    """Returns last 10 consensus decisions."""
    return {"log": sim_state["consensus_log"]}

@app.post("/start-sim")
def start_simulation(use_secure: bool = True):
    """
    Start background eval episode using the trained checkpoint (see sim_runner.py).
    use_secure=True (default) prefers the MAPPO+security checkpoint, falling back
    to the base MAPPO checkpoint, then to an untrained/random policy.
    """
    if sim_state["running"]:
        return {"ok": False, "msg": "Already running"}
    thread = threading.Thread(target=run_eval_episode, args=(use_secure,), daemon=True)
    thread.start()
    return {"ok": True, "msg": "Simulation started", "use_secure": use_secure}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_checkpoint_exists": BASE_CHECKPOINT_DIR.exists(),
        "secure_checkpoint_exists": SECURE_CHECKPOINT_DIR.exists(),
    }
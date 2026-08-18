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
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from dashboard.sim_runner import sim_state, run_eval_episode

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
SWETHA_DIR = ROOT / "Swetha_proj3_code"
PESU_DIR   = ROOT / "PESU_MARL_implementation"

METRICS_JSON     = SWETHA_DIR / "outputs" / "metrics" / "all_metrics.json"
COORDINATOR_JSON = SWETHA_DIR / "outputs" / "metrics" / "coordinator_metrics.json"
TELEMETRY_CSV    = ROOT / "data" / "ran_multi_agent_telemetry_70k.csv"
TRAINING_LOG   = PESU_DIR / "outputs" / "training_log.csv"
CHECKPOINT_DIR = PESU_DIR / "results" / "checkpoints" / "final"
RAG_CORPUS       = SWETHA_DIR / "data" / "rag_corpus.jsonl"

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Secure RAN Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RAG chatbot — initialise once at startup ───────────────────────────────
_chatbot = None

def get_chatbot():
    global _chatbot
    if _chatbot is not None:
        return _chatbot
    if not RAG_CORPUS.exists():
        return None
    try:
        # Add Swetha's src to path so her imports resolve
        swetha_src = str(SWETHA_DIR)
        if swetha_src not in sys.path:
            sys.path.insert(0, swetha_src)
        from src.chatbot.rag_chatbot import RAGChatbot
        _chatbot = RAGChatbot(RAG_CORPUS)
        print(f"[chatbot] RAG loaded — {len(_chatbot.chunks)} chunks from {RAG_CORPUS.name}")
    except Exception as e:
        print(f"[chatbot] Failed to load RAG: {e} — chatbot will return fallback answers")
        _chatbot = None
    return _chatbot

# Initialise at startup in background so it doesn't delay first request
threading.Thread(target=get_chatbot, daemon=True).start()

# ── Helper: safe JSON load ─────────────────────────────────────────────────
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

    num_cols  = ["rsrp_dbm", "rsrq_db", "sinr_db", "cqi",
                 "prb_utilization_pct", "trust_score", "attack_probability"]
    available = [c for c in num_cols if c in df.columns]
    corr      = df[available].corr().round(2).to_dict()

    lat   = df["latency_ms"].dropna().sort_values()
    step  = max(1, len(lat) // 200)
    cdf_x = lat.iloc[::step].tolist()
    cdf_y = (np.arange(1, len(lat) + 1) / len(lat))[::step].tolist()

    sinr_by_scenario = df.groupby("scenario_type")["sinr_db"].median().round(2).to_dict()

    _telem_cache = {
        "total_records":       len(df),
        "rsrp":                {"counts": rsrp_hist.tolist(), "edges": rsrp_edges.tolist()},
        "sinr":                {"counts": sinr_hist.tolist(), "edges": sinr_edges.tolist()},
        "scenario_distribution": scenario_counts,
        "correlation":         corr,
        "cdf":                 {"x": cdf_x, "y": cdf_y},
        "sinr_by_scenario":    sinr_by_scenario,
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
            agent["trust"]  = 0.41
            sim_state["byzantine_count"] += 1
            sim_state["alerts"].insert(0, {
                "type": "Byzantine",
                "msg":  f"Cell {agent_id} compromised — {attack_type} attack detected",
                "time": "Just now",
            })
            sim_state["alerts"].insert(0, {
                "type": "Consensus",
                "msg":  f"Cell {agent_id} excluded from voting",
                "time": "Just now",
            })
            sim_state["alerts"] = sim_state["alerts"][:10]
            break
    return {"ok": True, "agent_id": agent_id, "attack_type": attack_type}


@app.post("/clear")
def clear_faults():
    """Reset all agents to healthy."""
    for agent in sim_state["agents"]:
        agent["status"] = "healthy"
        agent["trust"]  = 1.0
    sim_state["byzantine_count"] = 0
    sim_state["alerts"].insert(0, {
        "type": "Recovered",
        "msg":  "All faults cleared — agents restored to healthy",
        "time": "Just now",
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
    df = pd.read_csv(TRAINING_LOG)
    return {
        "episodes":    df["episode"].tolist(),
        "rewards":     df["total_reward"].tolist(),
        "actor_loss":  df["actor_loss"].tolist(),
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
def start_simulation():
    """Start background eval episode using the trained checkpoint."""
    if sim_state["running"]:
        return {"ok": False, "msg": "Already running"}
    thread = threading.Thread(target=run_eval_episode, daemon=True)
    thread.start()
    return {"ok": True, "msg": "Simulation started"}


@app.post("/chat")
async def chat(request: Request):
    """
    RAG chatbot endpoint — searches Nokia's rag_corpus.jsonl using TF-IDF.
    Falls back to a helpful error message if the corpus hasn't loaded yet.

    Body: {"question": "your question here"}
    Returns: {"answer": str, "sources": list, "rag_available": bool}
    """
    body = await request.json()
    question = (body.get("question") or "").strip()

    if not question:
        return {"answer": "Please type a question.", "sources": [], "rag_available": False}

    bot = get_chatbot()

    if bot is None:
        return {
            "answer": (
                "The knowledge base is still loading or unavailable. "
                "Try asking about: RRC handover, O-RAN RIC, multi-agent consensus, "
                "Byzantine attacks, RSRP/SINR, or MAPPO training."
            ),
            "sources": [],
            "rag_available": False,
        }

    try:
        result = bot.query(question, top_k=3)
        # Only include sources that actually matched (score threshold)
        good_sources = [s for s in result.get("sources", []) if s.get("score", 0) > 0.05]
        return {
            "answer":        result["answer"],
            "sources":       good_sources,
            "rag_available": True,
        }
    except Exception as e:
        return {
            "answer":        f"Error querying knowledge base: {str(e)}",
            "sources":       [],
            "rag_available": False,
        }


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "checkpoint_exists": CHECKPOINT_DIR.exists(),
        "rag_loaded":        _chatbot is not None,
        "telemetry_exists":  TELEMETRY_CSV.exists(),
    }
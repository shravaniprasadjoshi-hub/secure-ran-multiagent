#!/usr/bin/env python3
"""
End-to-end pipeline: data load → train/val/test → agent models → consensus → digital twin → plots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import load_telemetry
from src.data.preprocessor import encode_features, encode_target, prepare_splits
from src.digital_twin.twin_engine import RANDigitalTwin
from src.orchestration.coordinator import MultiAgentCoordinator, save_coordinator_metrics
from src.training.trainer import train_agent
from src.visualization.plots import (
    plot_agent_metrics,
    plot_architecture_summary,
    plot_confusion_and_roc,
    plot_data_overview,
    plot_digital_twin,
    plot_learning_curves,
    save_metrics_json,
)


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    seed = cfg["project"]["seed"]

    for d in (paths["models_dir"], paths["plots_dir"], paths["metrics_dir"], paths["reports_dir"], paths["twin_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Secure Multi-Agent AI Framework — End-to-End Pipeline", flush=True)
    print("=" * 60, flush=True)

    # 1. Load data
    print("\n[1/6] Loading telemetry data...")
    df = load_telemetry(paths["telemetry_csv"])
    splits = prepare_splits(df, cfg["data_split"]["train_ratio"], cfg["data_split"]["val_ratio"], seed)
    print(f"  Train: {len(splits['train']):,} | Val: {len(splits['val']):,} | Test: {len(splits['test']):,}")

    # 2. Data plots
    print("\n[2/6] Generating data exploration plots...")
    plot_data_overview(df.sample(min(15000, len(df)), random_state=seed), Path(paths["plots_dir"]))

    # 3. Train all agents
    print("\n[3/6] Training agents (train/val/test)...")
    artifacts: dict = {}
    all_results: dict = {}
    for agent_name, agent_cfg in cfg["agents"].items():
        print(f"  Training {agent_name} ({agent_cfg['model']}, {agent_cfg['task']})...")
        artifact, _, result = train_agent(agent_name, agent_cfg, splits)
        artifacts[agent_name] = artifact
        all_results[agent_name] = result
        joblib.dump(artifact, Path(paths["models_dir"]) / f"{agent_name}.joblib")
        t, v, te = result.train_metrics, result.val_metrics, result.test_metrics
        if result.task == "classification":
            print(f"    F1 — train:{t.get('f1',0):.3f} val:{v.get('f1',0):.3f} test:{te.get('f1',0):.3f}")
        else:
            print(f"    R² — train:{t.get('r2',0):.3f} val:{v.get('r2',0):.3f} test:{te.get('r2',0):.3f}")

    # 4. Agent evaluation plots
    print("\n[4/6] Generating model evaluation plots...")
    plot_agent_metrics(all_results, Path(paths["plots_dir"]))
    for name, result in all_results.items():
        plot_confusion_and_roc(result, Path(paths["plots_dir"]))
        art = artifacts[name]
        X, _ = encode_features(splits["train"], art["features"], fit=False, scaler=art["scaler"])
        if art["task"] == "classification":
            y, _ = encode_target(splits["train"][art["target"]], fit=True)
        else:
            y = splits["train"][art["target"]].astype(float).values
        plot_learning_curves(art["model"], X, y, name, Path(paths["plots_dir"]))

    # 5. Multi-agent coordination + digital twin
    print("\n[5/6] Running multi-agent coordinator & digital twin...")
    coordinator = MultiAgentCoordinator(artifacts, cfg["consensus"])
    coord_metrics = coordinator.evaluate_batch(splits["test"], sample_size=500)
    save_coordinator_metrics(coord_metrics, Path(paths["metrics_dir"]) / "coordinator_metrics.json")

    twin_cfg = cfg["digital_twin"]
    twin = RANDigitalTwin(
        num_cells=twin_cfg.get("num_cells", 48),
        num_gnbs=twin_cfg.get("num_gnbs", 12),
        seed=seed,
    )
    twin_history = twin.run_simulation(steps=twin_cfg.get("simulation_steps", 200), attack_at=50)
    twin.save(Path(paths["twin_dir"]))

    plot_digital_twin(twin_history, twin.state, Path(paths["plots_dir"]))
    plot_architecture_summary(coord_metrics, Path(paths["plots_dir"]))

    save_metrics_json(all_results, coord_metrics, Path(paths["metrics_dir"]) / "all_metrics.json")

    # 6. Summary report
    print("\n[6/6] Writing summary report...")
    report = {
        "project": cfg["project"]["name"],
        "data_split": {k: len(v) for k, v in splits.items()},
        "agents_trained": list(artifacts.keys()),
        "consensus_accept_rate": coord_metrics["consensus_accept_rate"],
        "digital_twin_steps": len(twin_history),
    }
    (Path(paths["reports_dir"]) / "pipeline_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Models:  {paths['models_dir']}")
    print(f"  Plots:   {paths['plots_dir']}")
    print(f"  Metrics: {paths['metrics_dir']}")
    print(f"  Twin:    {paths['twin_dir']}")
    print("=" * 60)
    print("\nLaunch dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

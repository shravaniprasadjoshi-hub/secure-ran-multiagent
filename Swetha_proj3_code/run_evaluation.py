#!/usr/bin/env python3
"""Fast evaluation: load saved models, generate plots, metrics, digital twin, slides."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import load_telemetry
from src.data.preprocessor import encode_features, encode_target, prepare_splits
from src.digital_twin.twin_engine import RANDigitalTwin
from src.orchestration.coordinator import MultiAgentCoordinator, save_coordinator_metrics
from src.training.trainer import AgentResult, _cls_metrics, _reg_metrics
from src.visualization.plots import (
    plot_agent_metrics,
    plot_architecture_summary,
    plot_confusion_and_roc,
    plot_data_overview,
    plot_digital_twin,
    save_metrics_json,
)


def evaluate_saved(artifact: dict, splits: dict, agent_name: str) -> AgentResult:
    features = artifact["features"]
    target = artifact["target"]
    task = artifact["task"]
    model = artifact["model"]
    scaler = artifact["scaler"]
    le = artifact.get("label_encoder")

    def _eval(split_name):
        X, _ = encode_features(splits[split_name], features, fit=False, scaler=scaler)
        if task == "classification":
            y, _ = encode_target(splits[split_name][target], le=le, fit=False)
            pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            return y, pred, proba
        y = splits[split_name][target].astype(float).values
        return y, model.predict(X), None

    y_tr, p_tr, _ = _eval("train")
    y_va, p_va, _ = _eval("val")
    y_te, p_te, proba_te = _eval("test")

    if task == "classification":
        return AgentResult(
            agent_name=agent_name, task=task, model_type=artifact["model_type"],
            train_metrics=_cls_metrics(y_tr, p_tr),
            val_metrics=_cls_metrics(y_va, p_va),
            test_metrics=_cls_metrics(y_te, p_te, proba_te),
            predictions={"y_test": y_te, "y_pred": p_te, "y_proba": proba_te,
                         "classes": le.classes_.tolist() if le else []},
        )
    return AgentResult(
        agent_name=agent_name, task=task, model_type=artifact["model_type"],
        train_metrics=_reg_metrics(y_tr, p_tr),
        val_metrics=_reg_metrics(y_va, p_va),
        test_metrics=_reg_metrics(y_te, p_te),
        predictions={"y_test": y_te, "y_pred": p_te},
    )


def main():
    cfg = load_config()
    paths = cfg["paths"]
    for d in (paths["plots_dir"], paths["metrics_dir"], paths["reports_dir"], paths["twin_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    df = load_telemetry(paths["telemetry_csv"])
    splits = prepare_splits(df, cfg["data_split"]["train_ratio"], cfg["data_split"]["val_ratio"], cfg["project"]["seed"])

    print("Data plots...", flush=True)
    plot_data_overview(df.sample(10000, random_state=42), Path(paths["plots_dir"]))

    artifacts = {}
    all_results = {}
    models_dir = Path(paths["models_dir"])
    for agent_name in cfg["agents"]:
        model_path = models_dir / f"{agent_name}.joblib"
        if not model_path.exists():
            print(f"  Missing {agent_name}, run run_pipeline.py first", flush=True)
            continue
        artifact = joblib.load(model_path)
        artifacts[agent_name] = artifact
        result = evaluate_saved(artifact, splits, agent_name)
        all_results[agent_name] = result
        key = "f1" if result.task == "classification" else "r2"
        print(f"  {agent_name}: test {key}={result.test_metrics.get(key, 0):.3f}", flush=True)

    print("Agent plots...", flush=True)
    plot_agent_metrics(all_results, Path(paths["plots_dir"]))
    for name, result in all_results.items():
        plot_confusion_and_roc(result, Path(paths["plots_dir"]))

    print("Coordinator + twin...", flush=True)
    coordinator = MultiAgentCoordinator(artifacts, cfg["consensus"])
    coord_metrics = coordinator.evaluate_batch(splits["test"], sample_size=300)
    save_coordinator_metrics(coord_metrics, Path(paths["metrics_dir"]) / "coordinator_metrics.json")

    twin_cfg = cfg["digital_twin"]
    twin = RANDigitalTwin(
        num_cells=twin_cfg.get("num_cells", 48),
        num_gnbs=twin_cfg.get("num_gnbs", 12),
        seed=cfg["project"]["seed"],
    )
    twin_history = twin.run_simulation(steps=twin_cfg.get("simulation_steps", 200), attack_at=50)
    twin.save(Path(paths["twin_dir"]))

    plot_digital_twin(twin_history, twin.state, Path(paths["plots_dir"]))
    plot_architecture_summary(coord_metrics, Path(paths["plots_dir"]))
    save_metrics_json(all_results, coord_metrics, Path(paths["metrics_dir"]) / "all_metrics.json")

    report = {"agents": list(artifacts.keys()), "consensus_accept_rate": coord_metrics["consensus_accept_rate"]}
    (Path(paths["reports_dir"]) / "pipeline_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Done! Metrics and plots saved.", flush=True)


if __name__ == "__main__":
    main()

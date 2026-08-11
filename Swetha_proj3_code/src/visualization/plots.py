from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import learning_curve

from src.training.trainer import AgentResult

plt.style.use("seaborn-v0_8-darkgrid")
NOKIA_BLUE = "#124191"
NOKIA_TEAL = "#00C9FF"
ACCENT = "#FF6B35"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_data_overview(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Overview — RAN Multi-Agent Telemetry", fontsize=14, fontweight="bold", color=NOKIA_BLUE)

    sc = df["scenario_type"].value_counts()
    axes[0, 0].pie(sc.values, labels=sc.index, autopct="%1.1f%%", colors=sns.color_palette("Blues", len(sc)))
    axes[0, 0].set_title("Scenario Distribution")

    sns.histplot(df["rsrp_dbm"], kde=True, ax=axes[0, 1], color=NOKIA_BLUE)
    axes[0, 1].set_title("RSRP Distribution (dBm)")

    sns.histplot(df["sinr_db"], kde=True, ax=axes[1, 0], color=NOKIA_TEAL)
    axes[1, 0].set_title("SINR Distribution (dB)")

    sns.histplot(df["trust_score"], kde=True, ax=axes[1, 1], color=ACCENT)
    axes[1, 1].set_title("Trust Score Distribution")
    _save(fig, plots_dir / "data" / "01_dataset_overview.png")

    # Correlation heatmap
    num_cols = ["rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "prb_utilization_pct", "cell_load_pct",
                "dl_throughput_mbps", "latency_ms", "trust_score", "attack_probability", "anomaly_score"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", color=NOKIA_BLUE)
    _save(fig, plots_dir / "data" / "02_correlation_heatmap.png")

    # CDFs
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, label in zip(axes, ["latency_ms", "dl_throughput_mbps", "trust_score"], ["Latency (ms)", "DL Throughput (Mbps)", "Trust Score"]):
        sorted_vals = np.sort(df[col].dropna())
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=NOKIA_BLUE, linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF — {label}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cumulative Distribution Functions", fontweight="bold")
    _save(fig, plots_dir / "data" / "03_cdf_plots.png")


def plot_agent_metrics(all_results: dict[str, AgentResult], plots_dir: Path) -> None:
    names = list(all_results.keys())
    cls_agents = [n for n in names if all_results[n].task == "classification"]
    reg_agents = [n for n in names if all_results[n].task == "regression"]

    if cls_agents:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(cls_agents))
        w = 0.2
        for i, split in enumerate(["train", "val", "test"]):
            vals = [getattr(all_results[n], f"{split}_metrics").get("f1", 0) for n in cls_agents]
            ax.bar(x + i * w, vals, w, label=split.capitalize())
        ax.set_xticks(x + w)
        ax.set_xticklabels([n.replace("_", "\n") for n in cls_agents], fontsize=8)
        ax.set_ylabel("F1 Score")
        ax.set_title("Classification Agents — Train / Val / Test F1", fontweight="bold", color=NOKIA_BLUE)
        ax.legend()
        ax.set_ylim(0, 1.05)
        _save(fig, plots_dir / "agents" / "01_classification_f1_comparison.png")

    if reg_agents:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(reg_agents))
        for i, split in enumerate(["train", "val", "test"]):
            vals = [getattr(all_results[n], f"{split}_metrics").get("r2", 0) for n in reg_agents]
            ax.bar(x + i * 0.25, vals, 0.25, label=split.capitalize())
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels([n.replace("_", "\n") for n in reg_agents], fontsize=8)
        ax.set_ylabel("R² Score")
        ax.set_title("Regression Agents — Train / Val / Test R²", fontweight="bold", color=NOKIA_BLUE)
        ax.legend()
        _save(fig, plots_dir / "agents" / "02_regression_r2_comparison.png")

    # Overall heatmap
    metrics_rows = []
    for name, res in all_results.items():
        row = {"agent": name.replace("_agent", "")}
        for split in ["train", "val", "test"]:
            m = getattr(res, f"{split}_metrics")
            key = "f1" if res.task == "classification" else "r2"
            row[f"{split}_{key}"] = m.get(key, 0)
        metrics_rows.append(row)
    mdf = pd.DataFrame(metrics_rows).set_index("agent")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mdf, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Agent Performance Heatmap (Train/Val/Test)", fontweight="bold", color=NOKIA_BLUE)
    _save(fig, plots_dir / "agents" / "03_agent_performance_heatmap.png")


def plot_confusion_and_roc(result: AgentResult, plots_dir: Path) -> None:
    if result.task != "classification":
        return
    y_test = result.predictions["y_test"]
    y_pred = result.predictions["y_pred"]
    classes = result.predictions.get("classes", [])

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{result.agent_name} — Confusion Matrix (Test)", fontweight="bold")
    _save(fig, plots_dir / "agents" / f"{result.agent_name}_confusion_matrix.png")

    proba = result.predictions.get("y_proba")
    if proba is not None and len(classes) == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
        ax.plot(fpr, tpr, color=NOKIA_BLUE, linewidth=2, label=f"ROC (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.set_title(f"{result.agent_name} — ROC Curve (Test)", fontweight="bold")
        _save(fig, plots_dir / "agents" / f"{result.agent_name}_roc_curve.png")


def plot_learning_curves(model, X, y, agent_name: str, plots_dir: Path, max_samples: int = 5000) -> None:
    try:
        if len(X) > max_samples:
            idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        scoring = "f1_weighted" if len(np.unique(y)) < 20 else "r2"
        train_sizes = np.linspace(0.2, 1.0, 5)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=2, n_jobs=-1, train_sizes=train_sizes, scoring=scoring,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color=NOKIA_BLUE, label="Training")
        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color=ACCENT, label="Validation")
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color=NOKIA_BLUE)
        ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color=ACCENT)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Score")
        ax.set_title(f"{agent_name} — Learning Curve", fontweight="bold")
        ax.legend()
        _save(fig, plots_dir / "training" / f"{agent_name}_learning_curve.png")
    except Exception:
        pass


def plot_digital_twin(history: pd.DataFrame, cell_state: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Digital Twin — RAN Simulation", fontsize=14, fontweight="bold", color=NOKIA_BLUE)

    axes[0, 0].plot(history.index, history["mean_sinr"], color=NOKIA_TEAL, linewidth=2)
    axes[0, 0].axvline(50, color=ACCENT, linestyle="--", label="Attack injected")
    axes[0, 0].set_title("Mean SINR over Time")
    axes[0, 0].set_xlabel("Simulation Step")
    axes[0, 0].legend()

    axes[0, 1].plot(history.index, history["mean_throughput"], color=NOKIA_BLUE, linewidth=2)
    axes[0, 1].axvline(50, color=ACCENT, linestyle="--")
    axes[0, 1].set_title("Mean Throughput (Mbps)")

    axes[1, 0].plot(history.index, history["attacks_active"], color="red", linewidth=2)
    axes[1, 0].plot(history.index, history["mitigations"], color="green", linewidth=2, label="Mitigations")
    axes[1, 0].set_title("Active Attacks vs Mitigations")
    axes[1, 0].legend()

    axes[1, 1].plot(history.index, history["mean_trust"], color=ACCENT, linewidth=2)
    axes[1, 1].set_title("Mean Trust Score")
    _save(fig, plots_dir / "digital_twin" / "01_twin_time_series.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        cell_state["rsrp_dbm"], cell_state["sinr_db"],
        c=cell_state["prb_util_pct"], s=cell_state["active_ues"] / 2,
        cmap="viridis", alpha=0.8, edgecolors="white", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="PRB Util %")
    ax.set_xlabel("RSRP (dBm)")
    ax.set_ylabel("SINR (dB)")
    ax.set_title("Digital Twin — Cell State Map", fontweight="bold", color=NOKIA_BLUE)
    _save(fig, plots_dir / "digital_twin" / "02_twin_cell_map.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = cell_state.pivot_table(index="gnb_id", values=["prb_util_pct", "throughput_mbps", "trust_score"], aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    ax.set_title("gNB-Level KPI Heatmap", fontweight="bold")
    _save(fig, plots_dir / "digital_twin" / "03_gnb_kpi_heatmap.png")


def plot_architecture_summary(coordinator_metrics: dict, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overall Architecture Evaluation", fontweight="bold", color=NOKIA_BLUE)

    rate = coordinator_metrics.get("consensus_accept_rate", 0)
    axes[0].bar(["Consensus Accept Rate"], [rate], color=NOKIA_TEAL, width=0.4)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Multi-Agent Consensus")
    axes[0].axhline(0.7, color=ACCENT, linestyle="--", label="70% threshold")
    axes[0].legend()

    actions = coordinator_metrics.get("action_distribution", {})
    if actions:
        axes[1].pie(actions.values(), labels=actions.keys(), autopct="%1.1f%%")
        axes[1].set_title("RRC Action Distribution")
    _save(fig, plots_dir / "architecture" / "01_overall_evaluation.png")


def save_metrics_json(all_results: dict[str, AgentResult], coordinator_metrics: dict, path: Path) -> None:
    data = {
        "agents": {
            name: {
                "task": r.task,
                "model_type": r.model_type,
                "train": r.train_metrics,
                "val": r.val_metrics,
                "test": r.test_metrics,
            }
            for name, r in all_results.items()
        },
        "coordinator": coordinator_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

"""
training/stats_plots.py
Plot generation for MARL training logs
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#124191", "#FF6B35", "#00A86B", "#8B5CF6"]  # extend if >4 logs


def label_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def plot_reward_curve(dfs_labels, window, outdir):
    """Raw reward (faint) + rolling mean (bold) + EMA, one subplot per run overlayed on same axes."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for (df, label), color in zip(dfs_labels, COLORS):
        roll_mean = df["total_reward"].rolling(window).mean()
        ema = df["total_reward"].ewm(alpha=0.05, adjust=False).mean()
        ax.plot(df["episode"], df["total_reward"], color=color, alpha=0.05, linewidth=0.5)
        ax.plot(df["episode"], roll_mean, color=color, linewidth=2, label=f"{label} (rolling mean, w={window})")
        ax.plot(df["episode"], ema, color=color, linewidth=1, linestyle="--", label=f"{label} (EMA)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward — Raw / Rolling Mean / EMA")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, outdir, "01_reward_curve.png")


def plot_rolling_std(dfs_labels, window, outdir):
    """Rolling std of reward — stability/convergence indicator."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for (df, label), color in zip(dfs_labels, COLORS):
        roll_std = df["total_reward"].rolling(window).std()
        ax.plot(df["episode"], roll_std, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling Std (window={window})")
    ax.set_title("Reward Rolling Std — lower & flatter = more converged")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, outdir, "02_reward_rolling_std.png")


def plot_losses(dfs_labels, window, outdir):
    """Actor + critic loss, smoothed, one figure with 2 stacked subplots."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for (df, label), color in zip(dfs_labels, COLORS):
        for ax, col in zip(axes, ["actor_loss", "critic_loss"]):
            if col not in df.columns:
                continue
            smoothed = df[col].rolling(window).mean()
            ax.plot(df["episode"], df[col], color=color, alpha=0.05, linewidth=0.5)
            ax.plot(df["episode"], smoothed, color=color, linewidth=1.8, label=label)
    axes[0].set_title("Actor Loss (raw + smoothed)")
    axes[1].set_title("Critic Loss (raw + smoothed)")
    axes[1].set_xlabel("Episode")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, outdir, "03_actor_critic_loss.png")


def plot_reward_distribution(dfs_labels, tail, outdir):
    """Histogram + boxplot of last-N-episode reward distribution, per run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    box_data, box_labels = [], []
    for (df, label), color in zip(dfs_labels, COLORS):
        seg = df["total_reward"].tail(tail)
        ax1.hist(seg, bins=30, alpha=0.5, color=color, label=label, density=True)
        box_data.append(seg.values)
        box_labels.append(label)
    ax1.set_title(f"Reward Distribution (last {tail} episodes)")
    ax1.set_xlabel("Total Reward")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.boxplot(box_data, labels=box_labels, showmeans=True)
    ax2.set_title(f"Reward Spread (last {tail} episodes)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(alpha=0.3)
    _save(fig, outdir, "04_reward_distribution.png")


def plot_correlation_heatmap(dfs_labels, outdir):
    """Actor/critic/reward correlation heatmap, one panel per run."""
    n = len(dfs_labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (df, label) in zip(axes, dfs_labels):
        cols = [c for c in ["total_reward", "actor_loss", "critic_loss"] if c in df.columns]
        corr = df[cols].corr()
        im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(label, fontsize=10)
    fig.suptitle("Reward / Actor Loss / Critic Loss Correlation", fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.7, label="correlation")
    _save(fig, outdir, "05_correlation_heatmap.png", tight=False)


def _save(fig, outdir, filename, tight=True):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--tail", type=int, default=100)
    ap.add_argument("--outdir", default="training/outputs/plots")
    args = ap.parse_args()

    dfs_labels = [(pd.read_csv(p), label_from_path(p)) for p in args.logs]

    plot_reward_curve(dfs_labels, args.window, args.outdir)
    plot_rolling_std(dfs_labels, args.window, args.outdir)
    plot_losses(dfs_labels, args.window, args.outdir)
    plot_reward_distribution(dfs_labels, args.tail, args.outdir)
    plot_correlation_heatmap(dfs_labels, args.outdir)


if __name__ == "__main__":
    main()
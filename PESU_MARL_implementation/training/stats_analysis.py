"""
training/stats_analysis.py
Statistical analysis for MARL training logs (reward, actor_loss, critic_loss)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def trend_analysis(df, col="total_reward"):
    """Linear regression of col vs episode. Returns slope, R^2, p-value."""
    x = df["episode"].values
    y = df[col].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {"slope": slope, "r_squared": r**2, "p_value": p, "intercept": intercept}


def rolling_stats(df, col="total_reward", window=50):
    roll_mean = df[col].rolling(window).mean()
    roll_std = df[col].rolling(window).std()
    cv = (roll_std / roll_mean.abs()).replace([np.inf, -np.inf], np.nan)
    return roll_mean, roll_std, cv


def find_convergence_point(df, col="total_reward", window=50, slope_tol=0.5, std_ratio_tol=0.15):
    """
    Slides a window; convergence = first episode index where:
      - rolling mean's local slope (over the window) is near 0 (< slope_tol per-episode)
      - rolling std / |rolling mean| stays below std_ratio_tol
    and stays that way for the rest of the run (checked via suffix condition).
    Returns episode number, or None if never converges.
    """
    y = df[col].values
    n = len(y)
    roll_mean = df[col].rolling(window).mean().values
    roll_std = df[col].rolling(window).std().values

    for i in range(window, n - window):
        seg = roll_mean[i:i + window]
        if np.any(np.isnan(seg)):
            continue
        local_slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        ratio = roll_std[i] / (abs(roll_mean[i]) + 1e-8)
        if abs(local_slope) < slope_tol and ratio < std_ratio_tol:
            # check it holds for rest of training (not a fluke plateau)
            rest_std_ratio = np.nanmean(roll_std[i:] / (np.abs(roll_mean[i:]) + 1e-8))
            if rest_std_ratio < std_ratio_tol * 1.5:
                return int(df["episode"].iloc[i])
    return None


def ema(series, alpha=0.05):
    return series.ewm(alpha=alpha, adjust=False).mean()


def simple_cusum_changepoints(series, threshold_k=1.0, drift=0.5):
    """
    Lightweight CUSUM change-point detector (no external deps).
    Flags indices where cumulative deviation from running mean exceeds threshold.
    threshold_k: multiples of series std used as trigger.
    Returns list of episode indices flagged as change points.
    """
    x = series.values
    mean = np.nanmean(x)
    std = np.nanstd(x) + 1e-8
    threshold = threshold_k * std

    pos, neg = 0.0, 0.0
    changepoints = []
    for i, val in enumerate(x):
        dev = val - mean
        pos = max(0, pos + dev - drift)
        neg = min(0, neg + dev + drift)
        if pos > threshold or neg < -threshold:
            changepoints.append(i)
            pos, neg = 0.0, 0.0  # reset after flagging
    return changepoints


def tail_distribution(df, col="total_reward", tail=100):
    seg = df[col].tail(tail)
    return {
        "mean": seg.mean(),
        "std": seg.std(),
        "median": seg.median(),
        "iqr": seg.quantile(0.75) - seg.quantile(0.25),
        "skew": stats.skew(seg),
        "kurtosis": stats.kurtosis(seg),
        "min": seg.min(),
        "max": seg.max(),
    }


def correlation_block(df):
    cols = [c for c in ["total_reward", "actor_loss", "critic_loss"] if c in df.columns]
    return df[cols].corr().round(3).to_dict()


def analyze_one(path, window, tail):
    df = pd.read_csv(path)
    report = {"file": path, "n_episodes": len(df)}

    for col in ["total_reward", "actor_loss", "critic_loss"]:
        if col not in df.columns:
            continue
        report[f"{col}_trend"] = trend_analysis(df, col)
        report[f"{col}_tail_dist"] = tail_distribution(df, col, tail)

    report["convergence_episode"] = find_convergence_point(df, "total_reward", window)
    report["correlations"] = correlation_block(df)

    cps = simple_cusum_changepoints(df["total_reward"])
    report["reward_changepoints_episodes"] = [int(df["episode"].iloc[i]) for i in cps][:20]  # cap output

    # loss stability: variance in second half vs first half
    half = len(df) // 2
    for col in ["actor_loss", "critic_loss"]:
        if col not in df.columns:
            continue
        first_std = df[col].iloc[:half].std()
        second_std = df[col].iloc[half:].std()
        report[f"{col}_stability"] = {
            "first_half_std": first_std,
            "second_half_std": second_std,
            "ratio_second_over_first": second_std / (first_std + 1e-8),
        }

    return report, df


def compare_runs(reports):
    """Simple side-by-side reward trend + convergence comparison across multiple logs."""
    summary = []
    for r in reports:
        summary.append({
            "file": os.path.basename(r["file"]),
            "n_episodes": r["n_episodes"],
            "reward_slope": r.get("total_reward_trend", {}).get("slope"),
            "reward_r2": r.get("total_reward_trend", {}).get("r_squared"),
            "converged_at_ep": r.get("convergence_episode"),
            "tail_mean_reward": r.get("total_reward_tail_dist", {}).get("mean"),
            "tail_std_reward": r.get("total_reward_tail_dist", {}).get("std"),
        })
    return pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="one or more train_log CSV paths")
    ap.add_argument("--window", type=int, default=50, help="rolling window size")
    ap.add_argument("--tail", type=int, default=100, help="last N episodes for distribution stats")
    ap.add_argument("--out", default="training/outputs/stats_report.json")
    args = ap.parse_args()

    all_reports = []
    for path in args.logs:
        report, df = analyze_one(path, args.window, args.tail)
        all_reports.append(report)

        print(f"\n=== {path} ===")
        print(f"episodes: {report['n_episodes']}")
        rt = report.get("total_reward_trend", {})
        print(f"reward trend: slope={rt.get('slope'):.4f}  R^2={rt.get('r_squared'):.4f}  p={rt.get('p_value'):.2e}")
        print(f"convergence episode: {report['convergence_episode']}")
        td = report.get("total_reward_tail_dist", {})
        print(f"tail(last {args.tail}) reward: mean={td.get('mean'):.2f} std={td.get('std'):.2f} "
              f"skew={td.get('skew'):.2f}")
        print(f"reward changepoints (episodes): {report['reward_changepoints_episodes']}")
        for col in ["actor_loss", "critic_loss"]:
            st = report.get(f"{col}_stability")
            if st:
                print(f"{col} stability: first_half_std={st['first_half_std']:.4f} "
                      f"second_half_std={st['second_half_std']:.4f} "
                      f"ratio={st['ratio_second_over_first']:.2f}")
        print(f"correlations: {report['correlations']}")

    if len(all_reports) > 1:
        print("\n=== Cross-run comparison ===")
        cmp_df = compare_runs(all_reports)
        print(cmp_df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_reports, f, indent=2, default=float)
    print(f"\nfull report saved -> {args.out}")


if __name__ == "__main__":
    main()
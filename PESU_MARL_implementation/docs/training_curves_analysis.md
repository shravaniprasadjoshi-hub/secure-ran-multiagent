# MARL Training Curves - Statistical Analysis
**Secure Multi-Agent AI Framework for RAN Control Loops**
Comparing: `train_log.csv` (Exp 2 - MAPPO only) vs `train_log_secure.csv` (Exp 3 - MAPPO + full security stack)
Window = 50 episodes | Tail = last 100 episodes | N = 1000 episodes each

Refer to training/outputs/
    - plots/ 
    - stats_report.json
---

## Figure 1 - Reward Curve (Raw / Rolling Mean / EMA)

**What it shows:** Per-episode total reward (faint line), 50-episode rolling mean (bold), and exponential moving average (dashed), overlaid for both runs.

**Shape description:**
- **Clean (train_log):** Rises sharply from ~-2600 to ~1500 within the first ~100 episodes, then flattens into a stable plateau (~1600–1800) for the remaining 900 episodes. Classic fast-rise-then-plateau PPO signature.
- **Secure (train_log_secure):** Starts near -1000, dips further to ~-2600 by episode ~80, then oscillates in a wide band (-2500 to 0) with several local peaks/troughs (visible bumps around ep 350-400, 500, 780-800) and a slow net upward drift, only crossing into positive territory briefly near ep 950-1000.

**Key data points:**
| Metric | Clean | Secure |
|---|---|---|
| Reward slope (per episode) | +0.4475 | +1.7962 |
| R² of linear trend | 0.190 | 0.177 |
| Tail (last 100) mean | 1789.21 | -231.37 |
| Tail (last 100) std | 137.37 | 930.80 |
| Tail median | 1787.0 | -66.0 |

**Interpretation - is it proper?**
The clean run shows textbook convergence: fast climb, low final variance, mean ≈ median (symmetric, stable outcome distribution). The secure run's steeper raw slope (1.80 vs 0.45/episode) is misleading in isolation - it reflects climbing out of a much deeper hole, not superior learning. The real story is in tail variance: secure's std (930.8) is ~6.8x higher than clean's (137.4) at the same point in training.

**Good signal:** The secure run's trend is monotonically positive across the full 1000 episodes (statistically significant, p≈3.2e-44) - it is learning, not diverging or stuck. This supports the "trending correctly but not converged" framing already used in `experiment_3_findings.md`.

**Where we can improve:** 1000 episodes is insufficient for the secure variant to reach a plateau comparable to the clean run's ~100-episode convergence. Extending training, tuning learning rate/entropy coefficient, or revisiting reward scaling for the security penalties are the natural next steps before claiming convergence.

---

## Figure 2 - Reward Rolling Std (window=50)

**What it shows:** Rolling standard deviation of reward - a direct volatility/convergence indicator, independent of the mean level.

**Shape description:**
- **Clean:** Drops sharply from ~830 to ~150 by episode ~150, then stays flat in a narrow 100–170 band for the rest of training.
- **Secure:** Never drops below ~750 at any point. Fluctuates between ~750 and ~1450, with the *worst* point (peak ~1460) occurring late in training around episode 700-800 - not early, as might be expected from a simple "still warming up" story.

**Interpretation - is it proper?**
Clean: this is the plot that produces the `convergence_episode: 147` value in the JSON report. A flat, low rolling std sustained for the remainder of training is a strong, visually confirmable convergence signal.
Secure: `convergence_episode: null` is directly visible here - the curve never settles into a flat low band. The mid-training hump (ep ~700-800) is actually a regression, not a plateau being approached.

**Good signal:** For the clean run, this is the strongest single piece of evidence for "MAPPO converges cleanly" - recommend citing this plot directly alongside the numeric convergence episode in the ablation table.

**Where we can improve:** The ep 700-800 spike in the secure run deserves targeted investigation - check whether it aligns with a specific point in the Byzantine attack schedule (agents 3/6, gradual vs random attack types), a scenario-type transition, or an anomaly-detector/quarantine interaction. This is a concrete, specific claim that would strengthen the findings document beyond "it's noisy."

---

## Figure 3 - Actor Loss / Critic Loss (raw + smoothed)

**What it shows:** Two stacked panels - actor loss (top) and critic loss (bottom), raw (faint) and 50-episode smoothed (bold), for both runs.

**Shape description:**
- **Actor loss, clean:** Effectively flat at ~0 the entire run (true values ~-0.003, invisible at this plot's scale) - expected behavior for a PPO clipped-surrogate objective once training stabilizes.
- **Actor loss, secure:** Flat near 0 until ~episode 700, then rises steeply and continuously, reaching 300-600 by episode 1000 - an exponential-looking blow-up in the last ~30% of training.
- **Critic loss, both runs:** Clean stays in a stable 350-420 band throughout. Secure starts lower (~230), climbs to match clean's level by ~episode 400, tracks closely until ~episode 800, then also rises to ~470.

**Key data points:**
| Metric | Clean | Secure |
|---|---|---|
| Actor loss trend slope | 1.45e-06 (≈0, p=8.5e-22 but trivial magnitude) | 0.2024 (p=2.9e-91, real & strong) |
| Actor loss stability (2nd half std / 1st half std) | 0.222 (**improving**) | 2802.27 (**catastrophic**) |
| Critic loss stability (2nd half std / 1st half std) | 0.723 (mildly improving) | 1.044 (~flat) |

**Interpretation - is it proper?**
Clean: both losses bounded and stable - healthy PPO training, no red flags.
Secure: this is the clearest quantitative evidence of non-convergence in the entire report. Actor loss variance in the second half of training is ~2800x that of the first half - not gradual noise, but a genuine structural blow-up starting around episode 700 (statistically confirmed: slope=0.202, R²=0.337, p<1e-90 - a real trend, not random spikes).

**Good signal:** Critic loss stability ratio (1.04) shows the value function itself is not the source of instability - it's tracking returns about as consistently as the clean run's critic does. This narrows the problem to the actor/policy update step specifically, which is a more precise and useful diagnostic than "the security stack is unstable" in general.

**Where we can improve:** Candidate causes worth testing, in order of likely impact:
1. PPO clip range (`CLIP_EPS`) may be too permissive when advantage estimates are large under the security reward's wider variance
2. `MAX_GRAD_NORM=0.5` caps gradient *norm*, not the resulting loss magnitude - worth confirming clipping is actually engaging during the blow-up window
3. Advantage normalization may not be fully compensating for the security reward scale relative to JRHT's original reward magnitudes (documented in `docs/ablation_comparison_table.md`)
4. Consider a lower `LR_ACTOR` specifically for the secure training variant

---

## Figure 4 - Reward Distribution, Last 100 Episodes (Histogram + Boxplot)

**What it shows:** Distribution shape of reward in the final 100 episodes - the closest proxy for "what does the converged/near-converged policy actually deliver."

**Shape description:**
- **Clean:** Tight, roughly unimodal histogram centered at ~1789, narrow boxplot (no visible outliers), whiskers spanning 1466–2138.
- **Secure:** Broad, flat, roughly bimodal/spread-out histogram spanning -2700 to +1600. Boxplot shows median=-66, IQR=1246.5 (~6x wider than clean), with one distinct low outlier flagged at -2711.

**Key data points:**
| Metric | Clean | Secure |
|---|---|---|
| Tail mean | 1789.21 | -231.37 |
| Tail median | 1787.0 | -66.0 |
| Tail IQR | 202.5 | 1246.5 |
| Tail skew | 0.090 (near-symmetric) | -0.579 (left-tailed) |
| Tail kurtosis | -0.567 (slightly flat) | -0.218 (slightly flat) |
| Tail min / max | 1466 / 2138 | -2711 / 1576 |

**Interpretation - is it proper?**
Clean: a converged policy should produce a tight, near-symmetric reward distribution close to its performance ceiling - confirmed here (skew ≈ 0.09).
Secure: not consistent with a converged policy. Wide spread + negative skew (-0.579) indicates a subset of episodes with severe negative outcomes dragging the distribution's mean down further than its "typical" (median) behavior.

**Good signal:** The gap between mean (-231.37) and median (-66.0) for the secure run means a small number of very bad episodes are disproportionately affecting the mean. The median-based read is somewhat more optimistic than mean alone suggests - recommend reporting both statistics in the findings document rather than mean only, to avoid overstating how bad "typical" performance currently is.

**Where we can improve:** The single outlier at -2711 warrants a manual case-study look - identifying which episode number and which scenario/attack type was active would let the findings doc include a concrete failure-mode example (e.g., "Episode X: agent 6 gradual-drift attack triggered a cascading RLF chain") rather than only aggregate statistics.

---

## Figure 5 - Correlation Heatmap (Reward / Actor Loss / Critic Loss)

**What it shows:** Pearson correlation matrix across the three logged metrics, per run, over the full 1000-episode history.

**Key data points:**
| Pair | Clean | Secure |
|---|---|---|
| reward ↔ actor_loss | 0.80 | 0.40 |
| reward ↔ critic_loss | 0.56 | 0.39 |
| actor_loss ↔ critic_loss | 0.35 | 0.39 |

**Interpretation:**
Clean: The strong reward↔actor_loss correlation (0.80) mostly reflects both metrics moving together over the course of training (low reward + different loss magnitude early, high reward + settled loss later) - this is a co-movement artifact of the training trajectory, not a causal claim that lower actor loss directly produces higher reward (actor_loss here is PPO's clipped surrogate objective, not an accuracy/error metric).
Secure: All three pairwise correlations compress to a narrow, uniform range (~0.39–0.40). No single relationship dominates - consistent with a system where reward swings are driven more by external security-stack events (Byzantine attacks, quarantine triggers, anomaly flags) than by the ordinary actor/critic feedback loop you'd see in vanilla PPO.

**Good/improve signal:** This plot is best used as *supporting* evidence for "the secure run's dynamics are structurally different and noisier," rather than a standalone headline result - it doesn't provide an actionable lever on its own.

---

## JSON Report - Additional Findings (Not Fully Visible in Plots)

**Convergence episode:**
- Clean: **147**. This is the single cleanest, most citable quantitative result from this analysis - recommend using it directly in the ablation comparison table.
- Secure: **null** (never converges within 1000 episodes by the rolling-mean-slope + std-ratio criterion used). This is the correct, honest number to report - consistent with the project's stated principle of documenting non-convergence rather than presenting a misleadingly "converged" narrative.

**Change-point detection - known limitation, recommend excluding or re-running:**
The `reward_changepoints_episodes` field for the clean run returns `[1, 2, 3, ..., 20]` - this is an artifact of the CUSUM detector tripping on the initial -2600→1500 climb, not a meaningful set of regime changes. The secure run's list (`2, 3, 5, 7, 10, 12...`) has the same issue - it's flagging the noisy early-training region rather than genuine later-training shifts. **Recommendation:** either drop this metric from the final report, or re-run change-point detection with episodes < ~100–150 excluded (post-warmup) to surface genuine mid/late-training regime shifts if any exist. Happy to patch `stats_analysis.py` for this if useful before finalizing the report.

**Critic loss trend, clean run:** slope ≈ 0.00034, p = 0.956 - statistically indistinguishable from a flat trend. This confirms the critic is not drifting over time, a positive stability signal worth explicitly noting.

**Actor loss trend, secure run:** slope = 0.202, R² = 0.337, p = 2.9e-91 - this is a real, strong, statistically significant upward trend, not random noise. This is the numeric backbone behind the visual blow-up seen in Figure 3, and the strongest evidence in the whole report that the security stack introduces a genuine optimization instability rather than incidental noise.

---

## Summary Table - Clean vs Secure at a Glance

| Metric | Clean (Exp 2) | Secure (Exp 3) | Verdict |
|---|---|---|---|
| Converged? | Yes, ep 147 | No (null) | Clean converges; secure does not within 1000 eps |
| Tail reward mean | 1789.21 | -231.37 | Large gap - expected given security constraints |
| Tail reward std | 137.37 | 930.80 | Secure ~6.8x more volatile |
| Actor loss stability (2nd/1st half) | 0.22 (improving) | 2802x (blowing up) | Root cause of secure's non-convergence |
| Critic loss stability (2nd/1st half) | 0.72 | 1.04 | Critic is not the problem in either run |
| Reward trend direction | Positive, flattening | Positive, still climbing | Both learning; secure just needs more time/tuning |

**Overall conclusion for reporting:** Exp 2 (MAPPO only) is a clean, converged baseline suitable for publication as-is. Exp 3 (MAPPO + security stack) shows a genuine, statistically-confirmed learning trend but has not converged within the tested 1000 episodes, with the actor-loss instability after episode ~700 as the specific, targeted next engineering problem - not a general "security makes training worse" statement, but a precise, fixable optimization issue localized to the policy update step.
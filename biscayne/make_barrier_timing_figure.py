"""
Generates Table 4 / Figure 3 of the paper: the barrier commitment timing
diagnostic for DAPP, RDM-Conservative, and the Unified Framework (500 trials).

This complements run_revision_experiments.py's aggregated
results/<out>/barrier_timing.json (deployment rate, mean commit step, %
trials failing before barrier) with the raw per-trial commit/failure step
data needed to reproduce Figure 3's histogram and scatter plot.

Usage:
  python make_barrier_timing_figure.py [--n-trials 500] [--seed 321] [--output-dir results/dapp_final]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import biscayne_main as bm
import run_revision_experiments as rre

BARRIER_STRATEGIES = ["DAPP", "RDM_Conservative", "Unified_Framework"]
NEVER = -1


def collect_raw_barrier_data(trans_matrix, recharge, sigma_pre, sigma_post,
                              n_trials=500, T=24, seed=321):
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])
    regime_prior = np.array([0.41, 0.43, 0.16])
    raw = {k: {"commit": [], "fail": []} for k in BARRIER_STRATEGIES}

    for trial in range(n_trials):
        seq = bm._sample_regime_seq(rng, trans_matrix, T, regime_prior)
        for k in BARRIER_STRATEGIES:
            r = rre.STRATEGY_FUNCS[k](seq, recharge, sigma_pre, sigma_post,
                                       pi_r0.copy(), pi_m0.copy(), rng)
            raw[k]["commit"].append(r["commit_step"] if r["commit_step"] is not None else NEVER)
            raw[k]["fail"].append(r["first_failure_step"] if r["first_failure_step"] is not None else NEVER)
        if (trial + 1) % 100 == 0:
            print(f"  {trial+1}/{n_trials} …", flush=True)
    return raw


def plot_barrier_timing(raw, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    ax = axes[0]
    labels = ["DAPP (obs-trigger)", "RDM (h-trigger)", "Unified (real options)"]
    colors = ["#EF5350", "#7E57C2", "#1565C0"]
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
    bin_labels = ["0", "1", "2", "3", "4", "5", "6-7", "8-9", "10-11", "12-15", "16-23", "Never"]
    width = 0.25
    x_base = np.arange(len(bin_labels))
    for i, k in enumerate(BARRIER_STRATEGIES):
        commit = np.array(raw[k]["commit"])
        counts = [np.mean((commit >= bin_edges[b]) & (commit < bin_edges[b + 1]))
                  for b in range(len(bin_edges) - 1)]
        counts.append(np.mean(commit < 0))
        ax.bar(x_base + (i - 1) * width, counts, width=width,
               label=labels[i], color=colors[i], alpha=0.85)
    ax.set_xticks(x_base); ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_xlabel("Decision step barrier is built (binned; T=24 steps)")
    ax.set_ylabel("Proportion of trials")
    ax.set_title("Barrier Commitment Timing\n(500 MC trials, T=24 decision periods)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    commit = np.array(raw["DAPP"]["commit"], dtype=float)
    fail = np.array(raw["DAPP"]["fail"], dtype=float)
    PLOT_NEVER = 25
    commit_p = np.where(commit < 0, PLOT_NEVER, commit)
    fail_p = np.where(fail < 0, PLOT_NEVER, fail)
    rng = np.random.default_rng(0)
    jx = commit_p + rng.uniform(-0.2, 0.2, size=len(commit_p))
    jy = fail_p + rng.uniform(-0.2, 0.2, size=len(fail_p))
    ax.scatter(jx, jy, s=12, alpha=0.3, color="#1565C0")
    ax.plot([0, PLOT_NEVER], [0, PLOT_NEVER], "--", color="gray", lw=1,
            label="barrier = failure (simultaneous)")
    ax.set_xlim(-0.5, PLOT_NEVER + 0.5); ax.set_ylim(-0.5, PLOT_NEVER + 0.5)
    xt = list(range(0, 25, 5)) + [PLOT_NEVER]
    ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in range(0, 25, 5)] + ["Never"])
    ax.set_yticks(xt); ax.set_yticklabels([str(v) for v in range(0, 25, 5)] + ["Never"])
    ax.set_xlabel("Step barrier built")
    ax.set_ylabel("Step of first failure")
    ax.set_title("DAPP: Barrier Commitment vs First Failure\n(points above the diagonal = late trigger)")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output-dir", type=str, default="results/dapp_final")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Calibrating parameters …")
    trans_matrix, recharge, sigma_pre, sigma_post = rre.calibrate()

    print(f"Collecting raw barrier-timing data ({args.n_trials} trials) …")
    raw = collect_raw_barrier_data(trans_matrix, recharge, sigma_pre, sigma_post,
                                    n_trials=args.n_trials, seed=args.seed)

    with open(out / "barrier_timing_raw.json", "w") as f:
        json.dump(raw, f)
    print(f"  Saved → {out / 'barrier_timing_raw.json'}")

    plot_barrier_timing(raw, out / "fig_barrier_timing.png")
    print("Done.")


if __name__ == "__main__":
    main()

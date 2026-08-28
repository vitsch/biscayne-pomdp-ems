"""
CPU-time and solution-stability supplement for EJOR-D-26-02073 Reviewer #2, point 4.

Reuses the exact strategy implementations and calibration pipeline from
biscayne_main.py (imported, not reimplemented) so results are consistent
with the published Table 3 numbers. Adds two things Table 3 doesn't report:

  (A) Per-strategy CPU time at the published configuration (n=1000, T=24, seed=42).
  (B) Cross-seed solution stability at n=1000, T=24, across 5 independent seeds.

Usage:
  python run_cpu_stability_experiment.py
"""

import json
import time
from pathlib import Path

import numpy as np

import biscayne_main as bm

OUT_DIR = Path(__file__).parent / "results" / "cpu_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["BDT_Baseline", "Robust_Only", "Myopic_POMDP",
        "DAPP", "RDM_Conservative", "Unified_Framework"]

STRATEGY_FUNCS = {
    "BDT_Baseline":      bm.run_strategy_bdt,
    "Robust_Only":       bm.run_strategy_robust_only,
    "Myopic_POMDP":      bm.run_strategy_myopic_pomdp,
    "DAPP":              bm.run_strategy_dapp,
    "RDM_Conservative":  bm.run_strategy_rdm_conservative,
    "Unified_Framework": bm.run_strategy_unified,
}


def experiment_B_timed(trans_matrix, recharge, sigma_pre, sigma_post,
                        n_trials=1000, T=24, seed=42):
    """Same trial loop and RNG draw order as biscayne_main.experiment_B,
    with per-strategy CPU time (time.process_time) and wall time
    (time.perf_counter) accumulated around each run_strategy_* call.
    Draw order/RNG consumption is unchanged, so failure/cost results
    are bit-identical to biscayne_main.experiment_B for the same seed."""
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])
    regime_prior = np.array([0.41, 0.43, 0.16])

    fail_rates = {k: [] for k in KEYS}
    costs      = {k: [] for k in KEYS}
    cpu_time   = {k: 0.0 for k in KEYS}
    wall_time  = {k: 0.0 for k in KEYS}

    for trial in range(n_trials):
        seq = bm._sample_regime_seq(rng, trans_matrix, T, regime_prior)

        for k in KEYS:
            fn = STRATEGY_FUNCS[k]
            t_cpu0, t_wall0 = time.process_time(), time.perf_counter()
            r = fn(seq, recharge, sigma_pre, sigma_post,
                   pi_r0.copy(), pi_m0.copy(), rng)
            cpu_time[k]  += time.process_time() - t_cpu0
            wall_time[k] += time.perf_counter() - t_wall0
            fail_rates[k].append(r["failure_rate"])
            costs[k].append(-r["cum_reward"])

        if (trial + 1) % 200 == 0:
            print(f"  MC trial {trial+1}/{n_trials} …", flush=True)

    results = {}
    for k in KEYS:
        arr = np.array(fail_rates[k])
        c   = np.array(costs[k])
        results[k] = {
            "failure_rate_mean": float(arr.mean()),
            "failure_rate_std":  float(arr.std()),
            "failure_rate_ci95": (float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5))),
            "cost_mean": float(c.mean()),
            "cost_std":  float(c.std()),
            "cpu_time_total_s":  cpu_time[k],
            "cpu_time_per_trial_ms": 1000.0 * cpu_time[k] / n_trials,
            "wall_time_total_s": wall_time[k],
        }
    return results


def main():
    print("=" * 65)
    print("  CPU time + solution stability supplement (EJOR-D-26-02073, R2#4)")
    print("=" * 65)

    print("\n[1/3] Calibrating parameters from cached climate data …")
    calib = bm.calibrate_regimes(save_dir=str(bm.DATA_DIR))
    trans_matrix = calib["transition_matrix"]
    recharge     = calib["recharge_rates"]
    noise = bm.calibrate_obs_noise()
    sigma_pre, sigma_post = noise["sigma_pre"], noise["sigma_post"]

    # ── (A) CPU time at the published configuration (n=1000, seed=42) ─────────
    print("\n[2/3] CPU-time benchmark at published configuration (n=1000, seed=42) …")
    timed_results = experiment_B_timed(
        trans_matrix, recharge, sigma_pre, sigma_post,
        n_trials=1000, T=24, seed=42)

    print("\n  Sanity check vs. published Table 3 (should match to rounding):")
    for k in KEYS:
        r = timed_results[k]
        print(f"    {k:22s} fail={r['failure_rate_mean']*100:5.1f}%  "
              f"cost={r['cost_mean']:7.1f}  "
              f"CPU/trial={r['cpu_time_per_trial_ms']:.3f} ms  "
              f"CPU total={r['cpu_time_total_s']:.2f} s")

    # ── (B) Cross-seed solution stability (n=1000 each, 5 seeds) ───────────────
    seeds = [42, 43, 44, 45, 46]
    print(f"\n[3/3] Cross-seed stability: n=1000 trials x {len(seeds)} seeds {seeds} …")
    per_seed = {}
    for s in seeds:
        print(f"\n  --- seed {s} ---")
        per_seed[s] = bm.experiment_B(
            trans_matrix, recharge, sigma_pre, sigma_post,
            n_trials=1000, T=24, seed=s)
        for k in KEYS:
            r = per_seed[s][k]
            print(f"    {k:22s} fail={r['failure_rate_mean']*100:5.1f}%  cost={r['cost_mean']:7.1f}")

    stability = {}
    for k in KEYS:
        fails = np.array([per_seed[s][k]["failure_rate_mean"] for s in seeds])
        costs_ = np.array([per_seed[s][k]["cost_mean"] for s in seeds])
        stability[k] = {
            "failure_rate_across_seeds": fails.tolist(),
            "failure_rate_mean_of_means": float(fails.mean()),
            "failure_rate_std_across_seeds": float(fails.std()),
            "cost_across_seeds": costs_.tolist(),
            "cost_mean_of_means": float(costs_.mean()),
            "cost_std_across_seeds": float(costs_.std()),
            "cost_cv_across_seeds_pct": float(100 * costs_.std() / costs_.mean()) if costs_.mean() else 0.0,
        }

    out = {
        "cpu_time_benchmark": {k: {kk: vv for kk, vv in v.items()} for k, v in timed_results.items()},
        "stability_seeds": seeds,
        "stability_per_seed": {str(s): per_seed[s] for s in seeds},
        "stability_summary": stability,
    }
    out_path = OUT_DIR / "cpu_stability_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")

    print("\n" + "=" * 65)
    print("  Summary table: CPU time (published config) + cross-seed stability")
    print("=" * 65)
    print(f"  {'Strategy':22s} {'CPU/trial(ms)':>13s} {'FailRate mean±SD (5 seeds)':>28s} {'Cost mean±SD (5 seeds)':>24s}")
    for k in KEYS:
        cpu_ms = timed_results[k]["cpu_time_per_trial_ms"]
        fmean = stability[k]["failure_rate_mean_of_means"] * 100
        fstd  = stability[k]["failure_rate_std_across_seeds"] * 100
        cmean = stability[k]["cost_mean_of_means"]
        cstd  = stability[k]["cost_std_across_seeds"]
        print(f"  {k:22s} {cpu_ms:13.3f} {fmean:9.2f}% ± {fstd:5.2f}pp{'':10s} {cmean:9.2f} ± {cstd:5.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

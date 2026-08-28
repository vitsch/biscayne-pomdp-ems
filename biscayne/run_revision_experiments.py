"""
Full experiment regeneration for the EJOR-D-26-02073 revision, after fixing
voi_estimate() from a myopic one-step estimator to the non-myopic (Eq. 5)
rollout-based estimator, and fixing experiment_C's stale lambda grid.

Regenerates every number currently claimed in the manuscript that depends on
run_strategy_unified, plus two diagnostics (barrier-timing, 500-trajectory OV
calibration at b_0) whose original generating code was not present in this
repository (likely run in a separate notebook) and is reconstructed here from
the manuscript's stated methodology (500 trials / 500 trajectories) using the
existing, unmodified strategy functions.

Produces:
  results/revision_v2/experiment_summary.json   -- six-strategy comparison (n=1000)
  results/revision_v2/sensitivity_grid.npy       -- 4x4 grid (lambda now incl. 1.5)
  results/revision_v2/barrier_timing.json        -- Table 4 diagnostic (500 trials)
  results/revision_v2/ov_calibration.json        -- Remark 2 / Sec 3.4 (500 trajectories)
  results/revision_v2/uniform_prior_check.json   -- Sec 4.4.5 non-informative prior (n=500)
"""

import json
from pathlib import Path

import numpy as np

import biscayne_main as bm

OUT_DIR = Path(__file__).parent / "results" / "revision_v2"
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

BARRIER_STRATEGIES = ["DAPP", "RDM_Conservative", "Unified_Framework"]


def calibrate():
    calib = bm.calibrate_regimes(save_dir=str(bm.DATA_DIR))
    trans_matrix = calib["transition_matrix"]
    recharge     = calib["recharge_rates"]
    noise = bm.calibrate_obs_noise()
    return trans_matrix, recharge, noise["sigma_pre"], noise["sigma_post"]


def run_six_strategy_comparison(trans_matrix, recharge, sigma_pre, sigma_post,
                                 n_trials=1000, T=24, seed=42):
    """Same structure as biscayne_main.experiment_B, but also records
    commit_step / first_failure_step for the three barrier-deploying strategies
    (needed for the barrier-timing diagnostic at this same n=1000 sample if
    wanted, though Table 4 in the paper uses a separate 500-trial run — see
    run_barrier_timing_diagnostic below, which follows the paper's stated n)."""
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])
    regime_prior = np.array([0.41, 0.43, 0.16])

    fail_rates = {k: [] for k in KEYS}
    costs      = {k: [] for k in KEYS}

    for trial in range(n_trials):
        seq = bm._sample_regime_seq(rng, trans_matrix, T, regime_prior)
        for k in KEYS:
            r = STRATEGY_FUNCS[k](seq, recharge, sigma_pre, sigma_post,
                                   pi_r0.copy(), pi_m0.copy(), rng)
            fail_rates[k].append(r["failure_rate"])
            costs[k].append(-r["cum_reward"])
        if (trial + 1) % 200 == 0:
            print(f"  MC trial {trial+1}/{n_trials} …", flush=True)

    results = {}
    for k in KEYS:
        arr = np.array(fail_rates[k]); c = np.array(costs[k])
        n = len(arr)
        p = float(arr.mean())
        # Clopper-Pearson 95% upper bound for the failure-rate proportion
        from scipy import stats
        successes = int(round(p * n))
        cp_upper = float(stats.beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1.0
        results[k] = {
            "failure_rate_mean": p,
            "failure_rate_std": float(arr.std()),
            "failure_rate_ci95": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
            "failure_rate_clopper_pearson_upper95": cp_upper,
            "cost_mean": float(c.mean()),
            "cost_std": float(c.std()),
        }
    return results


def run_barrier_timing_diagnostic(trans_matrix, recharge, sigma_pre, sigma_post,
                                   n_trials=500, T=24, seed=321):
    """Table 4: barrier deployment rate, mean commit step, % late-commitment
    (commit_step > first_failure_step) for DAPP, RDM-Conservative, Unified."""
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])
    regime_prior = np.array([0.41, 0.43, 0.16])

    commit_steps = {k: [] for k in BARRIER_STRATEGIES}
    deployed     = {k: 0 for k in BARRIER_STRATEGIES}
    late         = {k: 0 for k in BARRIER_STRATEGIES}

    for trial in range(n_trials):
        seq = bm._sample_regime_seq(rng, trans_matrix, T, regime_prior)
        for k in BARRIER_STRATEGIES:
            r = STRATEGY_FUNCS[k](seq, recharge, sigma_pre, sigma_post,
                                   pi_r0.copy(), pi_m0.copy(), rng)
            if r["barrier_built"]:
                deployed[k] += 1
                commit_steps[k].append(r["commit_step"])
                if r["first_failure_step"] is not None and r["commit_step"] > r["first_failure_step"]:
                    late[k] += 1

    results = {}
    for k in BARRIER_STRATEGIES:
        cs = np.array(commit_steps[k]) if commit_steps[k] else np.array([])
        results[k] = {
            "deployment_rate": deployed[k] / n_trials,
            "mean_commit_step": float(cs.mean()) if len(cs) else None,
            "std_commit_step": float(cs.std()) if len(cs) else None,
            "pct_late_commitment_of_deployed": (late[k] / deployed[k]) if deployed[k] else None,
            "n_deployed": deployed[k],
            "n_trials": n_trials,
        }
    return results


def run_ov_calibration(recharge, seed=555, n_trajectories=500, T_horizon=6):
    """Remark 2 / Section 3.4: MC estimate of V_notB(b0), V_B(b0), OV(b0) at
    the initial belief, via literal 500-trajectory sampling from the
    calibrated DRY-dominant regime prior (matching the paper's stated
    methodology), rather than option_value_raw's exact expectation."""
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.41, 0.43, 0.16])  # calibrated stationary regime prior
    h0 = bm.H_INIT

    v_commit = -bm.COST_BARRIER + sum(
        bm.GAMMA**t * bm.reward_fn("MEDIUM", False) for t in range(T_horizon))

    v_delay_samples = []
    for _ in range(n_trajectories):
        ri = rng.choice(len(bm.REGIMES), p=pi_r0)
        regime = bm.REGIMES[ri]
        h_sim = h0; cum = 0.0
        for step in range(T_horizon):
            h_sim = np.clip(h_sim + recharge[regime] - bm.PUMP_RATE["LOW"], 0.05, 1.0)
            cum += bm.GAMMA**step * bm.reward_fn("LOW", h_sim <= bm.H_CRIT)
        v_delay_samples.append(cum)
    v_delay_samples = np.array(v_delay_samples)
    v_delay = float(v_delay_samples.mean())
    ov = v_delay - v_commit

    return {
        "n_trajectories": n_trajectories,
        "V_notB_b0": v_delay,
        "V_notB_b0_std": float(v_delay_samples.std()),
        "V_B_b0": v_commit,
        "OV_b0": ov,
        "proactive_commitment_optimal": bool(ov < 0),
    }


def run_uniform_prior_check(trans_matrix, recharge, sigma_pre, sigma_post,
                             n_trials=500, T=24, seed=777):
    """Section 4.4.5: non-informative uniform initial belief and uniform
    regime-sampling prior, Unified Framework only."""
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([1/3, 1/3, 1/3])
    pi_m0 = np.array([1/3, 1/3, 1/3])
    regime_prior = np.array([1/3, 1/3, 1/3])

    fails = []; costs = []
    for trial in range(n_trials):
        seq = bm._sample_regime_seq(rng, trans_matrix, T, regime_prior)
        r = bm.run_strategy_unified(seq, recharge, sigma_pre, sigma_post,
                                     pi_r0.copy(), pi_m0.copy(), rng)
        fails.append(r["failure_rate"]); costs.append(-r["cum_reward"])
    fails = np.array(fails); costs = np.array(costs)
    n = len(fails)
    p = float(fails.mean())
    from scipy import stats
    successes = int(round(p * n))
    cp_upper = float(stats.beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1.0
    return {
        "n_trials": n_trials,
        "failure_rate_mean": p,
        "failure_rate_clopper_pearson_upper95": cp_upper,
        "cost_mean": float(costs.mean()),
        "cost_std": float(costs.std()),
    }


def main():
    print("=" * 70)
    print("  Revision experiment regeneration (non-myopic VoI fix)")
    print("=" * 70)

    trans_matrix, recharge, sigma_pre, sigma_post = calibrate()

    print("\n[1/5] Six-strategy comparison (n=1000, seed=42) …")
    mc_results = run_six_strategy_comparison(trans_matrix, recharge, sigma_pre, sigma_post)
    for k in KEYS:
        r = mc_results[k]
        print(f"    {k:22s} fail={r['failure_rate_mean']*100:5.2f}%  "
              f"(CP95 upper={r['failure_rate_clopper_pearson_upper95']*100:.2f}%)  "
              f"cost={r['cost_mean']:7.2f}")
    with open(OUT_DIR / "experiment_summary.json", "w") as f:
        json.dump(mc_results, f, indent=2)

    print("\n[2/5] Barrier-timing diagnostic (500 trials) …")
    barrier = run_barrier_timing_diagnostic(trans_matrix, recharge, sigma_pre, sigma_post)
    for k in BARRIER_STRATEGIES:
        r = barrier[k]
        print(f"    {k:22s} deploy={r['deployment_rate']*100:.1f}%  "
              f"commit_step={r['mean_commit_step']}  late={r['pct_late_commitment_of_deployed']}")
    with open(OUT_DIR / "barrier_timing.json", "w") as f:
        json.dump(barrier, f, indent=2)

    print("\n[3/5] OV calibration at b0 (500 trajectories) …")
    ov = run_ov_calibration(recharge)
    print(f"    V_notB(b0)={ov['V_notB_b0']:.2f}  V_B(b0)={ov['V_B_b0']:.2f}  "
          f"OV(b0)={ov['OV_b0']:.2f}  proactive_optimal={ov['proactive_commitment_optimal']}")
    with open(OUT_DIR / "ov_calibration.json", "w") as f:
        json.dump(ov, f, indent=2)

    print("\n[4/5] Sensitivity grid (4x4, n=200/cell, lambda incl. 1.5) …")
    grid, delta_vals, lambda_vals = bm.experiment_C(trans_matrix, recharge, sigma_pre, sigma_post)
    np.save(OUT_DIR / "sensitivity_grid.npy", grid)
    with open(OUT_DIR / "sensitivity_grid.json", "w") as f:
        json.dump({"delta_vals": delta_vals, "lambda_vals": lambda_vals,
                    "grid_failure_rate": grid.tolist()}, f, indent=2)

    print("\n[5/5] Non-informative uniform prior check (n=500) …")
    uniform = run_uniform_prior_check(trans_matrix, recharge, sigma_pre, sigma_post)
    print(f"    fail={uniform['failure_rate_mean']*100:.2f}%  "
          f"(CP95 upper={uniform['failure_rate_clopper_pearson_upper95']*100:.2f}%)  "
          f"cost={uniform['cost_mean']:.2f}")
    with open(OUT_DIR / "uniform_prior_check.json", "w") as f:
        json.dump(uniform, f, indent=2)

    print(f"\nAll outputs written to {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

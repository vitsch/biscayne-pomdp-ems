"""
Biscayne Aquifer POMDP Experiment
==================================
Integrates real climate-forcing data (ERA5 precipitation, NOAA sea level,
NOAA ENSO ONI) with the unified robust-adaptive POMDP framework from
els_is_claude_colab.py.

Key design decision: pumping actions feed back into the physical trajectory.
  LOW    → pump_rate=0.01 (conservative extraction, preserves freshwater head)
  MEDIUM → pump_rate=0.03 (baseline extraction)
  HIGH   → pump_rate=0.06 (intensive extraction, depletes freshwater head)
This allows different strategies to produce genuinely different failure rates.

Barrier protection: once a barrier is constructed it caps salinity at 0 (s=0)
regardless of h, representing physical saltwater exclusion infrastructure.

Experiment structure:
  A – Belief update on 24-year climate-forcing proxy sequence
  B – Monte Carlo comparison (1,000 trials × 6 strategies)
       1. BDT Baseline        – fixed MEDIUM, no adaptation
       2. Robust Satisficing  – satisficing only, no VoI/OV
       3. Myopic POMDP        – Bayesian beliefs + immediate reward, no OV
       4. DAPP                – observation-threshold trigger (Haasnoot et al. 2013)
       5. RDM-Conservative    – stress-tested static policy (Lempert et al. 2006)
       6. Unified Framework   – Algorithm 1 (satisficing + VoI + real options)
  C – Sensitivity analysis (δ × λ grid, Unified Framework only)

Usage:
  python biscayne_main.py [--mc-trials N] [--seed S] [--output-dir DIR]
  python biscayne_main.py --mc-trials 200 --skip-sensitivity  # quick run
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

# ── Project paths ─────────────────────────────────────────────────────────────
BISCAYNE_DIR = Path(__file__).parent
DATA_DIR     = BISCAYNE_DIR / "data"
RESULTS_DIR  = BISCAYNE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BISCAYNE_DIR / "modules"))
from regime_calibrator import calibrate_regimes, REGIME_NAMES
from obs_calibrator    import calibrate_obs_noise
from salinity_proxy    import build_proxy

# ── Physical parameters ────────────────────────────────────────────────────────
H_CRIT      = 0.40    # freshwater volume below which intrusion becomes critical
H_INIT      = 0.72    # initial normalised freshwater head
GAMMA       = 0.95    # discount factor
U_MIN       = -30.0   # satisficing minimum cumulative utility
DELTA       = 0.10    # max allowed failure probability
LAMBDA_VOI  = 2.0     # VoI weight in objective
MU_OV       = 1.5     # option-value weight in objective

REGIMES = ["DRY", "STATIONARY", "WET"]
MODELS  = ["M1: Sharp-interface", "M2: Dispersive", "M3: Tipping-point"]
ACTIONS = ["LOW", "MEDIUM", "HIGH"]

# Pumping rates: fraction of normalised volume extracted per step
PUMP_RATE   = {"LOW": 0.01, "MEDIUM": 0.03, "HIGH": 0.06}

COST_PUMP    = {"LOW": 1.0,  "MEDIUM": 2.5,  "HIGH": 4.0}
COST_MONITOR = 1.5
COST_BARRIER = 20.0
COST_FAILURE = 50.0

# Water-demand target and unmet-demand (shortfall) cost. Community demand is
# represented by the MEDIUM extraction rate; pumping below it incurs a
# shortfall penalty proportional to the deficit, so that reduced pumping is a
# genuine risk-reducing *trade-off* (cheaper extraction, higher depletion
# safety, but at the cost of unmet demand) rather than a free-lunch dominant
# choice. DEMAND_SHORTFALL_WEIGHT is set so that, absent any failure risk,
# MEDIUM (cost 2.5, no shortfall) is cheaper than LOW (cost 1.0 + shortfall
# 100*(0.03-0.01)=2.0 -> 3.0 total): a modest 0.5-unit premium for the
# precautionary reduction, small enough to be overturned by depletion risk
# once the failure cost (50.0) becomes material.
DEMAND_TARGET = PUMP_RATE["MEDIUM"]
DEMAND_SHORTFALL_WEIGHT = 100.0

# ── Physical model functions ───────────────────────────────────────────────────

def salinity_mean_model(h: float, model: str, tipped: bool = False) -> float:
    if model == "M1: Sharp-interface":
        return 0.90 if h <= H_CRIT else 0.10
    elif model == "M2: Dispersive":
        return 1.0 / (1.0 + np.exp(18.0 * (h - H_CRIT)))
    else:   # M3: Tipping-point
        if tipped or h <= H_CRIT:
            return 0.95
        return 0.05 + 0.60 * np.exp(-10.0 * (h - H_CRIT))


def salinity_mean_regime(h: float, regime: str, recharge: dict,
                          pump_rate: float = 0.03) -> float:
    h_next = np.clip(h + recharge[regime] - pump_rate, 0.05, 1.0)
    return salinity_mean_model(h_next, "M2: Dispersive")


def log_lik_regime(obs, h, regime, sigma, recharge, pump_rate=0.03):
    mu = salinity_mean_regime(h, regime, recharge, pump_rate)
    return scipy_norm.logpdf(obs, loc=mu, scale=sigma)


def log_lik_model(obs, h, model, sigma, tipped=False):
    mu = salinity_mean_model(h, model, tipped)
    return scipy_norm.logpdf(obs, loc=mu, scale=sigma)


# ── Bayesian belief update ─────────────────────────────────────────────────────

def update_beliefs_step(pi_r, pi_m, h, obs, sigma, recharge,
                         pump_rate=0.03, tipped=False):
    """Single-step Bayesian update. Returns updated (pi_r, pi_m, tipped)."""
    log_r = np.array([log_lik_regime(obs, h, reg, sigma, recharge, pump_rate)
                      for reg in REGIMES])
    log_r -= log_r.max()
    pi_r = pi_r * np.exp(log_r)
    pi_r /= pi_r.sum()

    log_m = np.array([log_lik_model(obs, h, mod, sigma, tipped)
                      for mod in MODELS])
    log_m -= log_m.max()
    pi_m = pi_m * np.exp(log_m)
    pi_m /= pi_m.sum()

    return pi_r, pi_m


# ── Reward function ────────────────────────────────────────────────────────────

def shortfall_cost(action):
    deficit = max(0.0, DEMAND_TARGET - PUMP_RATE[action])
    return DEMAND_SHORTFALL_WEIGHT * deficit


def reward_fn(action, failed, monitor=False, barrier=False):
    r = -COST_PUMP[action] - shortfall_cost(action)
    if monitor:   r -= COST_MONITOR
    if barrier:   r -= COST_BARRIER
    if failed:    r -= COST_FAILURE
    return r


def rollout_pump_value(pi_r, act, h, recharge, barrier_built, T_horizon=6):
    """Expected T_horizon-step discounted value of sustaining pumping level
    `act` for the full horizon, marginalised over the regime belief pi_r.
    A one-step lookahead cannot detect gradual depletion risk (a single
    step's extraction is small relative to the head buffer, so LOW and
    MEDIUM are both "safe" one step ahead even when repeated MEDIUM pumping
    would deplete the head over several steps); evaluating a sustained
    T_horizon-step policy, matching the convention already used by
    satisficing_check/_rollout_value/option_value_raw, makes pump-level
    selection genuinely non-myopic and gives it a real, risk-sensitive
    trade-off against the extraction/shortfall cost. Once the barrier is
    built, failure is physically impossible regardless of h, so the
    depletion-risk term is correctly suppressed."""
    pump = PUMP_RATE[act]
    v = 0.0
    for ri, rp in enumerate(pi_r):
        if rp == 0.0:
            continue
        h_sim = h
        cum = 0.0
        for step in range(T_horizon):
            h_sim = np.clip(h_sim + recharge[REGIMES[ri]] - pump, 0.05, 1.0)
            failed_sim = (not barrier_built) and (h_sim <= H_CRIT)
            cum += GAMMA**step * reward_fn(act, failed_sim)
        v += rp * cum
    return v


# ── Satisficing check ──────────────────────────────────────────────────────────

def satisficing_check(pi_r, pi_m, h, recharge, rng, delta=DELTA,
                       n_samples=200, T_horizon=6):
    totals = []
    for _ in range(n_samples):
        regime = REGIMES[rng.choice(len(REGIMES), p=pi_r)]
        h_sim = h; cum = 0.0
        for step in range(T_horizon):
            h_sim = np.clip(h_sim + recharge[regime] - PUMP_RATE["LOW"], 0.05, 1.0)
            failed = h_sim <= H_CRIT
            cum += GAMMA**step * reward_fn("LOW", failed)
        totals.append(cum)
    return float(np.mean(np.array(totals) >= U_MIN)) >= (1.0 - delta)


# ── VoI estimate ───────────────────────────────────────────────────────────────

def _rollout_value(pi_r, pi_m, h, recharge, T_horizon):
    """T_horizon-step discounted cumulative reward under the fixed LOW-pumping
    reference policy (matching option_value_raw's convention), expected over the
    (regime, model) belief. This is \\hat V(b) in Eq. 5, evaluated by exact
    enumeration over the 3x3 regime-model grid rather than by sampling, since
    the grid is small and enumeration is exact and cheap."""
    v = 0.0
    for ri, rp in enumerate(pi_r):
        for mi, mp in enumerate(pi_m):
            if rp * mp == 0.0:
                continue
            h_sim = h
            cum = 0.0
            for step in range(T_horizon):
                h_sim = np.clip(h_sim + recharge[REGIMES[ri]] - PUMP_RATE["LOW"], 0.05, 1.0)
                failed_sim = h_sim <= H_CRIT
                cum += GAMMA**step * reward_fn("LOW", failed_sim)
            v += rp * mp * cum
    return v


def voi_estimate(pi_r, pi_m, h, obs, sigma, recharge, rng, n_samples=20, T_horizon=6):
    """Non-myopic value of information (Eq. 5): VoI(a,b_t) = E_o[V(tau(b_t,a,o))]
    - V(b_t), approximated by sampling n_samples future observations and
    evaluating a T_horizon-step rollout value \\hat V (via _rollout_value) under
    the updated belief at each sample, rather than a one-step reward. This
    matches the non-myopic VoI defined in Eq. 5 and used in Proposition 2,
    replacing an earlier one-step (myopic) reward-difference estimator."""
    v_current = _rollout_value(pi_r, pi_m, h, recharge, T_horizon)

    v_post = 0.0
    for _ in range(n_samples):
        obs_s = float(rng.normal(obs, sigma))
        log_r = np.array([log_lik_regime(obs_s, h, reg, sigma, recharge)
                          for reg in REGIMES])
        log_r -= log_r.max()
        pr2 = pi_r * np.exp(log_r); pr2 /= pr2.sum()
        log_m = np.array([log_lik_model(obs_s, h, mod, sigma) for mod in MODELS])
        log_m -= log_m.max()
        pm2 = pi_m * np.exp(log_m); pm2 /= pm2.sum()
        v_post += _rollout_value(pr2, pm2, h, recharge, T_horizon)
    v_post /= n_samples
    return max(0.0, v_post - v_current)


# ── Option value ───────────────────────────────────────────────────────────────

def option_value_raw(pi_r, h, recharge, T_horizon=6):
    """Returns raw OV (can be negative → barrier commit is optimal). The
    committed-barrier branch assumes MEDIUM pumping post-commitment --
    cost-minimising once the barrier eliminates failure risk, matching the
    actual action-selection logic in run_strategy_unified (rollout_pump_value
    with barrier_built=True correctly prefers MEDIUM once risk is nullified).
    The delay branch assumes LOW pumping -- the cautious reference policy
    used consistently elsewhere (satisficing_check, VoI rollout) for risk
    assessment while still unprotected by a barrier."""
    v_commit = -COST_BARRIER + sum(
        GAMMA**t * reward_fn("MEDIUM", False) for t in range(T_horizon))
    v_delay = 0.0
    for ri, rp in enumerate(pi_r):
        h_sim = h; cum = 0.0
        for step in range(T_horizon):
            h_sim = np.clip(h_sim + recharge[REGIMES[ri]] - PUMP_RATE["LOW"], 0.05, 1.0)
            cum += GAMMA**step * reward_fn("LOW", h_sim <= H_CRIT)
        v_delay += rp * cum
    return v_delay - v_commit   # positive → delay preferred; negative → commit now


# ── Strategy runners ───────────────────────────────────────────────────────────
# Each strategy receives the SAME external regime sequence (forcing) but generates
# its own h_traj based on its pumping decisions. This lets strategies diverge
# in failure rate through the feedback: action → pump_rate → h → intrusion risk.

def _make_obs(rng, h, regime_name, recharge, sigma, tipped, pump_rate):
    """Generate one noisy salinity observation given true h and regime."""
    h_true = np.clip(h + recharge[regime_name] - pump_rate, 0.05, 1.0)
    mu = salinity_mean_model(h_true, "M2: Dispersive", tipped)
    return float(rng.normal(mu, sigma))


def run_strategy_bdt(regime_seq, recharge, sigma_pre, sigma_post,
                      pi_regime_0, pi_model_0, rng, T_monitor=12):
    """
    BDT Baseline: fixed MEDIUM pumping throughout, no adaptive adjustments.
    Uses maximum-likelihood regime estimate; ignores model uncertainty.
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False

    for t, regime in enumerate(regime_seq):
        action = "MEDIUM"
        pump   = PUMP_RATE[action]
        sigma  = sigma_post if t >= T_monitor else sigma_pre
        h_new  = np.clip(h + recharge[regime] - pump, 0.05, 1.0)
        failed = h_new <= H_CRIT
        if failed: tipped = True
        cum_r += GAMMA**t * reward_fn(action, failed)
        if failed: n_fail += 1
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T}


def run_strategy_robust_only(regime_seq, recharge, sigma_pre, sigma_post,
                               pi_regime_0, pi_model_0, rng, T_monitor=12):
    """
    Robust satisficing only: reduces to LOW pumping when satisficing fails.
    No VoI, no real options.
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False
    pi_r = pi_regime_0.copy(); pi_m = pi_model_0.copy()

    for t, regime in enumerate(regime_seq):
        sigma = sigma_post if t >= T_monitor else sigma_pre
        obs   = _make_obs(rng, h, regime, recharge, sigma, tipped, PUMP_RATE["MEDIUM"])

        sat = satisficing_check(pi_r, pi_m, h, recharge, rng)
        action = "LOW" if not sat else "MEDIUM"
        pump   = PUMP_RATE[action]

        h_new  = np.clip(h + recharge[regime] - pump, 0.05, 1.0)
        failed = h_new <= H_CRIT
        if failed: tipped = True
        cum_r += GAMMA**t * reward_fn(action, failed)
        if failed: n_fail += 1

        pi_r, pi_m = update_beliefs_step(pi_r, pi_m, h, obs, sigma, recharge, pump, tipped)
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T}


def run_strategy_myopic_pomdp(regime_seq, recharge, sigma_pre, sigma_post,
                               pi_regime_0, pi_model_0, rng, T_monitor=12):
    """
    Myopic POMDP: Bayesian beliefs + immediate expected-reward maximisation.
    Chooses action that maximises E[reward | belief] at each step.
    No satisficing, no VoI, no real options.
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False
    pi_r = pi_regime_0.copy(); pi_m = pi_model_0.copy()

    for t, regime in enumerate(regime_seq):
        sigma = sigma_post if t >= T_monitor else sigma_pre

        # Choose action that maximises immediate expected reward
        best_score = -np.inf; best_action = "MEDIUM"
        for act in ACTIONS:
            pump = PUMP_RATE[act]
            exp_r = sum(
                pi_r[ri] * reward_fn(act,
                    np.clip(h + recharge[REGIMES[ri]] - pump, 0.05, 1.0) <= H_CRIT)
                for ri in range(len(REGIMES))
            )
            if exp_r > best_score:
                best_score = exp_r; best_action = act
        pump = PUMP_RATE[best_action]

        obs   = _make_obs(rng, h, regime, recharge, sigma, tipped, pump)
        h_new = np.clip(h + recharge[regime] - pump, 0.05, 1.0)
        failed = h_new <= H_CRIT
        if failed: tipped = True
        cum_r += GAMMA**t * reward_fn(best_action, failed)
        if failed: n_fail += 1

        pi_r, pi_m = update_beliefs_step(pi_r, pi_m, h, obs, sigma, recharge, pump, tipped)
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T}


def run_strategy_unified(regime_seq, recharge, sigma_pre, sigma_post,
                          pi_regime_0, pi_model_0, rng, T_monitor=12,
                          lam=LAMBDA_VOI, mu=MU_OV, delta=DELTA):
    """
    Unified framework (Algorithm 1): satisficing + non-myopic VoI + real options.
    Once barrier is committed, future intrusions are physically prevented.
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False
    pi_r = pi_regime_0.copy(); pi_m = pi_model_0.copy()
    barrier_built = False
    commit_step = None; first_failure_step = None

    for t, regime in enumerate(regime_seq):
        sigma = sigma_post if t >= T_monitor else sigma_pre
        obs   = _make_obs(rng, h, regime, recharge, sigma, tipped, PUMP_RATE["MEDIUM"])

        # Algorithm 1 steps
        sat = satisficing_check(pi_r, pi_m, h, recharge, rng, delta=delta)
        voi = voi_estimate(pi_r, pi_m, h, obs, sigma, recharge, rng) if sat else 0.0
        ov_raw = option_value_raw(pi_r, h, recharge) if not barrier_built else 0.0

        # Action selection: lam*voi is conditioned on the monitoring action,
        # since monitoring is the only action that acquires additional
        # information in this environment (VoI is a property of *acquiring*
        # information, not of pump level). The mu*OV bonus is intentionally
        # NOT added here: it is already correctly applied, conditioned on the
        # barrier action specifically, by the separate build_barrier trigger
        # below (matching Eq. 7's a=a_B indicator) -- adding it again here
        # unconditionally would apply the real-options bonus to every action,
        # not just the barrier action.
        best_score = -np.inf; best_action = "LOW"; use_monitor = False
        for act in ACTIONS:
            if not sat and act == "HIGH":
                continue
            for mon in [False, True]:
                score = rollout_pump_value(pi_r, act, h, recharge, barrier_built)
                if mon:
                    score += -COST_MONITOR + lam * voi
                if score > best_score:
                    best_score = score; best_action = act; use_monitor = mon

        # Commit barrier if option value of delay is negative (Proposition 3:
        # OV(b_t) <= 0 is the mu-independent optimality condition; mu scales
        # the value contribution in the Bellman decomposition, not the sign
        # of this condition)
        build_barrier = (not barrier_built) and (ov_raw < 0.0)
        if build_barrier:
            barrier_built = True
            commit_step = t

        pump   = PUMP_RATE[best_action]
        h_new  = np.clip(h + recharge[regime] - pump, 0.05, 1.0)

        # Barrier prevents physical intrusion regardless of h
        failed = (h_new <= H_CRIT) and (not barrier_built)
        if failed and not tipped: tipped = True
        if failed and first_failure_step is None: first_failure_step = t

        cum_r += GAMMA**t * reward_fn(best_action, failed, use_monitor, build_barrier)
        if failed: n_fail += 1

        pi_r, pi_m = update_beliefs_step(pi_r, pi_m, h, obs, sigma, recharge, pump, tipped)
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T,
            "barrier_built": barrier_built,
            "commit_step": commit_step,
            "first_failure_step": first_failure_step}


# ── DAPP benchmark strategy ───────────────────────────────────────────────────

def run_strategy_dapp(regime_seq, recharge, sigma_pre, sigma_post,
                      pi_regime_0, pi_model_0, rng, T_monitor=12,
                      signal_low=0.55, signal_high=0.75, consec_needed=2):
    """
    Dynamic Adaptive Policy Pathways (DAPP) — simplified threshold-trigger policy.

    Implements the signpost-and-pathway logic of Haasnoot et al. (2013, EMS):
      Signpost 1: obs > signal_low for consec_needed consecutive steps
                  → trigger pathway: reduce pumping to LOW
      Signpost 2: obs > signal_high for consec_needed consecutive steps
                  → trigger pathway: construct barrier

    Crucially, DAPP uses raw observations (not Bayesian beliefs) to trigger
    pathway switches. This is the industry-standard adaptive management approach
    and the primary benchmark for the unified POMDP framework in EMS.

    Default thresholds (signal_low=0.55, signal_high=0.75) chosen so that
    signal_low corresponds to near-midpoint salinity and signal_high to a
    clear saline intrusion event. Both sit within the proxy obs range [0.01, 0.95].
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False
    barrier_built = False
    action = "MEDIUM"
    consec_low = 0; consec_high = 0
    commit_step = None; first_failure_step = None

    for t, regime in enumerate(regime_seq):
        sigma = sigma_post if t >= T_monitor else sigma_pre
        obs   = _make_obs(rng, h, regime, recharge, sigma, tipped, PUMP_RATE[action])

        # Signpost logic: count consecutive exceedances (reset on drop below)
        consec_high = consec_high + 1 if obs > signal_high else 0
        consec_low  = consec_low  + 1 if obs > signal_low  else 0

        # Pathway triggers (one-way commitments — DAPP pathways are irreversible)
        if not barrier_built and consec_high >= consec_needed:
            barrier_built = True
            commit_step = t
        if action == "MEDIUM" and consec_low >= consec_needed:
            action = "LOW"

        pump   = PUMP_RATE[action]
        h_new  = np.clip(h + recharge[regime] - pump, 0.05, 1.0)
        failed = (h_new <= H_CRIT) and (not barrier_built)
        if failed and not tipped: tipped = True
        if failed and first_failure_step is None: first_failure_step = t

        # Charge the barrier cost exactly once — at the step it is first committed
        # (consec_high just reached consec_needed for the first time)
        charge_barrier = (consec_high == consec_needed)
        cum_r += GAMMA**t * reward_fn(action, failed, False, charge_barrier)
        if failed: n_fail += 1
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T, "barrier_built": barrier_built,
            "commit_step": commit_step, "first_failure_step": first_failure_step}


# ── RDM-Conservative benchmark strategy ───────────────────────────────────────

def run_strategy_rdm_conservative(regime_seq, recharge, sigma_pre, sigma_post,
                                   pi_regime_0, pi_model_0, rng, T_monitor=12,
                                   h_barrier_threshold=0.50,
                                   h_low_threshold=0.55):
    """
    Robust Decision Making (RDM) — stress-tested conservative static policy.

    Simplified representation of the RDM approach (Lempert et al. 2006):
    In full RDM, policies are optimised across an ensemble of deeply uncertain
    futures to minimise worst-case regret. The resulting policy for coastal
    aquifer management typically commits to conservative extraction early and
    uses observable physical state (freshwater head h) as the trigger variable.

    This implementation uses the observable h (freshwater head proxy) as the
    decision trigger — distinguishing RDM from DAPP (which uses salinity obs):
      If h < h_low_threshold    → reduce pumping to LOW (precautionary)
      If h < h_barrier_threshold → build barrier (pre-committed safeguard)

    Key distinction from DAPP: RDM triggers on the physical state (h, the
    consequence variable) rather than a monitoring signal (salinity obs).
    This represents a stress-tested commitment based on state thresholds derived
    from worst-case scenario analysis, without real-time belief updating.
    """
    T = len(regime_seq)
    h = H_INIT; cum_r = 0.0; n_fail = 0; tipped = False
    barrier_built = False
    action = "MEDIUM"
    commit_step = None; first_failure_step = None

    for t, regime in enumerate(regime_seq):
        sigma = sigma_post if t >= T_monitor else sigma_pre

        # RDM state-based trigger (no probabilistic beliefs, no obs model)
        if h < h_low_threshold and action == "MEDIUM":
            action = "LOW"
        charge_barrier = False
        if not barrier_built and h < h_barrier_threshold:
            barrier_built = True
            charge_barrier = True
            commit_step = t

        pump   = PUMP_RATE[action]
        h_new  = np.clip(h + recharge[regime] - pump, 0.05, 1.0)
        failed = (h_new <= H_CRIT) and (not barrier_built)
        if failed and not tipped: tipped = True
        if failed and first_failure_step is None: first_failure_step = t

        cum_r += GAMMA**t * reward_fn(action, failed, False, charge_barrier)
        if failed: n_fail += 1
        h = h_new

    return {"cum_reward": cum_r, "failures": n_fail,
            "failure_rate": n_fail / T, "barrier_built": barrier_built,
            "commit_step": commit_step, "first_failure_step": first_failure_step}


# ── Experiment A: Belief tracking on 24-year proxy sequence ───────────────────

def experiment_A(obs_seq_real, recharge, sigma_pre, sigma_post, T_monitor=12):
    T = len(obs_seq_real)
    h  = H_INIT
    h_traj = [h]
    # Use DRY-dominant prior reflecting known Biscayne 2000–2010 conditions
    pi_r = np.array([0.50, 0.35, 0.15])
    pi_m = np.array([0.40, 0.40, 0.20])
    pi_r_hist = [pi_r.copy()]
    pi_m_hist = [pi_m.copy()]
    tipped = False

    for t, obs in enumerate(obs_seq_real):
        sigma = sigma_post if t >= T_monitor else sigma_pre
        # Update beliefs using proxy obs and current h (no direct h data → fixed)
        pi_r, pi_m = update_beliefs_step(
            pi_r, pi_m, h, obs, sigma, recharge, PUMP_RATE["MEDIUM"], tipped)
        # Advance h with MEDIUM pumping under expected regime
        expected_regime = REGIMES[np.argmax(pi_r)]
        h = np.clip(h + recharge[expected_regime] - PUMP_RATE["MEDIUM"], 0.05, 1.0)
        if h <= H_CRIT:
            tipped = True
        h_traj.append(h)
        pi_r_hist.append(pi_r.copy())
        pi_m_hist.append(pi_m.copy())

    return np.array(pi_r_hist), np.array(pi_m_hist), np.array(h_traj)


# ── Experiment B: Monte Carlo comparison ──────────────────────────────────────

def _sample_regime_seq(rng, trans_matrix, T, regime_prior=None):
    if regime_prior is None:
        regime_prior = np.array([1/3, 1/3, 1/3])
    idx = rng.choice(len(REGIMES), p=regime_prior)
    seq = [REGIMES[idx]]
    for _ in range(T - 1):
        idx = rng.choice(len(REGIMES), p=trans_matrix[idx])
        seq.append(REGIMES[idx])
    return seq


def experiment_B(trans_matrix, recharge, sigma_pre, sigma_post,
                 n_trials=1000, T=24, seed=42):
    rng = np.random.default_rng(seed)
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])

    keys = ["BDT_Baseline", "Robust_Only", "Myopic_POMDP",
            "DAPP", "RDM_Conservative", "Unified_Framework"]
    fail_rates = {k: [] for k in keys}
    costs      = {k: [] for k in keys}

    regime_prior = np.array([0.41, 0.43, 0.16])  # from HMM calibration

    for trial in range(n_trials):
        seq = _sample_regime_seq(rng, trans_matrix, T, regime_prior)

        r_bdt  = run_strategy_bdt(seq, recharge, sigma_pre, sigma_post,
                                   pi_r0.copy(), pi_m0.copy(), rng)
        r_rob  = run_strategy_robust_only(seq, recharge, sigma_pre, sigma_post,
                                           pi_r0.copy(), pi_m0.copy(), rng)
        r_myp  = run_strategy_myopic_pomdp(seq, recharge, sigma_pre, sigma_post,
                                            pi_r0.copy(), pi_m0.copy(), rng)
        r_dapp = run_strategy_dapp(seq, recharge, sigma_pre, sigma_post,
                                    pi_r0.copy(), pi_m0.copy(), rng)
        r_rdm  = run_strategy_rdm_conservative(seq, recharge, sigma_pre, sigma_post,
                                                pi_r0.copy(), pi_m0.copy(), rng)
        r_uni  = run_strategy_unified(seq, recharge, sigma_pre, sigma_post,
                                       pi_r0.copy(), pi_m0.copy(), rng)

        for k, r in zip(keys, [r_bdt, r_rob, r_myp, r_dapp, r_rdm, r_uni]):
            fail_rates[k].append(r["failure_rate"])
            costs[k].append(-r["cum_reward"])

        if (trial + 1) % 200 == 0:
            print(f"  MC trial {trial+1}/{n_trials} …", flush=True)

    results = {}
    for k in keys:
        arr = np.array(fail_rates[k])
        c   = np.array(costs[k])
        results[k] = {
            "failure_rate_mean": float(arr.mean()),
            "failure_rate_std":  float(arr.std()),
            "failure_rate_ci95": (float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5))),
            "cost_mean": float(c.mean()),
            "cost_std":  float(c.std()),
        }
    return results


# ── Experiment C: Sensitivity analysis ────────────────────────────────────────

def experiment_C(trans_matrix, recharge, sigma_pre, sigma_post,
                 n_trials=200, T=24, seed=123):
    delta_vals  = [0.05, 0.10, 0.20, 0.30]
    lambda_vals = [0.5,  1.0,  1.5,  2.0]
    pi_r0 = np.array([0.50, 0.35, 0.15])
    pi_m0 = np.array([0.40, 0.40, 0.20])
    regime_prior = np.array([0.41, 0.43, 0.16])

    grid = np.zeros((len(delta_vals), len(lambda_vals)))
    for i, delta in enumerate(delta_vals):
        for j, lam in enumerate(lambda_vals):
            rng = np.random.default_rng(seed + i * 10 + j)
            frs = []
            for _ in range(n_trials):
                seq = _sample_regime_seq(rng, trans_matrix, T, regime_prior)
                r = run_strategy_unified(seq, recharge, sigma_pre, sigma_post,
                                          pi_r0.copy(), pi_m0.copy(), rng,
                                          delta=delta, lam=lam)
                frs.append(r["failure_rate"])
            grid[i, j] = float(np.mean(frs))
            print(f"  δ={delta:.2f} λ={lam:.1f} → fail={grid[i,j]*100:.1f}%",
                  flush=True)

    return grid, delta_vals, lambda_vals


# ── Plotting ───────────────────────────────────────────────────────────────────

DARK = "#263238"; PANEL = "#FAFAFA"


def plot_belief_trajectory(pi_r_hist, pi_m_hist, h_traj, obs_seq_real,
                            save_path=None):
    T = len(obs_seq_real)
    x = np.arange(T)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), facecolor="white")
    fig.suptitle("Biscayne Aquifer: Bayesian Belief Update on Real Climate Forcing\n"
                 "ERA5 Precipitation + NOAA Sea Level + ENSO ONI (2000–2024)",
                 fontsize=12, fontweight="bold", color=DARK)

    ax = axes[0]; ax.set_facecolor(PANEL)
    ax.plot(x, obs_seq_real, color="#1565C0", lw=1.2, alpha=0.8, label="proxy obs")
    ax.axhline(0.5, color="#EF5350", ls="--", lw=0.8, alpha=0.6, label="midpoint")
    ax.set_ylim(0, 1); ax.set_ylabel("Salinity proxy"); ax.set_title("Observation Sequence")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]; ax.set_facecolor(PANEL)
    ax.plot(x, h_traj[1:T+1], color="#2E7D32", lw=1.4, label="h (freshwater head)")
    ax.axhline(H_CRIT, color="#EF5350", ls="--", lw=1.0, label=f"H_crit={H_CRIT}")
    ax.set_ylim(0, 1); ax.set_ylabel("Normalised h"); ax.set_title("Freshwater Head")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[2]; ax.set_facecolor(PANEL)
    pi_r = pi_r_hist[1:T+1]
    ax.stackplot(x, pi_r[:,0], pi_r[:,1], pi_r[:,2],
                 labels=["DRY","STATIONARY","WET"],
                 colors=["#F4A460","#4CAF50","#4FC3F7"], alpha=0.85)
    ax.set_ylim(0, 1); ax.set_ylabel("P(regime)")
    ax.set_title("Climate Regime Beliefs π(θ)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[3]; ax.set_facecolor(PANEL)
    pi_m = pi_m_hist[1:T+1]
    ax.stackplot(x, pi_m[:,0], pi_m[:,1], pi_m[:,2],
                 labels=["M1: Sharp-interface","M2: Dispersive","M3: Tipping-point"],
                 colors=["#9C27B0","#26A69A","#D32F2F"], alpha=0.85)
    ax.set_ylim(0, 1); ax.set_ylabel("P(model)")
    ax.set_xlabel("Month (2000-01 → 2024-12)")
    ax.set_title("Intrusion Model Beliefs π(m)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_mc_comparison(mc_results, save_path=None):
    keys   = list(mc_results.keys())
    means  = [mc_results[k]["failure_rate_mean"] * 100 for k in keys]
    lows   = [mc_results[k]["failure_rate_ci95"][0] * 100 for k in keys]
    highs  = [mc_results[k]["failure_rate_ci95"][1] * 100 for k in keys]
    errs   = [[m - l for m, l in zip(means, lows)],
              [h - m for m, h in zip(means, highs)]]
    costs  = [mc_results[k]["cost_mean"] for k in keys]

    nice = {
        "BDT_Baseline":      "BDT\nBaseline",
        "Robust_Only":       "Robust\nSatisficing",
        "Myopic_POMDP":      "Myopic\nPOMDP",
        "DAPP":              "DAPP\n(Haasnoot\n2013)",
        "RDM_Conservative":  "RDM\nConservative\n(Lempert 2006)",
        "Unified_Framework": "Unified\nFramework",
    }
    labels = [nice.get(k, k) for k in keys]
    # Colour coding: grey scale for baselines → blue for EMS benchmarks → red for unified
    colors = ["#90A4AE",  # BDT — grey
              "#81C784",  # Robust — green
              "#FFB74D",  # Myopic — orange
              "#42A5F5",  # DAPP — blue (EMS benchmark)
              "#AB47BC",  # RDM — purple (EMS benchmark)
              "#E53935"]  # Unified — red (proposed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    # Panel 1: failure rates
    ax1.set_facecolor(PANEL)
    bars = ax1.bar(labels, means, color=colors, width=0.60, zorder=3)
    ax1.errorbar(labels, means, yerr=errs, fmt="none", color="black",
                 capsize=5, lw=1.5, zorder=4)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(means) * 0.01,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax1.set_ylabel("Mean failure rate (%)", fontsize=11)
    ax1.set_title("Strategy Comparison: Failure Rate\n(n=1,000 MC trials, 95% CI)",
                  fontsize=10)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.set_ylim(0, max(means) * 1.35)
    ax1.grid(axis="y", alpha=0.35, zorder=0)
    ax1.tick_params(axis="x", labelsize=8)

    # Panel 2: normalised costs
    ax2.set_facecolor(PANEL)
    bars2 = ax2.bar(labels, costs, color=colors, width=0.60, zorder=3)
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(costs) * 0.01,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_ylabel("Mean normalised cost (−cumulative reward)", fontsize=11)
    ax2.set_title("Strategy Comparison: Normalised Cost\n(n=1,000 MC trials)",
                  fontsize=10)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_ylim(0, max(costs) * 1.2)
    ax2.grid(axis="y", alpha=0.35, zorder=0)
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Biscayne Aquifer POMDP — 6-Strategy Monte Carlo Comparison\n"
                 "HMM-calibrated regime model, ERA5/NOAA real climate forcing",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_sensitivity_heatmap(grid, delta_vals, lambda_vals, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    im = ax.imshow(grid * 100, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=np.ceil(grid.max() * 100))
    ax.set_xticks(range(len(lambda_vals)))
    ax.set_xticklabels([f"λ={v}" for v in lambda_vals])
    ax.set_yticks(range(len(delta_vals)))
    ax.set_yticklabels([f"δ={v}" for v in delta_vals])
    ax.set_xlabel("VoI weight λ (LAMBDA_VOI)")
    ax.set_ylabel("Risk tolerance δ (DELTA)")
    ax.set_title("Sensitivity: Mean Failure Rate (%) — Unified Framework\n"
                 "Biscayne Aquifer, n=200 trials per cell", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean failure rate (%)")
    for i in range(len(delta_vals)):
        for j in range(len(lambda_vals)):
            ax.text(j, i, f"{grid[i,j]*100:.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if grid[i,j]*100 > grid.max()*50 else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-trials",        type=int, default=1000)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--output-dir",       type=str, default=str(RESULTS_DIR))
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Biscayne Aquifer POMDP Experiment")
    print("=" * 65)

    # ── Step 1: Calibrated parameters ─────────────────────────────────────────
    print("\n[1/5] Calibrating parameters from real climate data …")
    obs_seq_real, proxy_df = build_proxy()

    calib = calibrate_regimes(save_dir=str(DATA_DIR))
    trans_matrix = calib["transition_matrix"]
    recharge     = calib["recharge_rates"]
    print(f"  Recharge: {recharge}")

    noise = calibrate_obs_noise()
    sigma_pre  = noise["sigma_pre"]
    sigma_post = noise["sigma_post"]
    print(f"  σ_pre={sigma_pre:.4f}  σ_post={sigma_post:.4f}")

    # ── Step 2: Experiment A ───────────────────────────────────────────────────
    print(f"\n[2/5] Experiment A: Belief update on {len(obs_seq_real)}-month proxy …")
    pi_r_hist, pi_m_hist, h_traj = experiment_A(
        obs_seq_real, recharge, sigma_pre, sigma_post)

    plot_belief_trajectory(
        pi_r_hist, pi_m_hist, h_traj, obs_seq_real,
        save_path=str(out / "fig_belief_trajectory.png"))
    np.save(str(out / "pi_r_hist_real.npy"), pi_r_hist)
    np.save(str(out / "pi_m_hist_real.npy"), pi_m_hist)
    print("  Belief trajectory complete.")

    # ── Step 3: Experiment B ───────────────────────────────────────────────────
    print(f"\n[3/5] Experiment B: MC comparison ({args.mc_trials} trials) …")
    mc_results = experiment_B(
        trans_matrix, recharge, sigma_pre, sigma_post,
        n_trials=args.mc_trials, T=24, seed=args.seed)

    print("\n  Monte Carlo results:")
    for strat, res in mc_results.items():
        fr = res["failure_rate_mean"] * 100
        ci = tuple(v * 100 for v in res["failure_rate_ci95"])
        print(f"    {strat:22s}  fail={fr:.1f}%  CI [{ci[0]:.1f}%,{ci[1]:.1f}%]"
              f"  cost={res['cost_mean']:.1f}")

    with open(str(out / "mc_results.json"), "w") as f:
        json.dump(mc_results, f, indent=2)
    plot_mc_comparison(mc_results, save_path=str(out / "fig_mc_comparison.png"))

    # ── Step 4: Experiment C ───────────────────────────────────────────────────
    if not args.skip_sensitivity:
        print("\n[4/5] Experiment C: Sensitivity (δ × λ grid, 200 trials/cell) …")
        grid, delta_vals, lambda_vals = experiment_C(
            trans_matrix, recharge, sigma_pre, sigma_post, n_trials=200, T=24)
        np.save(str(out / "sensitivity_grid.npy"), grid)
        plot_sensitivity_heatmap(
            grid, delta_vals, lambda_vals,
            save_path=str(out / "fig_sensitivity_heatmap.png"))
    else:
        print("\n[4/5] Sensitivity skipped.")

    # ── Step 5: Summary ────────────────────────────────────────────────────────
    print("\n[5/5] Saving experiment summary …")
    summary = {
        "data_sources": {
            "precipitation": "Open-Meteo ERA5 reanalysis, Miami-Dade (2000–2024)",
            "sea_level":     "NOAA Tides & Currents, Key West station 8724580",
            "enso_oni":      "NOAA CPC Oceanic Niño Index (monthly)",
            "obs_proxy":     "Physics-based: obs = sigmoid(α·SLR_z + β·(−precip_z) + γ·ONI_z)",
        },
        "calibrated_params": {
            "recharge_rates":     recharge,
            "sigma_pre":          round(sigma_pre, 4),
            "sigma_post":         round(sigma_post, 4),
            "transition_matrix":  trans_matrix.tolist(),
            "regime_occupancy":   {"DRY": 0.41, "STATIONARY": 0.43, "WET": 0.16},
        },
        "mc_results": mc_results,
        "n_proxy_months": int(len(obs_seq_real)),
    }
    with open(str(out / "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs written to: {out}/")
    uni   = mc_results["Unified_Framework"]
    bdt   = mc_results["BDT_Baseline"]
    dapp  = mc_results["DAPP"]
    rdm   = mc_results["RDM_Conservative"]
    print(f"\nKey results (failure rate | normalised cost):")
    print(f"  BDT Baseline       : {bdt['failure_rate_mean']*100:.1f}% | {bdt['cost_mean']:.1f}")
    print(f"  DAPP               : {dapp['failure_rate_mean']*100:.1f}% | {dapp['cost_mean']:.1f}")
    print(f"  RDM Conservative   : {rdm['failure_rate_mean']*100:.1f}% | {rdm['cost_mean']:.1f}")
    print(f"  Unified Framework  : {uni['failure_rate_mean']*100:.1f}% | {uni['cost_mean']:.1f}")
    print(f"\n  Unified vs BDT:  "
          f"{bdt['failure_rate_mean']*100 - uni['failure_rate_mean']*100:.1f} pp failure reduction, "
          f"{100*(bdt['cost_mean'] - uni['cost_mean'])/bdt['cost_mean']:.1f}% cost reduction")
    print(f"  Unified vs DAPP: "
          f"{dapp['failure_rate_mean']*100 - uni['failure_rate_mean']*100:.1f} pp failure reduction")
    print(f"  Unified vs RDM:  "
          f"{rdm['failure_rate_mean']*100 - uni['failure_rate_mean']*100:.1f} pp failure reduction")
    print("Done.")


if __name__ == "__main__":
    main()

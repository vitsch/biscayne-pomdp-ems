# Biscayne Aquifer POMDP Experiment

Companion code for:

> Jakaite, L. & Schetinin, V. (2026). *A Unified POMDP Framework for Decisions under Deep Uncertainty: Robust Satisficing, Directed Learning, and Real Options.* European Journal of Operational Research (under revision).

---

## Repository structure

```
biscayne/
  biscayne_main.py              # Main experiment — runs all 6 strategies, sensitivity grid
  run_revision_experiments.py   # Barrier-timing diagnostic, OV calibration, non-informative-prior check
  make_barrier_timing_figure.py # Raw per-trial barrier-timing data + Figure 3
  run_cpu_stability_experiment.py # CPU-time benchmark + 5-seed stability check
  make_graphical_abstract.py    # Generates graphical abstract figure
  modules/
    data_loader.py              # USGS/NOAA groundwater data loader
    enso_loader.py              # NOAA CPC ONI index loader
    regime_calibrator.py        # HMM climate regime calibration (ERA5)
    obs_calibrator.py           # Observation noise calibration
    salinity_proxy.py           # Physics-based salinity proxy
  data/
    raw/                        # ERA5, NOAA sea-level, ONI, salinity proxy CSVs
    calibration_params.json     # Pre-computed HMM parameters
    transition_matrix.csv       # Calibrated regime transition matrix
  results/
    dapp_final/                 # Canonical results cited in the paper (n=1,000)
      experiment_summary.json / mc_results.json  # 6-strategy failure rates, costs
      fig_mc_comparison.png
      fig_barrier_timing.png, barrier_timing.json, barrier_timing_raw.json
      fig_belief_trajectory.png
      fig_sensitivity_heatmap.png, sensitivity_grid.npy
      ov_calibration.json       # Real-options calibration at b0 (Remark 2 / Section 3.4)
      uniform_prior_check.json  # Non-informative-prior robustness check (Section 4.4.5)
concept_colab.py                # Conceptual illustration of Algorithm 1
requirements.txt
```

---

## Reproducing the results

```bash
pip install -r requirements.txt

cd biscayne
python biscayne_main.py --mc-trials 1000 --seed 42
```

Expected output (`results/mc_results.json`), matching Table 3 of the paper:

| Strategy | Failure rate | Cost | CPU time/trial |
|---|---|---|---|
| Unified Framework | 0.0 % | 56.1 | 88.5 ms |
| RDM-Conservative (Lempert 2003) | 0.0 % | 52.9 | 0.044 ms |
| DAPP (Haasnoot 2013) | 9.8 % | 133.6 | 0.103 ms |
| Robust Satisficing | 43.5 % | 297.6 | 71.3 ms |
| Myopic POMDP | 54.3 % | 357.3 | 3.94 ms |
| BDT Baseline | 58.6 % | 390.9 | 0.048 ms |

Runtime: ~5–10 minutes on a standard laptop (Python 3.10+), including the 4×4 sensitivity grid.

To reproduce the remaining diagnostics cited in the paper (Table 4's barrier-timing
statistics, Figure 3, the real-options calibration in Remark 2 / Section 3.4, and the
non-informative-prior robustness check in Section 4.4.5):

```bash
python run_revision_experiments.py        # barrier_timing.json, ov_calibration.json, uniform_prior_check.json
python make_barrier_timing_figure.py       # fig_barrier_timing.png, barrier_timing_raw.json (500 trials)
python run_cpu_stability_experiment.py     # CPU-time benchmark + 5-seed stability check (Section 4.4.5)
```

By default these write to `results/revision_v4/` and `results/cpu_stability/`
respectively; pass `--output-dir` (where supported) to write elsewhere.

---

## Data sources

| Dataset | Source | Period |
|---|---|---|
| ERA5 precipitation | ECMWF open data | 2000–2023 |
| NOAA Key West sea level | NOAA Tides & Currents | 2000–2023 |
| ENSO ONI index | NOAA CPC | 2000–2023 |
| Physics-based salinity proxy | Computed (see `modules/salinity_proxy.py`) | 2000–2023 |

---

## Conceptual illustration (optional)

`concept_colab.py` is an interactive illustration of Algorithm 1 on a synthetic
6-step trajectory. It does not reproduce the paper's quantitative results but
demonstrates the belief-update and decision logic in isolation. Note that it
predates the reward-function and action-selection fixes described below, so its
qualitative illustration of pumping/monitoring choices does not reflect the
current, published mechanism.

```bash
python concept_colab.py
# Outputs: output/f4_reproduced.pdf, output/f5_reproduced.pdf
```

---

## Revision history

The code was corrected during EJOR peer review to fix four issues in the original
(EMS-submission-era) implementation, each verified against a full rerun before being
adopted:

1. **Non-myopic value-of-information.** `voi_estimate()` previously computed a
   one-step reward difference (self-documented as "myopic" in the original
   docstring), contradicting the non-myopic VoI defined in the paper (Eq. 5,
   Proposition 2). It now uses a `T_horizon`-step rollout value, matching the
   convention already used for satisficing and real-options estimation.
2. **Action-selection causality.** The Unified Framework's action-selection score
   previously added `lambda*VoI` and `mu*OV` as constants identical across every
   candidate action, so they had no effect on which action was chosen. `lambda*VoI`
   now conditions specifically on the monitoring action; `mu*OV`'s effect is
   correctly confined to the (separate, already-correct) barrier-commitment trigger.
3. **Demand-shortfall cost.** The reward function had no benefit term for pumping,
   so any cost-aware strategy trivially converged to minimal pumping regardless of
   risk. A water-demand target and unmet-demand penalty now give pumping intensity
   a genuine, risk-sensitive trade-off (see Eq. 10/Section 4.3 of the paper).
4. **Real-options post-commitment cost.** `option_value_raw`'s committed-barrier
   branch assumed continued minimal pumping even after the barrier eliminates
   failure risk; it now correctly assumes cost-minimising pumping post-commitment.

Fixing (3) changed the six-strategy comparison materially (Myopic POMDP's failure
rate rose from 41.6% to 54.3%, correctly reflecting the weakness of one-step-ahead
reasoning; RDM-Conservative became cheaper than the Unified Framework, reflecting a
discounting effect of earlier barrier commitment) — see the paper's Section 4.4.3
and Discussion for the full explanation. Fixes (1)-(2) left the headline 0.0%
failure result unchanged.

---

## License

MIT — see `LICENSE`.

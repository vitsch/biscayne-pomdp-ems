# Biscayne Aquifer POMDP Experiment

Companion code for:

> Jakaite, L. & Schetinin, V. (2026). *A Unified POMDP Framework with Robust Satisficing, Directed Learning, and Real Options for Environmental Decisions under Deep Uncertainty.* Environmental Modelling & Software (submitted).

---

## Repository structure

```
biscayne/
  biscayne_main.py            # Main experiment — runs all 6 strategies
  make_graphical_abstract.py  # Generates graphical abstract figure
  modules/
    data_loader.py            # USGS/NOAA groundwater data loader
    enso_loader.py            # NOAA CPC ONI index loader
    regime_calibrator.py      # HMM climate regime calibration (ERA5)
    obs_calibrator.py         # Observation noise calibration
    salinity_proxy.py         # Physics-based salinity proxy
  data/
    raw/                      # ERA5, NOAA sea-level, ONI, salinity proxy CSVs
    calibration_params.json   # Pre-computed HMM parameters
    transition_matrix.csv     # Calibrated regime transition matrix
  results/
    dapp_final/               # Canonical results cited in the paper (n=1,000)
      mc_results.json         # 6-strategy failure rates and costs
      fig_mc_comparison.png
      fig_barrier_timing.png
      fig_belief_trajectory.png
      fig_sensitivity_heatmap.png
concept_colab.py              # Conceptual illustration of Algorithm 1
requirements.txt
```

---

## Reproducing the results

```bash
pip install -r requirements.txt

cd biscayne
python biscayne_main.py --mc-trials 1000 --seed 42
```

Expected output (`results/mc_results.json`):

| Strategy | Failure rate | Cost |
|---|---|---|
| Unified Framework | 0.0 % | 34.2 |
| RDM-Conservative (Lempert 2006) | 0.0 % | 35.7 |
| DAPP (Haasnoot 2013) | 9.5 % | 118.9 |
| Myopic POMDP | 41.6 % | — |
| Robust Satisficing | 43.4 % | — |
| BDT Baseline | 59.1 % | 393.9 |

Runtime: ~3–5 minutes on a standard laptop (Python 3.10+).

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
demonstrates the belief-update and decision logic in isolation.

```bash
python concept_colab.py
# Outputs: output/f4_reproduced.pdf, output/f5_reproduced.pdf
```

---

## License

MIT — see `LICENSE`.

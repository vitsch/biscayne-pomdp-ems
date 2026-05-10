"""
Module M2: HMM-based climate regime calibration.

Fits a 3-state Gaussian HMM to the monthly precipitation time series
(2000–2024) to calibrate the regime transition matrix Λ_real and
emission parameters that replace the synthetic values in els_is_claude_colab.py.

States: 0 = DRY, 1 = STATIONARY, 2 = WET
  (ordered by ascending mean precipitation after fitting)

Output artefacts:
  - Λ_real     : (3×3) transition matrix
  - means_real : (3,) mean precipitation per state (mm/month)
  - stds_real  : (3,) std precipitation per state
  - regime_seq : (T,) int array of decoded regime labels (Viterbi)
  - recharge   : dict mapping regime name → recharge rate (calibrated)
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm

DATA_DIR  = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent / "data"
REGIME_NAMES = ["DRY", "STATIONARY", "WET"]


def _assign_state_labels(model: hmm.GaussianHMM) -> np.ndarray:
    """Return index permutation so states are sorted by ascending mean precip."""
    return np.argsort(model.means_.ravel())


def calibrate_regimes(
    precip_path: str | None = None,
    n_states: int = 3,
    n_iter: int = 500,
    random_seed: int = 42,
    save_dir: str | None = None,
) -> dict:
    """
    Fit a 3-state Gaussian HMM to monthly precipitation.

    Returns
    -------
    result : dict with keys
        transition_matrix  – (3,3) np.ndarray, rows sum to 1
        means_mm           – (3,) mean precip per regime (DRY→WET order)
        stds_mm            – (3,) std precip per regime
        regime_seq         – (T,) int Viterbi state sequence (0=DRY,2=WET)
        recharge_rates     – dict {DRY: float, STATIONARY: float, WET: float}
        precip_series      – pd.Series with DatetimeIndex (input data)
        fit_stats          – dict {log_likelihood, aic, bic}
    """
    precip_path = precip_path or str(DATA_DIR / "noaa_precip.csv")
    df = pd.read_csv(precip_path, parse_dates=["date"])
    precip = df.set_index("date")["precip_mm"]

    X = precip.values.reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_seed,
            init_params="stmc",
        )
        model.fit(X)

    # Re-order states DRY < STATIONARY < WET by mean precipitation
    order = _assign_state_labels(model)
    inv_order = np.argsort(order)  # inverse permutation

    trans_raw = model.transmat_
    # Reorder rows and columns
    trans_cal = trans_raw[np.ix_(order, order)]
    means_cal = model.means_.ravel()[order]
    stds_cal  = np.sqrt(model.covars_.ravel())[order]

    raw_seq = model.predict(X)
    regime_seq = inv_order[raw_seq]  # remap to 0=DRY, 1=STAT, 2=WET

    n_params = n_states * n_states + 2 * n_states  # trans + means + variances
    ll = model.score(X) * len(X)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(len(X))

    # Calibrate recharge rates from mean precipitation
    # Reference: synthetic values {DRY:-0.08, STAT:-0.01, WET:+0.05}
    # Scale linearly so that STATIONARY mean precip maps to -0.01
    precip_mean_all = precip.mean()
    recharge = {}
    for i, name in enumerate(REGIME_NAMES):
        # Deviation from overall mean, normalised
        dev = (means_cal[i] - precip_mean_all) / precip_mean_all
        # Map: dry deviation → negative recharge, wet → positive
        recharge[name] = float(np.clip(dev * 0.12, -0.12, 0.08))

    result = {
        "transition_matrix": trans_cal,
        "means_mm": means_cal,
        "stds_mm": stds_cal,
        "regime_seq": regime_seq,
        "recharge_rates": recharge,
        "precip_series": precip,
        "fit_stats": {"log_likelihood": ll, "aic": aic, "bic": bic},
    }

    print("HMM calibration complete:")
    print(f"  Log-likelihood: {ll:.1f}   AIC: {aic:.1f}   BIC: {bic:.1f}")
    print(f"  Regime means (DRY/STAT/WET): "
          f"{means_cal[0]:.1f} / {means_cal[1]:.1f} / {means_cal[2]:.1f} mm/month")
    print(f"  Calibrated recharge rates: {recharge}")
    print(f"  Transition matrix:\n{np.round(trans_cal, 3)}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_dir / "transition_matrix.csv", trans_cal,
                   delimiter=",", header="DRY,STATIONARY,WET", comments="")
        regime_df = pd.DataFrame({"date": precip.index, "regime": regime_seq,
                                  "regime_name": [REGIME_NAMES[r] for r in regime_seq]})
        regime_df.to_csv(save_dir / "regime_sequence.csv", index=False)
        with open(save_dir / "calibration_params.json", "w") as f:
            json.dump({
                "transition_matrix": trans_cal.tolist(),
                "means_mm": means_cal.tolist(),
                "stds_mm": stds_cal.tolist(),
                "recharge_rates": recharge,
                "fit_stats": {"log_likelihood": ll, "aic": aic, "bic": bic},
            }, f, indent=2)
        print(f"  Saved calibration artefacts → {save_dir}")

    return result


if __name__ == "__main__":
    result = calibrate_regimes(
        save_dir="/home/vs/Downloads/Project_Conceptual_EMS/biscayne/data"
    )

    # Show state occupancy
    seq = result["regime_seq"]
    total = len(seq)
    for i, name in enumerate(REGIME_NAMES):
        count = np.sum(seq == i)
        print(f"  {name}: {count} months ({100*count/total:.1f}%)")

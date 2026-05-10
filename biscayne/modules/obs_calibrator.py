"""
Module M3: Observation noise calibration.

Estimates the observation noise parameter σ for the POMDP from the
salinity proxy residuals. The proxy residual variance captures
unmodelled variability (local well effects, tidal fluctuations,
measurement error) that the model treats as observation noise.

Method:
  1. Fit a piecewise-regime mean to obs_proxy using the decoded regime sequence.
  2. Compute residuals = obs_proxy − regime_mean.
  3. σ_pre  = std of residuals across all months (unconditional noise).
  4. σ_post = std of residuals within each regime (conditional noise, lower
     because we account for regime identity — analogous to post-monitoring state).

The ratio σ_post / σ_pre reflects the Value of Information from monitoring,
matching the OBS_SIGMA_PRE / OBS_SIGMA_POST distinction in els_is_claude_colab.py.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REGIME_NAMES = ["DRY", "STATIONARY", "WET"]


def calibrate_obs_noise(
    proxy_path: str | None = None,
    regime_path: str | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Estimate observation noise parameters from proxy residuals.

    Returns
    -------
    dict with keys:
        sigma_pre   – float, unconditional observation std
        sigma_post  – float, regime-conditional observation std
        regime_means – dict {regime_name: mean_obs}
        summary_df  – pd.DataFrame with per-month columns
    """
    proxy_path  = proxy_path  or str(DATA_DIR / "raw" / "salinity_proxy.csv")
    regime_path = regime_path or str(DATA_DIR / "regime_sequence.csv")

    proxy  = pd.read_csv(proxy_path,  parse_dates=["date"])
    regime = pd.read_csv(regime_path, parse_dates=["date"])

    df = proxy[["date", "obs_proxy"]].merge(
         regime[["date", "regime", "regime_name"]], on="date")

    # Regime-conditional means
    regime_means = df.groupby("regime_name")["obs_proxy"].mean().to_dict()

    # Residuals
    df["regime_mean"] = df["regime_name"].map(regime_means)
    df["residual"]    = df["obs_proxy"] - df["regime_mean"]

    sigma_pre  = float(df["obs_proxy"].std(ddof=0))
    sigma_post = float(df["residual"].std(ddof=0))

    result = {
        "sigma_pre":    sigma_pre,
        "sigma_post":   sigma_post,
        "voi_ratio":    round(sigma_pre / sigma_post, 3),
        "regime_means": regime_means,
    }

    print("Observation noise calibration:")
    print(f"  σ_pre  (unconditional)  = {sigma_pre:.4f}")
    print(f"  σ_post (regime-knowing) = {sigma_post:.4f}")
    print(f"  VoI noise ratio         = {result['voi_ratio']:.3f}x")
    print(f"  Regime mean obs_proxy:")
    for name, val in sorted(regime_means.items()):
        print(f"    {name}: {val:.3f}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump({k: v for k, v in result.items()
                       if k != "regime_means"} | {"regime_means": regime_means},
                      f, indent=2)
        print(f"  Saved → {save_path}")

    result["summary_df"] = df
    return result


if __name__ == "__main__":
    result = calibrate_obs_noise(
        save_path="/home/vs/Downloads/Project_Conceptual_EMS/biscayne/data/obs_noise_params.json"
    )

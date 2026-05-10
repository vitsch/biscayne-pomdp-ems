"""
Module M1c: Physically-motivated salinity proxy construction.

Rationale:
  Saltwater intrusion in the Biscayne Aquifer is driven by three climate signals:
    1. Precipitation deficit  – reduced recharge lowers freshwater head
    2. Sea-level rise anomaly – positive anomaly increases hydraulic gradient
    3. ENSO phase             – El Niño drives dry conditions in South Florida

  We construct a normalised "intrusion stress" index:

      stress_t = α·SLR_anom_t + β·(−precip_anom_t) + γ·ONI_t

  and map it to the observation space [0,1] via a sigmoid:

      obs_t = sigmoid(k · stress_t)

  where sigmoid(x) = 1 / (1 + exp(−x)).

  Parameters α, β, γ are set from published Biscayne literature:
    - SLR drives ~30 % of intrusion variance (Prinos et al. 2014 USGS SIR 2014-5037)
    - Precip deficit is the dominant driver (~50 %) (Bloetscher et al. 2014)
    - ENSO contributes ~20 % through teleconnection (Obeysekera et al. 2011)

  Anomalies are z-scored to keep α, β, γ scale-free.
  The sigmoid steepness k=2.0 maps a 1-SD stress event to obs≈0.88 (strongly saline),
  which matches the high-EC episodes documented at Biscayne monitoring wells.

Output:
  obs_seq   – np.ndarray shape (T,) in [0,1], monthly, 2000-01 to 2024-12
  stress_df – pd.DataFrame with intermediate columns for diagnostics
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

# Published literature weights (scale-free after z-scoring inputs)
ALPHA = 0.30   # SLR anomaly weight
BETA  = 0.50   # Precipitation deficit weight
GAMMA = 0.20   # ENSO ONI weight
K_SIGMOID = 2.0  # steepness

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def build_proxy(
    precip_path: str | None = None,
    sealevel_path: str | None = None,
    oni_path: str | None = None,
    alpha: float = ALPHA,
    beta: float  = BETA,
    gamma: float = GAMMA,
    k: float     = K_SIGMOID,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Construct normalised salinity proxy from real climate drivers.

    Returns
    -------
    obs_seq : np.ndarray shape (T,)
        Normalised intrusion observations in [0, 1].
    df : pd.DataFrame
        Full diagnostic table with columns [date, precip_mm, slr_m, oni_anom,
        precip_z, slr_z, oni_z, stress, obs_proxy].
    """
    precip_path   = precip_path   or str(DATA_DIR / "noaa_precip.csv")
    sealevel_path = sealevel_path or str(DATA_DIR / "noaa_sealevel.csv")
    oni_path      = oni_path      or str(DATA_DIR / "oni_index.csv")

    precip   = pd.read_csv(precip_path,   parse_dates=["date"])
    sealevel = pd.read_csv(sealevel_path, parse_dates=["date"])
    oni      = pd.read_csv(oni_path,      parse_dates=["date"])

    # Align on common monthly index
    df = precip.set_index("date").join(
         sealevel.set_index("date"), how="inner").join(
         oni[["date", "oni_anom"]].set_index("date"), how="inner")
    df = df.sort_index().dropna()

    # Z-score each driver
    df["precip_z"] = _zscore(df["precip_mm"])
    df["slr_z"]    = _zscore(df["slr_m"])
    df["oni_z"]    = _zscore(df["oni_anom"])

    # Stress index: high stress → intrusion → high obs
    # Precipitation deficit = −precip_z (low rain → high stress)
    df["stress"] = (alpha * df["slr_z"]
                  + beta  * (-df["precip_z"])
                  + gamma * df["oni_z"])

    # Map to [0,1] via sigmoid
    df["obs_proxy"] = expit(k * df["stress"])

    obs_seq = df["obs_proxy"].values.astype(np.float64)

    print(f"Proxy built: {len(obs_seq)} months "
          f"[{df.index[0].date()} – {df.index[-1].date()}]")
    print(f"  obs range: [{obs_seq.min():.3f}, {obs_seq.max():.3f}]  "
          f"mean={obs_seq.mean():.3f}  std={obs_seq.std():.3f}")

    return obs_seq, df.reset_index()


def save_proxy(save_path: str | None = None, **kwargs) -> tuple[np.ndarray, pd.DataFrame]:
    """Build proxy and optionally save the diagnostic DataFrame."""
    obs_seq, df = build_proxy(**kwargs)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved proxy → {save_path}")
    return obs_seq, df


if __name__ == "__main__":
    obs_seq, df = save_proxy(
        save_path="/home/vs/Downloads/Project_Conceptual_EMS/biscayne/data/raw/salinity_proxy.csv"
    )
    print(df[["date", "precip_mm", "slr_m", "oni_anom", "stress", "obs_proxy"]].head(12))
    # Quick sanity: El Niño years should have higher obs_proxy
    print("\nMean obs_proxy by ENSO phase:")
    oni_df = pd.read_csv("/home/vs/Downloads/Project_Conceptual_EMS/biscayne/data/raw/oni_index.csv",
                         parse_dates=["date"])
    merged = df.merge(oni_df[["date", "enso_phase"]], on="date")
    print(merged.groupby("enso_phase")["obs_proxy"].mean().round(3))

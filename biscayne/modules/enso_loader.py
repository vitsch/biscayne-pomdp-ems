"""
Module M1b: ENSO ONI index download and monthly alignment.
Downloads NOAA CPC Oceanic Nino Index and interpolates to calendar months.
"""

import io
import urllib.request
import numpy as np
import pandas as pd

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Season centre-month mapping: DJF → Jan, JFM → Feb, …
_SEASON_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def download_oni(start_year: int = 2000, end_year: int = 2024,
                 save_path: str | None = None) -> pd.DataFrame:
    """
    Download NOAA CPC ONI index and return monthly DataFrame.

    Columns:
        date        – first day of each month
        oni_anom    – 3-month running mean SST anomaly (°C)
        enso_phase  – categorical: 'ElNino' (>+0.5), 'LaNina' (<-0.5), 'Neutral'

    Parameters
    ----------
    start_year, end_year : inclusive year range to keep
    save_path : if given, saves CSV to this path
    """
    with urllib.request.urlopen(ONI_URL, timeout=20) as resp:
        text = resp.read().decode("utf-8")

    rows = []
    for line in text.strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) < 4:
            continue
        season, year_str, _, anom_str = parts[0], parts[1], parts[2], parts[3]
        month = _SEASON_MONTH.get(season)
        if month is None:
            continue
        year = int(year_str)
        # NDJ season belongs to the *next* calendar year
        if season == "NDJ":
            year += 1
        rows.append({"date": pd.Timestamp(year, month, 1),
                     "oni_anom": float(anom_str)})

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    df = df[(df["date"].dt.year >= start_year) &
            (df["date"].dt.year <= end_year)].reset_index(drop=True)

    df["enso_phase"] = "Neutral"
    df.loc[df["oni_anom"] > 0.5,  "enso_phase"] = "ElNino"
    df.loc[df["oni_anom"] < -0.5, "enso_phase"] = "LaNina"

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved ONI data: {len(df)} months → {save_path}")

    return df


if __name__ == "__main__":
    df = download_oni(save_path="/home/vs/Downloads/Project_Conceptual_EMS/biscayne/data/raw/oni_index.csv")
    print(df.head(12))
    print(f"\nPhase counts:\n{df['enso_phase'].value_counts()}")

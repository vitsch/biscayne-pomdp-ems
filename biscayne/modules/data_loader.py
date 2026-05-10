"""
Module M1: USGS NWIS data acquisition and preprocessing for Biscayne Aquifer.

Downloads specific conductance, groundwater level, and chloride records
from Miami-Dade County monitoring wells, then aligns and normalises
all series to monthly time steps in the [0,1] observation space
expected by els_is_claude_colab.py.
"""

import dataretrieval.nwis as nwis
import pandas as pd
import numpy as np
import requests
import json
import os
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / 'data' / 'raw'
PROC_DIR = Path(__file__).parent.parent / 'data' / 'processed'

# Biscayne Aquifer physical calibration constants (Prinos et al. 2014)
EC_FRESHWATER  = 500.0    # μS/cm — fresh groundwater baseline
EC_SALTWATER   = 45_000.0 # μS/cm — saltwater-intruded reference
CHLORIDE_THRESHOLD_MGL = 250.0   # mg/L — WHO drinking water limit
H_DEPTH_MAX    = 30.0     # feet — reference maximum water table depth (NAVD88)
H_CRIT_DEPTH   = 8.0      # feet below LS — critical intrusion depth (SFWMD)


# ─────────────────────────────────────────────────────────────────
# 1. SITE DISCOVERY
# ─────────────────────────────────────────────────────────────────

def discover_sites(state: str = 'FL',
                   county: str = '086',
                   start: str = '2000-01-01',
                   end: str = '2024-12-31') -> pd.DataFrame:
    """
    Discover USGS groundwater quality sites in Miami-Dade County with
    specific conductance (00095) or chloride (00940) records.
    """
    results = []
    for param, name in [('00095', 'spec_conductance'), ('00940', 'chloride')]:
        try:
            sites, _ = nwis.get_sites(
                stateCd=state,
                countyCd=county,
                parameterCd=param,
                siteType='GW',
                hasDataTypeCd='qw'
            )
            sites['param_available'] = name
            results.append(sites)
            print(f'  Found {len(sites)} sites with {name}')
        except Exception as e:
            print(f'  Site discovery for {name}: {e}')

    if not results:
        print('  No sites found — check network connection')
        return pd.DataFrame()

    all_sites = pd.concat(results, ignore_index=True)
    # Remove duplicate site numbers, keep first occurrence
    all_sites = all_sites.drop_duplicates(subset='site_no').reset_index(drop=True)
    print(f'  Total unique sites: {len(all_sites)}')
    return all_sites


# ─────────────────────────────────────────────────────────────────
# 2. DOWNLOAD SPECIFIC CONDUCTANCE (water quality)
# ─────────────────────────────────────────────────────────────────

def download_conductance(site_ids: list,
                          start: str = '2000-01-01',
                          end: str = '2024-12-31') -> pd.DataFrame:
    """
    Download discrete specific conductance samples (μS/cm) from NWIS.
    Returns long-format DataFrame with columns: [date, ec_uScm, site_no].
    """
    records = []
    for site in site_ids:
        try:
            df, _ = nwis.get_qwdata(
                sites=site,
                parameterCd='00095',
                start=start,
                end=end
            )
            if df is None or len(df) == 0:
                continue
            df = df.reset_index()
            # Column may be named 'p00095' or 'result_va' depending on version
            ec_col = next((c for c in df.columns if '00095' in c or c == 'result_va'), None)
            dt_col = next((c for c in df.columns if 'datetime' in c.lower() or c == 'sample_dt'), None)
            if ec_col and dt_col:
                sub = pd.DataFrame({
                    'date':    pd.to_datetime(df[dt_col], errors='coerce'),
                    'ec_uScm': pd.to_numeric(df[ec_col], errors='coerce'),
                    'site_no': site
                })
                sub = sub.dropna()
                if len(sub) > 0:
                    records.append(sub)
                    print(f'  {site}: {len(sub)} conductance records')
        except Exception as e:
            print(f'  {site} conductance: {e}')

    if not records:
        return pd.DataFrame(columns=['date', 'ec_uScm', 'site_no'])
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────
# 3. DOWNLOAD GROUNDWATER LEVEL
# ─────────────────────────────────────────────────────────────────

def download_gwlevel(site_ids: list,
                      start: str = '2000-01-01',
                      end: str = '2024-12-31') -> pd.DataFrame:
    """
    Download discrete groundwater levels (feet below land surface).
    Returns long-format DataFrame with columns: [date, lev_ft, site_no].
    """
    records = []
    for site in site_ids:
        try:
            df, _ = nwis.get_gwlevels(sites=site, start=start, end=end)
            if df is None or len(df) == 0:
                continue
            df = df.reset_index()
            dt_col  = next((c for c in df.columns if 'dt' in c.lower() or 'datetime' in c.lower()), None)
            lev_col = next((c for c in df.columns if 'lev_va' in c or 'lev_ft' in c), None)
            if dt_col and lev_col:
                sub = pd.DataFrame({
                    'date':   pd.to_datetime(df[dt_col], errors='coerce'),
                    'lev_ft': pd.to_numeric(df[lev_col], errors='coerce'),
                    'site_no': site
                })
                sub = sub.dropna()
                if len(sub) > 0:
                    records.append(sub)
                    print(f'  {site}: {len(sub)} water level records')
        except Exception as e:
            print(f'  {site} gwlevel: {e}')

    if not records:
        return pd.DataFrame(columns=['date', 'lev_ft', 'site_no'])
    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────
# 4. DOWNLOAD NOAA PRECIPITATION
# ─────────────────────────────────────────────────────────────────

def download_noaa_precip_direct(start_year: int = 2000,
                                 end_year: int   = 2024) -> pd.DataFrame:
    """
    Download monthly precipitation from NOAA NCEI Climate at a Glance
    for Florida Division 6 (South Florida) — no API key required.
    Falls back to GHCND Miami International Airport if division fails.
    """
    # NOAA Climate at a Glance: Florida Climate Division 6 (Southern)
    url = (
        'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
    )
    # Try direct bulk CSV download from Climate at a Glance
    cag_url = (
        f'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/'
        f'divisional/time-series/1206/pcp/1/{start_year}/{end_year}.csv'
    )
    try:
        resp = requests.get(cag_url, timeout=30)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            # Skip header comment lines starting with non-digit
            data_lines = [l for l in lines if l and l[0].isdigit()]
            rows = []
            for line in data_lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ym = str(parts[0]).strip()
                    val = parts[1].strip()
                    try:
                        year = int(ym[:4])
                        month = int(ym[4:])
                        rows.append({'date': pd.Timestamp(year=year, month=month, day=1),
                                     'precip_mm': float(val) * 25.4})  # inches → mm
                    except (ValueError, IndexError):
                        continue
            if rows:
                df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
                print(f'  NOAA Climate at a Glance: {len(df)} monthly precipitation records')
                return df
    except Exception as e:
        print(f'  NOAA Climate at a Glance failed: {e}')

    # Fallback: NOAA Open-Meteo historical (no key required)
    try:
        om_url = (
            'https://archive-api.open-meteo.com/v1/archive'
            '?latitude=25.79&longitude=-80.28'   # Miami
            '&start_date=2000-01-01&end_date=2024-12-31'
            '&monthly=precipitation_sum'
            '&timezone=America/New_York'
        )
        resp = requests.get(om_url, timeout=30)
        data = resp.json()
        dates = pd.to_datetime(data['monthly']['time'])
        precip = data['monthly']['precipitation_sum']
        df = pd.DataFrame({'date': dates, 'precip_mm': precip}).dropna()
        print(f'  Open-Meteo fallback: {len(df)} monthly precipitation records')
        return df
    except Exception as e:
        print(f'  Open-Meteo fallback failed: {e}')
        return pd.DataFrame(columns=['date', 'precip_mm'])


# ─────────────────────────────────────────────────────────────────
# 5. DOWNLOAD NOAA SEA LEVEL
# ─────────────────────────────────────────────────────────────────

def download_sealevel(station: str = '8723214',   # Virginia Key, Biscayne Bay
                       start: str = '20000101',
                       end: str   = '20241231') -> pd.DataFrame:
    """
    Download mean monthly sea level from NOAA Tides and Currents (no key required).
    Station 8723214 = Virginia Key, Biscayne Bay.
    Station 8724580 = Key West (longer record, fallback).
    """
    for sta in [station, '8724580']:
        try:
            url = (
                f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
                f'?begin_date={start}&end_date={end}'
                f'&station={sta}&product=monthly_mean'
                f'&datum=MSL&time_zone=LST&units=metric&format=json'
            )
            resp = requests.get(url, timeout=30)
            raw = resp.json()
            if 'data' not in raw:
                continue
            rows = []
            for rec in raw['data']:
                try:
                    rows.append({'date': pd.Timestamp(rec['t']),
                                 'slr_m': float(rec['MSL'])})
                except (KeyError, ValueError):
                    continue
            df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
            print(f'  Sea level station {sta}: {len(df)} monthly records')
            return df
        except Exception as e:
            print(f'  Sea level station {sta}: {e}')

    return pd.DataFrame(columns=['date', 'slr_m'])


# ─────────────────────────────────────────────────────────────────
# 6. PREPROCESSING UTILITIES
# ─────────────────────────────────────────────────────────────────

def aggregate_to_monthly(df: pd.DataFrame,
                          date_col: str,
                          value_col: str,
                          agg: str = 'mean') -> pd.Series:
    """Resample to month-start frequency; fill short gaps by linear interpolation."""
    s = df.set_index(pd.to_datetime(df[date_col]))[value_col]
    s = s.sort_index()
    s_monthly = s.resample('MS').agg(agg)
    s_monthly = s_monthly.interpolate(method='linear', limit=3)
    return s_monthly


def remove_outliers(s: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """Replace values beyond n_sigma with NaN, then interpolate."""
    mu, sigma = s.mean(), s.std()
    s_clean = s.where((s > mu - n_sigma * sigma) & (s < mu + n_sigma * sigma))
    return s_clean.interpolate(method='linear', limit=3)


def normalise_ec(ec_series: pd.Series) -> pd.Series:
    """Map specific conductance (μS/cm) to normalised salinity proxy in [0, 1]."""
    clipped = ec_series.clip(EC_FRESHWATER, EC_SALTWATER)
    return (clipped - EC_FRESHWATER) / (EC_SALTWATER - EC_FRESHWATER)


def chloride_to_ec(cl_mgl: pd.Series) -> pd.Series:
    """Convert chloride (mg/L) to approximate specific conductance (μS/cm)."""
    return cl_mgl * 1.8 + 400.0


def normalise_gwlevel(lev_ft: pd.Series,
                       h_depth_max: float = H_DEPTH_MAX) -> pd.Series:
    """
    Map water table depth (ft below land surface) to normalised freshwater
    volume h ∈ [0, 1].  Deeper table = lower h.
    """
    return (1.0 - (lev_ft / h_depth_max)).clip(0.0, 1.0)


def estimate_sigma_from_pairs(series_a: pd.Series,
                               series_b: pd.Series) -> float:
    """
    Estimate observation noise σ from two co-located monitoring wells.
    Both series must share a monthly DatetimeIndex.
    """
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 12:
        print(f'  Warning: only {len(aligned)} paired obs — using fallback σ=0.08')
        return 0.08
    residuals = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    sigma = float(residuals.std() / np.sqrt(2))
    print(f'  σ estimate: {sigma:.4f}  (n={len(aligned)} paired months)')
    return sigma


def select_best_site(ec_wide: pd.DataFrame, min_years: int = 8) -> str:
    """
    From a wide-format EC DataFrame (columns = site_no, index = monthly date),
    return the site_no with the longest continuous non-NaN record.
    """
    counts = ec_wide.notna().sum()
    threshold = min_years * 12
    candidates = counts[counts >= threshold]
    if candidates.empty:
        best = counts.idxmax()
        print(f'  No site meets {min_years}-year minimum; using best available: {best}')
    else:
        best = candidates.idxmax()
    print(f'  Selected primary site: {best} ({counts[best]} monthly observations)')
    return best


# ─────────────────────────────────────────────────────────────────
# 7. MAIN DOWNLOAD PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_download_pipeline(start: str = '2000-01-01',
                           end: str   = '2024-12-31') -> dict:
    """
    Full acquisition pipeline.  Downloads all raw data and writes to
    biscayne/data/raw/.  Returns dict of raw DataFrames.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print('\n── Step 1: Site Discovery ─────────────────────────────')
    sites = discover_sites(start=start, end=end)
    if sites.empty:
        print('  ERROR: No sites returned. Check network.')
        return {}
    sites.to_csv(RAW_DIR / 'sites_discovered.csv', index=False)
    site_ids = sites['site_no'].astype(str).tolist()[:30]  # cap at 30 for speed
    print(f'  Using up to 30 sites for initial download.')

    print('\n── Step 2: Download Specific Conductance ──────────────')
    ec_raw = download_conductance(site_ids, start=start, end=end)
    if not ec_raw.empty:
        ec_raw.to_csv(RAW_DIR / 'usgs_conductance.csv', index=False)
        print(f'  Saved {len(ec_raw)} rows → usgs_conductance.csv')

    print('\n── Step 3: Download Groundwater Levels ─────────────────')
    gw_raw = download_gwlevel(site_ids, start=start, end=end)
    if not gw_raw.empty:
        gw_raw.to_csv(RAW_DIR / 'usgs_gwlevel.csv', index=False)
        print(f'  Saved {len(gw_raw)} rows → usgs_gwlevel.csv')

    print('\n── Step 4: Download NOAA Precipitation ─────────────────')
    precip = download_noaa_precip_direct(start_year=int(start[:4]),
                                         end_year=int(end[:4]))
    if not precip.empty:
        precip.to_csv(RAW_DIR / 'noaa_precip.csv', index=False)
        print(f'  Saved {len(precip)} rows → noaa_precip.csv')

    print('\n── Step 5: Download NOAA Sea Level ──────────────────────')
    slr = download_sealevel(start=start.replace('-', ''), end=end.replace('-', ''))
    if not slr.empty:
        slr.to_csv(RAW_DIR / 'noaa_sealevel.csv', index=False)
        print(f'  Saved {len(slr)} rows → noaa_sealevel.csv')

    return {'sites': sites, 'ec': ec_raw, 'gwlevel': gw_raw,
            'precip': precip, 'slr': slr}


def run_preprocessing_pipeline(raw: dict) -> dict:
    """
    Clean, align, and normalise raw data.  Writes processed files to
    biscayne/data/processed/.  Returns dict of processed series.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    # ── Precipitation ───────────────────────────────────────────
    if 'precip' in raw and not raw['precip'].empty:
        p = aggregate_to_monthly(raw['precip'], 'date', 'precip_mm')
        p = remove_outliers(p)
        out['precip'] = p
        p.to_csv(PROC_DIR / 'precip_monthly.csv', header=['precip_mm'])
        print(f'  Precipitation: {len(p)} monthly values saved')

    # ── Conductance → wide matrix → select best site ────────────
    if 'ec' in raw and not raw['ec'].empty:
        ec_long = raw['ec'].copy()
        ec_long['date'] = pd.to_datetime(ec_long['date'])
        # Wide pivot: rows=month, columns=site_no
        ec_monthly_long = (ec_long.groupby(['site_no', pd.Grouper(key='date', freq='MS')])
                                   ['ec_uScm'].mean().reset_index())
        ec_wide = ec_monthly_long.pivot(index='date', columns='site_no', values='ec_uScm')
        ec_wide = ec_wide.interpolate(method='linear', axis=0, limit=3)
        ec_wide.to_csv(RAW_DIR / 'usgs_conductance_monthly_wide.csv')

        best_site = select_best_site(ec_wide)
        ec_best   = remove_outliers(ec_wide[best_site].dropna())
        obs_norm  = normalise_ec(ec_best)

        out['ec_best_site'] = best_site
        out['ec_raw_best']  = ec_best
        out['obs_norm']     = obs_norm
        obs_df = obs_norm.reset_index()
        obs_df.columns = ['date', 'obs_norm']
        obs_df.to_csv(PROC_DIR / 'obs_sequence.csv', index=False)
        print(f'  Observation sequence: {len(obs_norm)} monthly values saved')
        print(f'  obs range: [{obs_norm.min():.3f}, {obs_norm.max():.3f}]')

        # Sigma estimate from second-best site if available
        sorted_counts = ec_wide.notna().sum().sort_values(ascending=False)
        if len(sorted_counts) >= 2:
            site_b = sorted_counts.index[1]
            ec_b   = remove_outliers(ec_wide[site_b].dropna())
            norm_a = normalise_ec(ec_best).rename('a')
            norm_b = normalise_ec(ec_b).rename('b')
            out['sigma'] = estimate_sigma_from_pairs(norm_a, norm_b)
        else:
            out['sigma'] = 0.08

    # ── Groundwater level ────────────────────────────────────────
    if 'gwlevel' in raw and not raw['gwlevel'].empty:
        best_site = out.get('ec_best_site')
        gw = raw['gwlevel'].copy()
        gw['date'] = pd.to_datetime(gw['date'])
        if best_site and best_site in gw['site_no'].values:
            gw_site = gw[gw['site_no'] == best_site]
        else:
            # Use site with most records
            best_gw = gw.groupby('site_no').size().idxmax()
            gw_site = gw[gw['site_no'] == best_gw]
            print(f'  GW level using site {best_gw} (EC site not available)')
        gw_monthly = aggregate_to_monthly(gw_site, 'date', 'lev_ft')
        gw_monthly = remove_outliers(gw_monthly)
        h_norm     = normalise_gwlevel(gw_monthly)
        out['h_norm'] = h_norm
        h_df = h_norm.reset_index()
        h_df.columns = ['date', 'h_norm']
        h_df.to_csv(PROC_DIR / 'h_sequence.csv', index=False)
        print(f'  Freshwater level: {len(h_norm)} monthly values saved')

    return out


if __name__ == '__main__':
    print('═' * 55)
    print('  Module M1: Biscayne Data Loader')
    print('═' * 55)
    raw = run_download_pipeline()
    if raw:
        processed = run_preprocessing_pipeline(raw)
        print('\nProcessed outputs:')
        for k, v in processed.items():
            if hasattr(v, '__len__'):
                print(f'  {k}: {len(v)} items')
            else:
                print(f'  {k}: {v}')
    print('\nDone.')

"""
5_sensitivity.py

Parameter sensitivity analysis for the CMVO strategy.
Sweeps one parameter at a time (holding others at baseline) and reports
OOS metrics on a date-aligned common period.

Parameter grids:
  Lookback (years)   : 2, 3, 4, 5
  Cluster count k    : 4, 5, 6, 7, 8, 9
  Cluster cap (%)    : 20, 25, 30, 35, 40, 50, 100 (unconstrained)
  Stock cap (%)      : 10, 15, 20, 25, 30, 50, 100 (unconstrained)
  Covariance method  : sample, ledoit_wolf, oracle_approximating

Outputs:
  5 printed tables (one per parameter sweep)
  data/5_sensitivity_*.csv for each sweep
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

from config import BASELINE_CONFIG, TICKERS
from engine import run_rolling_backtest
from metrics import compute_metrics, compute_turnover, compute_es


# ═══════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════

def compute_row(label, res, weekly_returns=None):
    """Compute the sensitivity metrics for one backtest run."""
    ret = res["portfolio_returns"]
    if len(ret) < 10:
        return None
    m = compute_metrics(ret, label=label)
    to_series = compute_turnover(res.get("weights", {}), tickers=TICKERS, returns=weekly_returns)
    avg_to = float(to_series.mean() * 100) if len(to_series) > 0 else 0.0

    # HHI (avg sum of squared weights)
    ws = res.get("weights", {})
    if ws:
        hhis = [sum(v ** 2 for v in ws[d].values()) for d in sorted(ws.keys())]
        hhi = round(float(np.mean(hhis)), 4)
    else:
        hhi = np.nan

    return {
        "Config":              label,
        "Ann. Sharpe":         m.get("Sharpe", np.nan),
        "OOS Std %":           round(ret.std() * 100, 3),
        "ES (5%) %":           round(compute_es(ret) * 100, 3),
        "Median Wkly Ret %":   round(ret.median() * 100, 3),
        "Kurtosis":            round(float(ret.kurtosis()), 2),
        "HHI":                 hhi,
        "Avg Turnover %":      round(avg_to, 1),
    }


# ═══════════════════════════════════════════════════════════
#  SWEEP RUNNER
# ═══════════════════════════════════════════════════════════

_GLOBAL_DONE = 0
_GLOBAL_TOTAL = 0


def run_sweep(param_name, values, make_kwargs_fn, align_date, start_str, weekly_returns=None):
    """Run a parameter sweep and return a DataFrame of metrics.

    values can be plain values or (display_label, actual_value) tuples.
    """
    global _GLOBAL_DONE
    rows = []
    for i, val in enumerate(values):
        if isinstance(val, tuple):
            display, actual = val
            label = f"{param_name}={display}"
        else:
            label = f"{param_name}={val}"
        _GLOBAL_DONE += 1
        print(f"    [{_GLOBAL_DONE}/{_GLOBAL_TOTAL}] {label} ...", end="", flush=True)
        kwargs = make_kwargs_fn(val)
        try:
            res = run_rolling_backtest(start_str, **kwargs)
            # Align to common date
            ret = res["portfolio_returns"]
            res["portfolio_returns"] = ret.loc[align_date:]
            row = compute_row(label, res, weekly_returns=weekly_returns)
            if row:
                rows.append(row)
                print(f"  Sharpe={row['Ann. Sharpe']:.2f}")
            else:
                print("  (insufficient data)")
        except Exception as e:
            print(f"  FAILED: {e}")
    return pd.DataFrame(rows)


def print_table(title, df):
    """Pretty-print a sensitivity table."""
    print(f"\n  {title}")
    print("  " + "=" * 115)
    print(f"  {'Config':<22} {'AnnSharpe':>10} {'OOSStd%':>9} {'ES5%%':>9} {'MedRet%':>9}"
          f" {'Kurt':>7} {'HHI':>7} {'TO%':>7}")
    print("  " + "-" * 115)
    for _, r in df.iterrows():
        print(f"  {r['Config']:<22} {r['Ann. Sharpe']:>9.2f} {r['OOS Std %']:>9.3f}"
              f" {r['ES (5%) %']:>9.3f} {r['Median Wkly Ret %']:>9.3f}"
              f" {r['Kurtosis']:>7.2f} {r['HHI']:>7.4f} {r['Avg Turnover %']:>7.1f}")
    print("  " + "=" * 115)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Sensitivity Analysis")
    print("=" * 60)

    from data_loader import load_data
    daily_close, weekly_close, _ = load_data()
    start_str = str(daily_close.index.min().date())
    bc = BASELINE_CONFIG

    # ── Compute alignment date ──
    # Use the actual first OOS date from the longest-lookback baseline run so the
    # sensitivity sweeps match the main results exactly.
    weekly_returns = weekly_close.pct_change().dropna()
    reference = run_rolling_backtest(
        start_str,
        lookback_yrs=bc["lookback"],
        rebal_freq=bc["rebal"],
        n_clusters=bc["k"],
        linkage_method=bc["linkage"],
        opt_target=bc["target"],
        cov_method=bc["cov"],
        cluster_cap=bc["cluster_cap"],
        cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"],
        rf_annual=bc["rf_annual"],
    )
    align_date = reference["portfolio_returns"].index.min()

    print(f"\n  Date alignment: all runs trimmed to start at {align_date.strftime('%Y-%m-%d')}")
    print(f"  (= first realized OOS week from the 5-year baseline configuration)")
    print(f"  This ensures every config is evaluated on the same {len(weekly_returns.loc[align_date:])} weekly periods.")

    global _GLOBAL_DONE, _GLOBAL_TOTAL
    _GLOBAL_DONE = 0
    _GLOBAL_TOTAL = 4 + 6 + 7 + 7 + 3  # lookback + k + cluster_cap + stock_cap + cov
    print(f"\n  Total configurations to run: {_GLOBAL_TOTAL}")

    all_tables = {}

    # ── 1. Lookback sweep ──
    print("\n[1/5] Lookback (years): 2, 3, 4, 5")
    lookbacks = [2, 3, 4, 5]
    df = run_sweep("Lookback", lookbacks, lambda v: dict(
        lookback_yrs=v, rebal_freq=bc["rebal"], n_clusters=bc["k"],
        linkage_method=bc["linkage"], opt_target=bc["target"], cov_method=bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    ), align_date, start_str, weekly_returns=weekly_returns)
    print_table("Lookback Sensitivity", df)
    all_tables["lookback"] = df

    # ── 2. Cluster count k sweep ──
    print("\n[2/5] Cluster count k: 4, 5, 6, 7, 8, 9")
    ks = [4, 5, 6, 7, 8, 9]
    df = run_sweep("k", ks, lambda v: dict(
        lookback_yrs=bc["lookback"], rebal_freq=bc["rebal"], n_clusters=v,
        linkage_method=bc["linkage"], opt_target=bc["target"], cov_method=bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    ), align_date, start_str, weekly_returns=weekly_returns)
    print_table("Cluster Count (k) Sensitivity", df)
    all_tables["k"] = df

    # ── 3. Cluster cap sweep ──
    print("\n[3/5] Cluster cap: 20%, 25%, 30%, 35%, 40%, 50%, 100%")
    caps = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.00]
    cap_labels = ["20%", "25%", "30%", "35%", "40%", "50%", "100%"]
    df = run_sweep("ClusterCap", list(zip(cap_labels, caps)), lambda v: dict(
        lookback_yrs=bc["lookback"], rebal_freq=bc["rebal"], n_clusters=bc["k"],
        linkage_method=bc["linkage"], opt_target=bc["target"], cov_method=bc["cov"],
        cluster_cap=v[1], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    ), align_date, start_str, weekly_returns=weekly_returns)
    print_table("Cluster Cap Sensitivity", df)
    all_tables["cluster_cap"] = df

    # ── 4. Stock cap sweep ──
    print("\n[4/5] Stock cap: 10%, 15%, 20%, 25%, 30%, 50%, 100%")
    scaps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
    scap_labels = ["10%", "15%", "20%", "25%", "30%", "50%", "100%"]
    df = run_sweep("StockCap", list(zip(scap_labels, scaps)), lambda v: dict(
        lookback_yrs=bc["lookback"], rebal_freq=bc["rebal"], n_clusters=bc["k"],
        linkage_method=bc["linkage"], opt_target=bc["target"], cov_method=bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=v[1], rf_annual=bc["rf_annual"],
    ), align_date, start_str, weekly_returns=weekly_returns)
    print_table("Stock Cap Sensitivity", df)
    all_tables["stock_cap"] = df

    # ── 5. Covariance method sweep ──
    print("\n[5/5] Covariance: sample, ledoit_wolf, oracle_approximating")
    cov_methods = [("Sample", "sample"), ("Ledoit-Wolf", "ledoit_wolf"),
                   ("Oracle Approx.", "oracle_approximating")]
    df = run_sweep("Cov", cov_methods, lambda v: dict(
        lookback_yrs=bc["lookback"], rebal_freq=bc["rebal"], n_clusters=bc["k"],
        linkage_method=bc["linkage"], opt_target=bc["target"], cov_method=v[1],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    ), align_date, start_str, weekly_returns=weekly_returns)
    print_table("Covariance Estimator Sensitivity", df)
    all_tables["cov"] = df

    # ── Save CSVs ──
    print("\n" + "=" * 60)
    print("  Saving outputs")
    print("=" * 60)
    for name, tbl in all_tables.items():
        path = os.path.join(DATA_DIR, f"5_sensitivity_{name}.csv")
        tbl.to_csv(path, index=False)
        print(f"  -> data/5_sensitivity_{name}.csv")

    print("\n" + "=" * 60)
    print("  Done. 5 sensitivity tables saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()

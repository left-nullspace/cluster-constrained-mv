"""
7_nco.py

Nested Clustered Optimization (Lopez de Prado, 2020 Ch.7.6).
Walk-forward NCO backtest compared against CMVO, Plain MV, and Equal Weight (1/N).

Outputs:
  Table — NCO vs CMVO vs Plain MV vs Equal Weight (1/N) performance comparison
  data/7_nco_performance.csv
"""

import os, importlib.util
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

from config import TICKERS, BASELINE_CONFIG
from data_loader import load_data
from engine import get_rebalance_dates, simulate_portfolio, _required_lookback_obs
from metrics import compute_metrics, compute_turnover, compute_es


def run_nco_backtest(start_str, lookback_yrs, rebal_freq, rf_annual=0.0):
    """Walk-forward NCO backtest using Riskfolio-Lib via VBT Pro."""
    import vectorbtpro as vbt

    _, weekly_close, _ = load_data()
    close = weekly_close[TICKERS].loc[pd.Timestamp(start_str):]
    ret = close.pct_change().dropna()

    rebal_dates = get_rebalance_dates(ret.index, rebal_freq)
    lookback_td = pd.DateOffset(years=lookback_yrs)
    min_obs = _required_lookback_obs(lookback_yrs, 52)

    weights_schedule = {}
    n_failed = 0

    for i, rd in enumerate(rebal_dates):
        window_start = rd - lookback_td
        wr = ret.loc[window_start:rd]
        if len(wr) < min_obs:
            continue
        try:
            w = vbt.riskfolio_optimize(
                wr,
                port_cls="HCPortfolio",
                model="NCO",
                codependence="pearson",
                method_cov="hist",
                obj="Sharpe",
                rm="MV",
                rf=rf_annual,
                l=2,
                linkage="ward",
                max_k=10,
                leaf_order=True,
            )
            weights_schedule[rd] = w
        except Exception:
            n_failed += 1
            weights_schedule[rd] = {t: 1 / len(TICKERS) for t in TICKERS}

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(rebal_dates)}]")

    pf_returns = simulate_portfolio(ret, weights_schedule, TICKERS)
    print(f"    {len(weights_schedule)} splits, {n_failed} failed")
    return {"portfolio_returns": pf_returns, "weights": weights_schedule}


def _load_main_results_module():
    """Load submission/4_results.py so this script can reuse Table 5 baselines exactly."""
    path = os.path.join(BASE, "4_results.py")
    spec = importlib.util.spec_from_file_location("submission_4_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 60)
    print("  NCO Comparison (Lopez de Prado, 2020 Ch.7.6)")
    print("=" * 60)

    try:
        import vectorbtpro as vbt
    except ImportError:
        print("\n  [SKIP] vectorbtpro not installed.")
        print("  Skipped: NCO backtest comparison, data/7_nco_performance.csv")
        return

    daily_close, weekly_close, _ = load_data()
    start_str = str(daily_close.index.min().date())
    bc = BASELINE_CONFIG
    weekly_returns = weekly_close[TICKERS].pct_change().dropna()

    print("\n  [1/2] Loading Table 5 baseline results ...")
    results4 = _load_main_results_module()
    main_results, common_start, weekly_returns = results4.run_all_backtests()
    baseline_perf = results4.performance_table(
        main_results, results4.STRATEGY_ORDER, weekly_returns=weekly_returns
    ).set_index("Strategy")

    print("  [2/2] NCO ...")
    nco = run_nco_backtest(start_str, bc["lookback"], bc["rebal"], rf_annual=bc["rf_annual"])

    nco_ret = nco["portfolio_returns"]
    nco_ret = nco_ret[nco_ret.index >= common_start]
    nco_weights = {d: w for d, w in nco["weights"].items() if d >= common_start}
    print(
        f"\n  Aligned to Table 5 common window: {common_start.strftime('%Y-%m-%d')} "
        f"to {nco_ret.index.max().strftime('%Y-%m-%d')} ({len(nco_ret)} weeks)"
    )

    # Table
    strategy_order = ["CMVO", "Plain MV", "NCO", "Equal Weight (1/N)"]
    rows = []

    print(f"\n  {'Strategy':<12} {'AnnRet%':>8} {'AnnVol%':>8} {'Sharpe':>8} "
          f"{'MaxDD%':>8} {'ES5%':>7} {'HHI':>7} {'TO%':>6}")
    print("  " + "-" * 75)

    baseline_name_map = {
        "CMVO": "CMVO",
        "Plain MV": "Plain MV",
        "Equal Weight (1/N)": "Equal Weight (1/N)",
    }

    for name in strategy_order:
        if name in baseline_name_map:
            src = baseline_perf.loc[baseline_name_map[name]]
            ann_ret = float(src["Ann. Return %"])
            ann_vol = float(src["Ann. Volatility %"])
            sharpe = float(src["Ann. Sharpe"])
            max_dd = float(src["Max DD %"])
            es = float(src["ES (5%) %"])
            hhi = float(src["HHI"])
            avg_to = float(src["Avg Turnover %"])
        else:
            m = compute_metrics(nco_ret, label=name)
            hhis = [sum(v ** 2 for v in nco_weights[d].values()) for d in sorted(nco_weights.keys())]
            hhi = np.mean(hhis)
            to = compute_turnover(nco_weights, tickers=TICKERS, returns=weekly_returns)
            avg_to = to.mean() * 100 if len(to) > 0 else 0
            es = compute_es(nco_ret) * 100
            ann_ret = float(m["Ann. Return %"])
            ann_vol = float(m["Ann. Volatility %"])
            sharpe = float(m["Sharpe"])
            max_dd = float(m["Max DD %"])

        print(f"  {name:<12} {ann_ret:>7.1f}  {ann_vol:>7.1f}  "
              f"{sharpe:>7.2f}  {max_dd:>7.1f}  {es:>6.2f}  {hhi:>6.4f}  {avg_to:>5.1f}")

        rows.append({
            "Strategy": name,
            "Ann. Return %": round(ann_ret, 1),
            "Ann. Volatility %": round(ann_vol, 1),
            "Ann. Sharpe": round(sharpe, 2),
            "Max DD %": round(max_dd, 1),
            "ES (5%) %": round(es, 2),
            "HHI": round(hhi, 4),
            "Avg Turnover %": round(avg_to, 1),
            "OOS Start": common_start.strftime("%Y-%m-%d"),
            "OOS End": nco_ret.index.max().strftime("%Y-%m-%d"),
            "N Weeks": len(nco_ret),
        })

    print("  " + "=" * 75)

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "7_nco_performance.csv"), index=False)
    print("  -> data/7_nco_performance.csv")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

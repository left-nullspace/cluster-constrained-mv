"""
Performance metrics and distributional statistics.
"""

import numpy as np
import pandas as pd
from config import RF_ANNUAL, TICKERS


def rf_periodic(rf_annual=RF_ANNUAL, periods_per_year=52):
    """Risk-free rate per period for any frequency."""
    return (1 + rf_annual) ** (1 / periods_per_year) - 1


def downside_deviation(excess_returns, periods_per_year=52):
    """Proper downside deviation: sqrt(mean(min(r,0)^2)) * sqrt(periods_per_year)."""
    downside = np.minimum(excess_returns.values, 0)
    dd = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
    return dd


def compute_metrics(returns, label="Strategy", rf_annual=RF_ANNUAL, periods_per_year=52):
    """Compute standard performance metrics from a return series."""
    ret = returns[returns.index >= returns[returns != 0].index[0]] if (returns != 0).any() else returns
    n = len(ret)
    if n < 2:
        return {}
    rf_p = rf_periodic(rf_annual, periods_per_year)
    excess = ret - rf_p
    total = (1 + ret).prod() - 1
    ann_ret = (1 + total) ** (periods_per_year / max(n, 1)) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (excess.mean() * periods_per_year) / (excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) > 0 else 0
    dd_ann = downside_deviation(excess, periods_per_year)
    sortino = (excess.mean() * periods_per_year) / dd_ann if dd_ann > 0 else 0
    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    hit_rate = (ret > 0).mean()
    return {
        "Strategy": label,
        "Total Return %": round(total * 100, 1),
        "Ann. Return %": round(ann_ret * 100, 1),
        "Ann. Volatility %": round(ann_vol * 100, 1),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Max DD %": round(max_dd * 100, 1),
        "Hit Rate %": round(hit_rate * 100, 1),
    }


def compute_turnover(weights_schedule, tickers=TICKERS, returns=None):
    """Compute turnover at each rebalance (DeMiguel et al. 2009, Eq.15).

    Turnover_t = sum_j |w_new,j - w_drifted,j|
    where w_drifted are the weights after one period of price drift,
    before rebalancing to w_new.

    If returns is None, falls back to target-vs-target (legacy behavior).
    """
    dates = sorted(weights_schedule.keys())
    turnovers = {}
    for i in range(1, len(dates)):
        prev_target = pd.Series(weights_schedule[dates[i - 1]]).reindex(tickers).fillna(0)
        curr_target = pd.Series(weights_schedule[dates[i]]).reindex(tickers).fillna(0)

        if returns is not None:
            ret_slice = returns.loc[dates[i - 1]:dates[i]]
            if len(ret_slice) > 1:
                period_ret = ret_slice.iloc[1:][tickers]
                if len(period_ret) > 0:
                    cum_ret = (1 + period_ret).prod()
                    drifted = prev_target * cum_ret
                    drifted = drifted / drifted.sum()
                else:
                    drifted = prev_target
            else:
                drifted = prev_target
        else:
            drifted = prev_target

        turnovers[dates[i]] = (curr_target - drifted).abs().sum()
    return pd.Series(turnovers)


def compute_es(returns, alpha=0.05):
    """Expected Shortfall (CVaR) at alpha -- mean of returns <= VaR(alpha)."""
    var = returns.quantile(alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def oos_return_distribution_stats(oos_returns_series, periods_per_year=52):
    """Compute distributional statistics for OOS weekly returns."""
    s = oos_returns_series.dropna()
    if len(s) < 2:
        return {}
    var_5 = float(s.quantile(0.05))
    cvar_5 = float(s[s <= var_5].mean()) if (s <= var_5).any() else var_5
    ann_sharpe = (s.mean() * periods_per_year) / (s.std(ddof=1) * np.sqrt(periods_per_year)) if s.std(ddof=1) > 0 else 0
    return {
        "N obs": len(s),
        "Mean %": round(s.mean() * 100, 3),
        "Median %": round(s.median() * 100, 3),
        "Std %": round(s.std() * 100, 3),
        "Skew": round(s.skew(), 3),
        "Kurtosis": round(s.kurtosis() + 3, 3),
        "P5 %": round(s.quantile(0.05) * 100, 3),
        "P10 %": round(s.quantile(0.10) * 100, 3),
        "P25 %": round(s.quantile(0.25) * 100, 3),
        "P50 %": round(s.quantile(0.50) * 100, 3),
        "P75 %": round(s.quantile(0.75) * 100, 3),
        "P90 %": round(s.quantile(0.90) * 100, 3),
        "P95 %": round(s.quantile(0.95) * 100, 3),
        "Hit Rate %": round((s > 0).mean() * 100, 1),
        "VaR(5%) %": round(var_5 * 100, 3),
        "ES(5%) %": round(cvar_5 * 100, 3),
        "Ann. Sharpe": round(ann_sharpe, 3),
        "Min %": round(s.min() * 100, 3),
        "Max %": round(s.max() * 100, 3),
    }


def trim_all_to_common_start(series_dict):
    """Trim multiple return series to their common first-nonzero date."""
    first_dates = []
    for s in series_dict.values():
        nonzero = s[s != 0]
        if len(nonzero) > 0:
            first_dates.append(nonzero.index[0])
    if not first_dates:
        return series_dict, None
    common = max(first_dates)
    return {k: v.loc[common:] for k, v in series_dict.items()}, common

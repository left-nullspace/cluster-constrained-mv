"""
Core algorithms: clustering, optimization, simulation, rolling backtests.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from pypfopt import expected_returns, risk_models, EfficientFrontier, HRPOpt
from pypfopt.risk_models import CovarianceShrinkage

from config import TICKERS, FIXED_CLUSTERS, RF_ANNUAL

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════════════════════════

def cluster_from_returns(returns_window, n_clusters, linkage_method, tickers):
    """Compute hierarchical clusters from a return window."""
    corr = returns_window[tickers].corr()
    dist = np.sqrt(0.5 * (1 - corr.values))  # Lopez de Prado (2016) correlation distance
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_map = {t: f"C{int(l)}" for t, l in zip(tickers, labels)}
    return cluster_map, corr, Z


def align_cluster_labels(prev_map, curr_map, tickers):
    """Align cluster labels between consecutive windows using the Hungarian algorithm."""
    prev_labels = sorted(set(prev_map.values()))
    curr_labels = sorted(set(curr_map.values()))
    n_prev = len(prev_labels)
    n_curr = len(curr_labels)
    n = max(n_prev, n_curr)

    cost = np.zeros((n, n))
    for i, cl in enumerate(curr_labels):
        curr_members = {t for t in tickers if curr_map.get(t) == cl}
        for j, pl in enumerate(prev_labels):
            prev_members = {t for t in tickers if prev_map.get(t) == pl}
            cost[i, j] = -len(curr_members & prev_members)

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_curr and c < n_prev:
            mapping[curr_labels[r]] = prev_labels[c]

    used = set(mapping.values())
    next_id = n + 1
    for cl in curr_labels:
        if cl not in mapping:
            while f"C{next_id}" in used:
                next_id += 1
            mapping[cl] = f"C{next_id}"
            used.add(f"C{next_id}")

    aligned = {t: mapping.get(l, l) for t, l in curr_map.items()}
    return aligned


# ═══════════════════════════════════════════════════════════════
# Optimization
# ═══════════════════════════════════════════════════════════════

def optimize_single_period(prices_window, returns_window, target, cov_method,
                           sector_mapper, sector_upper, sector_lower, tickers,
                           stock_cap=1.0, linkage_matrix=None, frequency=52):
    """Run one optimization. Returns {ticker: weight} dict. Falls back to EW on failure."""
    n = len(tickers)
    ew = {t: 1.0 / n for t in tickers}

    try:
        mu = expected_returns.mean_historical_return(prices_window[tickers], frequency=frequency)

        if target in ("max_sharpe", "min_volatility", "max_quadratic_utility"):
            if cov_method == "ledoit_wolf":
                S = CovarianceShrinkage(prices_window[tickers], frequency=frequency).ledoit_wolf()
            elif cov_method == "oracle_approximating":
                S = CovarianceShrinkage(prices_window[tickers], frequency=frequency).oracle_approximating()
            else:
                S = risk_models.sample_cov(prices_window[tickers], frequency=frequency)

            ef = EfficientFrontier(mu, S, weight_bounds=(0, min(stock_cap, 1.0)))
            if sector_mapper and sector_upper:
                _lower = sector_lower if sector_lower else {k: 0 for k in sector_upper}
                ef.add_sector_constraints(sector_mapper, _lower, sector_upper)

            if target == "max_sharpe":
                ef.max_sharpe()
            elif target == "max_quadratic_utility":
                ef.max_quadratic_utility()
            else:
                ef.min_volatility()

            return ef.clean_weights()

        elif target == "hrp":
            import collections
            hrp = HRPOpt(returns_window[tickers])
            if linkage_matrix is not None:
                corr, cov = returns_window[tickers].corr(), returns_window[tickers].cov()
                hrp.clusters = linkage_matrix
                sort_ix = HRPOpt._get_quasi_diag(linkage_matrix)
                ordered_tickers = corr.index[sort_ix].tolist()
                raw = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
                weights = collections.OrderedDict(raw.sort_index())
                hrp.set_weights(weights)
                return weights
            return hrp.optimize()

        elif target == "min_cvar":
            from pypfopt import EfficientCVaR
            ef = EfficientCVaR(mu, returns_window[tickers], weight_bounds=(0, min(stock_cap, 1.0)))
            if sector_mapper and sector_upper:
                _lower = sector_lower if sector_lower else {k: 0 for k in sector_upper}
                ef.add_sector_constraints(sector_mapper, _lower, sector_upper)
            ef.min_cvar()
            return ef.clean_weights()

        elif target == "min_cdar":
            from pypfopt import EfficientCDaR
            ef = EfficientCDaR(mu, returns_window[tickers], weight_bounds=(0, min(stock_cap, 1.0)))
            if sector_mapper and sector_upper:
                _lower = sector_lower if sector_lower else {k: 0 for k in sector_upper}
                ef.add_sector_constraints(sector_mapper, _lower, sector_upper)
            ef.min_cdar()
            return ef.clean_weights()

        elif target == "ecw":
            if sector_mapper:
                clusters = {}
                for t in tickers:
                    c = sector_mapper.get(t, "C1")
                    clusters.setdefault(c, []).append(t)
                k = len(clusters)
                w = {}
                for members in clusters.values():
                    member_w = 1.0 / (k * len(members))
                    for t in members:
                        w[t] = member_w
                return w
            return ew

    except Exception:
        return ew

    return ew


# ═══════════════════════════════════════════════════════════════
# Portfolio Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_portfolio(weekly_returns, weights_schedule, tickers):
    """Simulate a portfolio with weight drift between rebalances."""
    dates = weekly_returns.index
    rebal_dates = sorted(weights_schedule.keys())

    portfolio_returns = pd.Series(0.0, index=dates, dtype=float)
    current_weights = pd.Series(0.0, index=tickers, dtype=float)
    pending_weights = None
    active = False
    rebal_idx = 0

    for date in dates:
        if pending_weights is not None:
            current_weights = pending_weights
            pending_weights = None
            active = True

        if rebal_idx < len(rebal_dates) and date >= rebal_dates[rebal_idx]:
            w = weights_schedule[rebal_dates[rebal_idx]]
            pending_weights = pd.Series(w, dtype=float).reindex(tickers).fillna(0)
            rebal_idx += 1

        if not active:
            continue

        stock_rets = weekly_returns.loc[date, tickers]
        pf_ret = (current_weights * stock_rets).sum()
        portfolio_returns.loc[date] = pf_ret

        denom = 1 + pf_ret
        if abs(denom) > 1e-10:
            current_weights = current_weights * (1 + stock_rets) / denom

    nonzero = portfolio_returns[portfolio_returns != 0]
    if len(nonzero) > 0:
        portfolio_returns = portfolio_returns.loc[nonzero.index[0]:]
    else:
        portfolio_returns = portfolio_returns.iloc[0:0]

    return portfolio_returns


# ═══════════════════════════════════════════════════════════════
# Rebalance Date Generation
# ═══════════════════════════════════════════════════════════════

def get_rebalance_dates(weekly_index, rebal_freq):
    """Identify rebalance dates from a DatetimeIndex."""
    s = weekly_index.to_series()
    if rebal_freq in ("Weekly", "Every"):
        mask = pd.Series(True, index=s.index)
        mask.iloc[0] = False
    elif rebal_freq == "Monthly":
        mask = s.dt.to_period("M") != s.shift(1).dt.to_period("M")
    elif rebal_freq == "Quarterly":
        mask = s.dt.to_period("Q") != s.shift(1).dt.to_period("Q")
    else:  # Yearly
        mask = s.dt.to_period("Y") != s.shift(1).dt.to_period("Y")
    mask.iloc[0] = False
    return weekly_index[mask]


# ═══════════════════════════════════════════════════════════════
# Rolling Backtest Engine
# ═══════════════════════════════════════════════════════════════

def _required_lookback_obs(lookback_yrs, ann_factor):
    """Minimum number of observations required to honor the full lookback window."""
    return max(ann_factor, int(np.ceil(lookback_yrs * ann_factor)))


def run_rolling_backtest(start_date_str, lookback_yrs, rebal_freq, n_clusters,
                         linkage_method, opt_target, cov_method,
                         cluster_cap=0.30, cluster_floor=0.0,
                         stock_cap=1.0, rf_annual=RF_ANNUAL):
    """Core rolling-cluster backtest. Clusters re-estimated each window."""
    from data_loader import load_data

    _, weekly_close, monthly_close = load_data()

    if rebal_freq in ("Monthly", "Quarterly"):
        close = monthly_close.loc[pd.Timestamp(start_date_str):]
        ann_factor = 12
        min_obs = _required_lookback_obs(lookback_yrs, ann_factor)
    else:
        close = weekly_close.loc[pd.Timestamp(start_date_str):]
        ann_factor = 52
        min_obs = _required_lookback_obs(lookback_yrs, ann_factor)

    returns = close.pct_change().dropna()
    tickers = TICKERS

    rebal_dates = get_rebalance_dates(returns.index, rebal_freq)
    lookback_td = pd.DateOffset(years=lookback_yrs)

    weights_schedule = {}
    cluster_history = {}
    corr_history = {}
    linkage_history = {}
    is_sharpe_history = {}
    silhouette_history = {}
    failed_dates = []
    prev_cluster_map = FIXED_CLUSTERS

    rf_p = (1 + rf_annual) ** (1 / ann_factor) - 1

    for rd in rebal_dates:
        window_start = rd - lookback_td
        wr = returns.loc[window_start:rd]
        wp = close.loc[window_start:rd]

        if len(wr) < min_obs:
            continue

        raw_map, corr, Z = cluster_from_returns(wr, n_clusters, linkage_method, tickers)
        aligned_map = align_cluster_labels(prev_cluster_map, raw_map, tickers)
        cluster_history[rd] = aligned_map
        corr_history[rd] = corr
        linkage_history[rd] = Z
        prev_cluster_map = aligned_map

        labels_arr = np.array([aligned_map[t] for t in tickers])
        unique_labels = np.unique(labels_arr)
        if len(unique_labels) >= 2:
            dist_matrix = np.sqrt(0.5 * (1 - corr.loc[tickers, tickers].values))
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = np.clip(dist_matrix, 0, None)
            label_ints = np.array([int(aligned_map[t][1:]) for t in tickers])
            sil_mean = silhouette_score(dist_matrix, label_ints, metric="precomputed")
            sil_per_stock = silhouette_samples(dist_matrix, label_ints, metric="precomputed")
            silhouette_history[rd] = {
                "mean": sil_mean,
                "per_stock": dict(zip(tickers, sil_per_stock)),
            }

        unique_clusters = sorted(set(aligned_map.values()))
        sector_upper = {c: cluster_cap for c in unique_clusters}
        sector_lower = {c: cluster_floor for c in unique_clusters}

        w = optimize_single_period(wp, wr, opt_target, cov_method,
                                   aligned_map, sector_upper, sector_lower, tickers,
                                   stock_cap=stock_cap)
        is_ew = all(abs(v - 1.0 / len(tickers)) < 1e-6 for v in w.values())
        if is_ew:
            failed_dates.append(rd)
        weights_schedule[rd] = w

        w_series = pd.Series(w).reindex(tickers, fill_value=0.0)
        is_pf_ret = (wr[tickers] * w_series).sum(axis=1)
        if len(is_pf_ret) > 1 and is_pf_ret.std(ddof=1) > 0:
            is_excess = is_pf_ret - rf_p
            is_sharpe_val = (is_excess.mean() * ann_factor) / (is_excess.std(ddof=1) * np.sqrt(ann_factor))
        else:
            is_sharpe_val = np.nan
        is_sharpe_history[rd] = is_sharpe_val

    pf_returns = simulate_portfolio(returns, weights_schedule, tickers)

    sorted_dates = sorted(cluster_history.keys())
    stability = {}
    for i in range(1, len(sorted_dates)):
        prev_labels = [cluster_history[sorted_dates[i - 1]].get(t, "?") for t in tickers]
        curr_labels = [cluster_history[sorted_dates[i]].get(t, "?") for t in tickers]
        stability[sorted_dates[i]] = adjusted_rand_score(prev_labels, curr_labels)

    oos_returns_per_window = {}
    oos_returns_all = []
    rebal_sorted = sorted(weights_schedule.keys())
    for i, rd in enumerate(rebal_sorted):
        oos_end = rebal_sorted[i + 1] if i + 1 < len(rebal_sorted) else returns.index[-1]
        oos_ret = returns.loc[rd:oos_end].iloc[1:]
        if oos_ret.empty:
            continue
        w_series = pd.Series(weights_schedule[rd]).reindex(tickers, fill_value=0.0)
        oos_pf_ret = (oos_ret[tickers] * w_series).sum(axis=1)
        oos_returns_per_window[rd] = oos_pf_ret
        oos_returns_all.append(oos_pf_ret)

    oos_weekly_returns = pd.concat(oos_returns_all) if oos_returns_all else pd.Series(dtype=float)

    return {
        "portfolio_returns": pf_returns,
        "weights": weights_schedule,
        "cluster_history": cluster_history,
        "corr_history": corr_history,
        "linkage_history": linkage_history,
        "cluster_stability": pd.Series(stability),
        "failed_dates": failed_dates,
        "rebal_dates": sorted_dates,
        "is_sharpe": pd.Series(is_sharpe_history),
        "oos_weekly_returns": oos_weekly_returns,
        "oos_returns_per_window": oos_returns_per_window,
        "silhouette_history": silhouette_history,
    }


def run_ew_backtest(start_date_str, rebal_freq):
    """Equal-weight benchmark."""
    from data_loader import load_data

    _, weekly_close, monthly_close = load_data()

    if rebal_freq in ("Monthly", "Quarterly"):
        close = monthly_close.loc[pd.Timestamp(start_date_str):]
    else:
        close = weekly_close.loc[pd.Timestamp(start_date_str):]

    returns = close.pct_change().dropna()
    tickers = TICKERS
    n = len(tickers)

    rebal_dates = get_rebalance_dates(returns.index, rebal_freq)
    weights_schedule = {rd: {t: 1.0 / n for t in tickers} for rd in rebal_dates}
    pf_returns = simulate_portfolio(returns, weights_schedule, tickers)
    return {
        "portfolio_returns": pf_returns,
        "weights": weights_schedule,
        "oos_weekly_returns": pf_returns.copy(),
    }


def run_plain_mv_backtest(start_date_str, lookback_yrs, rebal_freq, cov_method="sample",
                          opt_target="max_sharpe", rf_annual=RF_ANNUAL):
    """Plain mean-variance backtest with no cluster constraints."""
    from data_loader import load_data

    _, weekly_close, monthly_close = load_data()

    if rebal_freq in ("Monthly", "Quarterly"):
        close = monthly_close.loc[pd.Timestamp(start_date_str):]
        ann_factor = 12
        min_obs = _required_lookback_obs(lookback_yrs, ann_factor)
    else:
        close = weekly_close.loc[pd.Timestamp(start_date_str):]
        ann_factor = 52
        min_obs = _required_lookback_obs(lookback_yrs, ann_factor)

    returns = close.pct_change().dropna()
    tickers = TICKERS

    rebal_dates = get_rebalance_dates(returns.index, rebal_freq)
    lookback_td = pd.DateOffset(years=lookback_yrs)

    weights_schedule = {}
    is_sharpe_history = {}
    rf_p = (1 + rf_annual) ** (1 / ann_factor) - 1

    for rd in rebal_dates:
        window_start = rd - lookback_td
        wr = returns.loc[window_start:rd]
        wp = close.loc[window_start:rd]
        if len(wr) < min_obs:
            continue
        w = optimize_single_period(wp, wr, opt_target, cov_method,
                                   None, None, None, tickers, frequency=ann_factor)
        weights_schedule[rd] = w

        w_series = pd.Series(w).reindex(tickers, fill_value=0.0)
        is_pf_ret = (wr[tickers] * w_series).sum(axis=1)
        if len(is_pf_ret) > 1 and is_pf_ret.std(ddof=1) > 0:
            is_excess = is_pf_ret - rf_p
            is_sharpe_history[rd] = (is_excess.mean() * ann_factor) / (is_excess.std(ddof=1) * np.sqrt(ann_factor))

    oos_returns_per_window = {}
    oos_returns_all = []
    rebal_sorted = sorted(weights_schedule.keys())
    for i, rd in enumerate(rebal_sorted):
        oos_end = rebal_sorted[i + 1] if i + 1 < len(rebal_sorted) else returns.index[-1]
        oos_ret = returns.loc[rd:oos_end].iloc[1:]
        if oos_ret.empty:
            continue
        w_series = pd.Series(weights_schedule[rd]).reindex(tickers, fill_value=0.0)
        oos_pf_ret = (oos_ret[tickers] * w_series).sum(axis=1)
        oos_returns_per_window[rd] = oos_pf_ret
        oos_returns_all.append(oos_pf_ret)

    oos_weekly_returns = pd.concat(oos_returns_all) if oos_returns_all else pd.Series(dtype=float)

    pf_returns = simulate_portfolio(returns, weights_schedule, tickers)
    return {
        "portfolio_returns": pf_returns,
        "weights": weights_schedule,
        "is_sharpe": pd.Series(is_sharpe_history),
        "oos_weekly_returns": oos_weekly_returns,
        "oos_returns_per_window": oos_returns_per_window,
    }

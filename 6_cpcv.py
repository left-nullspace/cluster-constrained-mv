"""
6_cpcv.py

Combinatorial Purged Cross-Validation (Lopez de Prado, 2018 Ch.12).
Tests CMVO, CMVO (5% Stock Cap), Plain MV, and Equal Weight (1/N)
across textbook-style stitched CPCV paths.

Outputs:
  Table — CPCV OOS Sharpe distribution summary
  figures/6a_cpcv_split_structure.png
"""

import os
from math import comb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE, "figures")
DATA_DIR = os.path.join(BASE, "data")

from config import TICKERS, BASELINE_CONFIG
from data_loader import load_data
from engine import cluster_from_returns
from pypfopt import expected_returns, risk_models, EfficientFrontier

PLOT_BG = "#FFFFFF"; PLOT_TEXT = "#3B3B3B"; PLOT_SPINE = "#AAAAAA"; PLOT_GRID = "#D5D5D5"
SAGE = "#5B7553"; TERRA = "#C2785C"


def _plot_split_structure(splitter, index, out_path):
    """Render a static CPCV split map similar to VBT Pro's interactive view."""
    split_rows = []
    for mask_df in splitter.get_iter_split_masks():
        row = np.zeros(len(index), dtype=int)
        row[mask_df["train"].values] = 1
        row[mask_df["test"].values] = 2
        split_rows.append(row)

    split_matrix = np.vstack(split_rows)

    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([PLOT_BG, SAGE, TERRA])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(20, 12))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.imshow(split_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    n_splits = split_matrix.shape[0]
    xticks = np.linspace(0, len(index) - 1, 8, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([index[i].strftime("%Y-%m") for i in xticks], fontsize=18, color=PLOT_TEXT)

    ytick_idx = np.linspace(0, n_splits - 1, min(n_splits, 10), dtype=int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f"Split {i + 1}" for i in ytick_idx], fontsize=18, color=PLOT_TEXT)

    ax.set_xlabel("Weekly Return Index", fontsize=22, color=PLOT_TEXT)
    ax.set_ylabel("CPCV Split", fontsize=22, color=PLOT_TEXT)
    ax.set_title(
        "CPCV Split Structure\n"
        "(green = train, orange = test, white = purged/embargoed)",
        fontsize=28,
        fontweight="bold",
        color=PLOT_TEXT,
    )
    ax.tick_params(colors=PLOT_TEXT, labelsize=18)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


def _run_cmvo_on_split(train_ret, test_ret, tickers, bc, stock_cap=1.0):
    """Cluster on train, constrained max-Sharpe optimize, return test-period returns."""
    if len(train_ret) < 52 or len(test_ret) < 4:
        return pd.Series(dtype=float)
    try:
        cmap, _, _ = cluster_from_returns(train_ret, bc["k"], bc["linkage"], tickers)
        clusters = sorted(set(cmap.values()))
        sector_mapper = {t: cmap[t] for t in tickers}
        prices_approx = (1 + train_ret).cumprod() * 100
        mu = expected_returns.mean_historical_return(prices_approx, frequency=52)
        S = risk_models.sample_cov(prices_approx, frequency=52)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, min(stock_cap, 1.0)))
        ef.add_sector_constraints(sector_mapper,
            {c: 0.0 for c in clusters}, {c: bc["cluster_cap"] for c in clusters})
        ef.max_sharpe()
        weights = ef.clean_weights()
        w = pd.Series(weights).reindex(tickers).fillna(0)
        return (test_ret[tickers] * w).sum(axis=1)
    except Exception:
        pass
    return pd.Series(dtype=float)


def _run_plain_mv_on_split(train_ret, test_ret, tickers):
    """Unconstrained max-Sharpe optimize, return test-period returns."""
    if len(train_ret) < 52 or len(test_ret) < 4:
        return pd.Series(dtype=float)
    try:
        prices_approx = (1 + train_ret).cumprod() * 100
        mu = expected_returns.mean_historical_return(prices_approx, frequency=52)
        S = risk_models.sample_cov(prices_approx, frequency=52)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_sharpe()
        weights = ef.clean_weights()
        w = pd.Series(weights).reindex(tickers).fillna(0)
        return (test_ret[tickers] * w).sum(axis=1)
    except Exception:
        pass
    return pd.Series(dtype=float)


def _run_ew_on_split(test_ret, tickers):
    """Equal-weight, return test-period returns."""
    if len(test_ret) < 4:
        return pd.Series(dtype=float)
    w = pd.Series(1 / len(tickers), index=tickers)
    return (test_ret[tickers] * w).sum(axis=1)


def _ann_sharpe(returns):
    """Annualized Sharpe from weekly returns."""
    s = pd.Series(returns).dropna()
    if len(s) < 2 or s.std(ddof=1) <= 0:
        return np.nan
    return float((s.mean() * 52) / (s.std(ddof=1) * np.sqrt(52)))


def _build_fold_indices(n_obs, n_folds):
    """Recreate the equal-sized ordered folds used by PurgedKFoldCV."""
    return [np.asarray(fold, dtype=int) for fold in np.array_split(np.arange(n_obs), n_folds)]


def _identify_test_folds(test_index, fold_dates):
    """Identify which fold IDs appear in a split's test set."""
    pair = []
    test_idx = pd.Index(test_index)
    for fold_id, fd in enumerate(fold_dates):
        if len(test_idx.intersection(fd)) > 0:
            pair.append(fold_id)
    return tuple(pair)


def _build_pair_matchings_k2(n_folds):
    """Decompose all fold pairs into N-1 perfect matchings for even N."""
    if n_folds % 2 != 0:
        raise ValueError("k=2 CPCV path stitching requires an even number of folds.")

    nodes = list(range(n_folds))
    rounds = []
    for _ in range(n_folds - 1):
        pairs = []
        for i in range(n_folds // 2):
            pair = tuple(sorted((nodes[i], nodes[-(i + 1)])))
            pairs.append(pair)
        rounds.append(pairs)
        nodes = [nodes[0]] + [nodes[-1]] + nodes[1:-1]

    all_pairs = sorted(pair for rnd in rounds for pair in rnd)
    expected_pairs = sorted((i, j) for i in range(n_folds) for j in range(i + 1, n_folds))
    if all_pairs != expected_pairs:
        raise ValueError("Round-robin pair decomposition did not cover each split exactly once.")
    return rounds


def _stitch_cpcv_paths_k2(split_returns_by_pair, fold_dates, full_index):
    """Build textbook-style CPCV paths for k=2 using disjoint pair matchings."""
    n_folds = len(fold_dates)
    pair_rounds = _build_pair_matchings_k2(n_folds)

    path_returns = {}
    for r, pairs in enumerate(pair_rounds, start=1):
        used_folds = set()
        segments = []
        for pair in pairs:
            if pair not in split_returns_by_pair:
                raise ValueError(f"Missing split results for fold pair {pair}")
            used_folds.update(pair)
            split_ret = split_returns_by_pair[pair]
            expected_idx = fold_dates[pair[0]].append(fold_dates[pair[1]]).sort_values()
            segment = split_ret.reindex(expected_idx)
            if segment.isna().any() or len(segment) != len(expected_idx):
                raise ValueError(f"Incomplete stitched segment for pair {pair}")
            segments.append(segment)

        if used_folds != set(range(n_folds)):
            raise ValueError(f"Path {r} does not cover all folds exactly once.")

        path = pd.concat(segments).sort_index()
        if not path.index.equals(full_index):
            raise ValueError(f"Path {r} does not reconstruct the full chronology.")
        path_returns[f"Path {r}"] = path
    return path_returns


def main():
    print("=" * 60)
    print("  CPCV Robustness Check")
    print("=" * 60)

    try:
        import vectorbtpro as vbt
    except ImportError:
        print("\n  [SKIP] vectorbtpro not installed.")
        print("  Skipped: CPCV split structure (6a), CPCV Sharpe distribution (6b),")
        print("           data/6_cpcv_sharpe.csv, data/6_cpcv_path_sharpes.csv")
        return

    _, weekly_close, _ = load_data()
    returns = weekly_close[TICKERS].pct_change().dropna()
    bc = BASELINE_CONFIG
    n_folds = 10
    n_test_folds = 2
    purge_td = "2W"
    embargo_td = "1W"

    splitter = vbt.Splitter.from_purged_kfold(
        returns.index,
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purge_td=purge_td,
        embargo_td=embargo_td,
    )
    n_splits = splitter.n_splits
    phi = int((n_test_folds * comb(n_folds, n_test_folds)) / n_folds)
    print(f"\n  Splitter: {n_splits} combinatorial splits")
    print(f"  {n_folds} folds, {n_test_folds} test folds, 2-week purge, 1-week embargo")
    print(f"  Data: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Textbook CPCV paths: {phi}")

    train_data = splitter.take(returns, set_=0)
    test_data = splitter.take(returns, set_=1)
    fold_indices = _build_fold_indices(len(returns), n_folds)
    fold_dates = [returns.index[idx] for idx in fold_indices]

    split_fig_path = os.path.join(FIGURES_DIR, "6a_cpcv_split_structure.png")
    _plot_split_structure(splitter, returns.index, split_fig_path)
    print("  -> figures/6a_cpcv_split_structure.png")

    if n_test_folds != 2:
        raise NotImplementedError("This stitched CPCV implementation currently supports k=2 only.")

    strategy_names = ["CMVO", "CMVO (5% Stock Cap)", "Plain MV", "Equal Weight (1/N)"]
    split_results = {name: {} for name in strategy_names}
    print(f"\n  Running {n_splits} splits x 4 strategies ...")

    for i in range(n_splits):
        train_ret = train_data[i]
        test_ret = test_data[i]
        test_pair = _identify_test_folds(test_ret.index, fold_dates)
        if len(test_pair) != 2:
            raise ValueError(f"Expected 2 test folds, got {test_pair}")

        split_results["CMVO"][test_pair] = _run_cmvo_on_split(train_ret, test_ret, TICKERS, bc, stock_cap=1.0)
        split_results["CMVO (5% Stock Cap)"][test_pair] = _run_cmvo_on_split(train_ret, test_ret, TICKERS, bc, stock_cap=0.05)
        split_results["Plain MV"][test_pair] = _run_plain_mv_on_split(train_ret, test_ret, TICKERS)
        split_results["Equal Weight (1/N)"][test_pair] = _run_ew_on_split(test_ret, TICKERS)

        if (i + 1) % 15 == 0:
            print(f"    [{i+1}/{n_splits}]")

    path_returns = {name: _stitch_cpcv_paths_k2(split_results[name], fold_dates, returns.index)
                    for name in strategy_names}
    path_sharpes = {
        name: pd.Series({path_name: _ann_sharpe(path_ret)
                         for path_name, path_ret in path_returns[name].items()}).dropna()
        for name in strategy_names
    }

    # Print summary table
    print(f"\n  CPCV OOS Sharpe Distribution (stitched paths)")
    print("  " + "=" * 65)
    print(f"  {'Strategy':<12} {'N':>4} {'Mean':>7} {'Median':>8} {'Std':>7} {'Min':>7} {'Max':>7}")
    print("  " + "-" * 65)
    for name in strategy_names:
        s = path_sharpes[name]
        print(f"  {name:<12} {len(s):>4} {s.mean():>7.2f} {s.median():>8.2f} "
              f"{s.std():>7.2f} {s.min():>7.2f} {s.max():>7.2f}")
    print("  " + "=" * 65)

    # Save CSV
    rows = []
    for name in strategy_names:
        s = path_sharpes[name]
        rows.append({"Strategy": name, "N": len(s), "Mean": round(s.mean(), 3),
                      "Median": round(s.median(), 3), "Std": round(s.std(), 3),
                      "Min": round(s.min(), 3), "Max": round(s.max(), 3)})
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "6_cpcv_sharpe.csv"), index=False)
    print("  -> data/6_cpcv_sharpe.csv")

    detail_rows = []
    for name in strategy_names:
        for path_name, sharpe in path_sharpes[name].items():
            detail_rows.append({"Strategy": name, "Path": path_name, "Sharpe": round(float(sharpe), 6)})
    pd.DataFrame(detail_rows).to_csv(os.path.join(DATA_DIR, "6_cpcv_path_sharpes.csv"), index=False)
    print("  -> data/6_cpcv_path_sharpes.csv")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

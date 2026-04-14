"""
2_cluster_analysis.py

Full-period and rolling clustering analysis to select k.
Uses Ward hierarchical clustering on weekly return correlation-distance.
Justifies k=6 via metrics, elbow analysis, and rolling stability.

Outputs:
  data/2_full_period_metrics.csv      -> k vs silhouette/DB/purity/singletons
  data/2_rolling_stability.csv        -> ARI/co-clustering/persistence summary
  data/2_cluster_compositions_k6.csv  -> cluster members at k=6
  data/2_co_clustering_matrix.csv     -> 33x33 pairwise co-clustering frequency
  figures/2a_cluster_selection_metrics.png -> 2-panel: silhouette, Davies-Bouldin
  figures/2b_dendrogram.png               -> full-period dendrogram at k=6
  figures/2c_cluster_membership_heatmap.png -> rolling cluster membership over time
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendro
from scipy.cluster.hierarchy import set_link_color_palette
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, FIGURES_DIR, LOOKBACK_YEARS, LINKAGE, K_RANGE,
    PLOT_BG, PLOT_TEXT, PLOT_GRID, PLOT_ACCENT, PLOT_ACCENT2, PLOT_SPINE,
    CLUSTER_COLORS,
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def load_data():
    """Load universe prices and ticker list from Script 1 outputs."""
    prices = pd.read_csv(os.path.join(DATA_DIR, "universe_prices.csv"),
                         parse_dates=["Date"], index_col="Date")
    tl = pd.read_csv(os.path.join(DATA_DIR, "ticker_list.csv"))
    tickers = list(prices.columns)
    sector_map = dict(zip(tl["ticker"], tl["sector"]))
    return prices, tickers, sector_map


def resample_weekly(prices):
    """Resample daily prices to weekly (Friday) and compute returns."""
    weekly = prices.resample("W-FRI").last()
    returns = weekly.pct_change().dropna()
    return weekly, returns


def compute_corr_dist(returns, tickers):
    """Correlation matrix and correlation-distance matrix."""
    corr = returns[tickers].corr()
    dist = np.sqrt(0.5 * (1 - corr.values))  # Lopez de Prado (2016) correlation distance
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    return corr, dist


def run_clustering(returns, tickers, k, method):
    """Ward hierarchical clustering. Returns cluster_map, corr, dist, Z."""
    corr, dist = compute_corr_dist(returns, tickers)
    Z = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    cluster_map = {t: f"C{int(l)}" for t, l in zip(tickers, labels)}
    return cluster_map, corr, dist, Z


def compute_silhouette(dist, cluster_map, tickers):
    """Silhouette score on precomputed distance matrix."""
    label_ints = np.array([int(cluster_map[t][1:]) for t in tickers])
    if len(np.unique(label_ints)) < 2:
        return np.nan
    return silhouette_score(dist, label_ints, metric="precomputed")


def compute_db(returns, cluster_map, tickers):
    """Davies-Bouldin index on return space (stocks as samples)."""
    X = returns[tickers].T.values  # (n_stocks, n_weeks)
    label_ints = np.array([int(cluster_map[t][1:]) for t in tickers])
    if len(np.unique(label_ints)) < 2:
        return np.nan
    return davies_bouldin_score(X, label_ints)


def count_singletons(cluster_map):
    """Count clusters with exactly 1 member."""
    counts = Counter(cluster_map.values())
    return sum(1 for c in counts.values() if c == 1)


def align_cluster_labels(prev_map, curr_map, tickers):
    """Align labels between consecutive windows using the Hungarian algorithm."""
    prev_labels = sorted(set(prev_map.values()))
    curr_labels = sorted(set(curr_map.values()))
    n = max(len(prev_labels), len(curr_labels))

    cost = np.zeros((n, n))
    for i, cl in enumerate(curr_labels):
        curr_members = {t for t in tickers if curr_map.get(t) == cl}
        for j, pl in enumerate(prev_labels):
            prev_members = {t for t in tickers if prev_map.get(t) == pl}
            cost[i, j] = -len(curr_members & prev_members)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(curr_labels) and c < len(prev_labels):
            mapping[curr_labels[r]] = prev_labels[c]

    used = set(mapping.values())
    next_id = n + 1
    for cl in curr_labels:
        if cl not in mapping:
            while f"C{next_id}" in used:
                next_id += 1
            mapping[cl] = f"C{next_id}"
            used.add(f"C{next_id}")

    return {t: mapping.get(l, l) for t, l in curr_map.items()}


def compute_jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_cluster_persistence(prev_map, curr_map, tickers):
    """Average Jaccard similarity of cluster memberships vs previous window."""
    curr_clusters = {}
    for t in tickers:
        curr_clusters.setdefault(curr_map[t], set()).add(t)
    prev_clusters = {}
    for t in tickers:
        prev_clusters.setdefault(prev_map[t], set()).add(t)
    jaccards = []
    for label, curr_members in curr_clusters.items():
        prev_members = prev_clusters.get(label, set())
        jaccards.append(compute_jaccard(curr_members, prev_members))
    return np.mean(jaccards)


# ═══════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════

def full_period_analysis(weekly_returns, tickers, sector_map, k_range, method):
    """Run clustering for each k, compute full-period metrics."""
    metrics = []
    results = {}
    n = len(tickers)

    for k in k_range:
        cmap, corr, dist, Z = run_clustering(weekly_returns, tickers, k, method)
        sil = compute_silhouette(dist, cmap, tickers)
        db = compute_db(weekly_returns, cmap, tickers)
        singletons = count_singletons(cmap)

        metrics.append({
            "k": k, "Silhouette": round(sil, 4), "Davies-Bouldin": round(db, 3),
            "Singletons": singletons,
        })
        results[k] = {"cluster_map": cmap, "corr": corr, "dist": dist, "Z": Z}

    return metrics, results


def _rolling_singleton_pct(weekly_returns, tickers, lookback_years, method, k_range):
    """Quick pass: compute % of rolling windows with singletons for each k."""
    lookback = pd.DateOffset(years=lookback_years)
    dates = weekly_returns.index
    first_valid = dates[0] + lookback
    counts = {k: 0 for k in k_range}
    n_windows = 0

    for rd in dates:
        if rd < first_valid:
            continue
        window = weekly_returns.loc[rd - lookback:rd]
        if len(window) < 52:
            continue
        corr, dist = compute_corr_dist(window, tickers)
        Z_w = linkage(squareform(dist, checks=False), method=method)
        for k in k_range:
            labels = fcluster(Z_w, t=k, criterion="maxclust")
            if any(s == 1 for s in Counter(labels.tolist()).values()):
                counts[k] += 1
        n_windows += 1
        if n_windows % 200 == 0:
            print(f"    ... {n_windows} windows")

    return {k: counts[k] / n_windows if n_windows > 0 else 0 for k in k_range}


def rolling_analysis(weekly_returns, tickers, sector_map, lookback_years, method, k,
                     singleton_ks=None):
    """Rolling 5-year window clustering at fixed k. Returns stability metrics.
    If singleton_ks is provided, also counts singleton % per window for those k values.
    """
    lookback = pd.DateOffset(years=lookback_years)
    dates = weekly_returns.index
    first_valid = dates[0] + lookback
    if singleton_ks is None:
        singleton_ks = []

    ari_series = {}
    sil_series = {}
    persistence_series = {}
    cluster_history = {}   # {date: {ticker: cluster_label}}
    n_tickers = len(tickers)
    co_cluster = np.zeros((n_tickers, n_tickers))
    singleton_windows = {sk: 0 for sk in singleton_ks}

    prev_map = None
    n_windows = 0

    for rd in dates:
        if rd < first_valid:
            continue
        window = weekly_returns.loc[rd - lookback:rd]
        if len(window) < 52:
            continue

        cmap, corr, dist, Z_w = run_clustering(window, tickers, k, method)

        # track singletons for extra k values using same linkage
        for sk in singleton_ks:
            labels_sk = fcluster(Z_w, t=sk, criterion="maxclust")
            sizes_sk = Counter(labels_sk.tolist())
            if any(s == 1 for s in sizes_sk.values()):
                singleton_windows[sk] += 1

        if prev_map is not None:
            aligned = align_cluster_labels(prev_map, cmap, tickers)
            cluster_history[rd] = aligned

            prev_labels = [prev_map[t] for t in tickers]
            curr_labels = [aligned[t] for t in tickers]
            ari_series[rd] = adjusted_rand_score(prev_labels, curr_labels)
            sil_series[rd] = compute_silhouette(dist, aligned, tickers)
            persistence_series[rd] = compute_cluster_persistence(prev_map, aligned, tickers)

            label_arr = [aligned[t] for t in tickers]
            for i in range(n_tickers):
                for j in range(i, n_tickers):
                    if label_arr[i] == label_arr[j]:
                        co_cluster[i, j] += 1
                        co_cluster[j, i] += 1

            prev_map = aligned
        else:
            label_arr = [cmap[t] for t in tickers]
            for i in range(n_tickers):
                for j in range(i, n_tickers):
                    if label_arr[i] == label_arr[j]:
                        co_cluster[i, j] += 1
                        co_cluster[j, i] += 1
            cluster_history[rd] = cmap
            prev_map = cmap

        n_windows += 1
        if n_windows % 100 == 0:
            print(f"    ... {n_windows} windows processed")

    if n_windows > 0:
        co_cluster /= n_windows

    co_df = pd.DataFrame(co_cluster, index=tickers, columns=tickers)
    singleton_pct = {sk: singleton_windows[sk] / n_windows if n_windows > 0 else 0
                     for sk in singleton_ks}

    return {
        "ari_series": pd.Series(ari_series),
        "silhouette_series": pd.Series(sil_series),
        "persistence_series": pd.Series(persistence_series),
        "co_clustering_matrix": co_df,
        "n_windows": n_windows,
        "singleton_pct": singleton_pct,
        "cluster_history": cluster_history,
    }


# ═══════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def plot_cluster_selection_metrics(results_by_k, metrics, k_range, tickers):
    """2-panel figure: Silhouette and Davies-Bouldin by k."""
    ks = list(k_range)

    sils = [m["Silhouette"] for m in metrics]
    dbs = [m["Davies-Bouldin"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PLOT_BG)

    panels = [
        (axes[0], sils, "Avg Silhouette Score", "higher = tighter clusters", PLOT_ACCENT),
        (axes[1], dbs, "Davies-Bouldin Index", "lower = better separation", PLOT_ACCENT2),
    ]

    for ax, vals, title, subtitle, color in panels:
        ax.set_facecolor(PLOT_BG)
        ax.plot(ks, vals, "o-", color=color, linewidth=2.5, markersize=9)
        # Highlight k=6
        k6_idx = ks.index(6)
        ax.plot(6, vals[k6_idx], "o", color="#C2785C", markersize=14, zorder=5)
        ax.set_xticks(ks)
        ax.set_xlabel("k", color=PLOT_TEXT, fontsize=12)
        ax.set_title(title, color=PLOT_TEXT, fontsize=14, fontweight="bold", pad=8)
        ax.text(0.98, 0.02, subtitle, transform=ax.transAxes, fontsize=13,
                ha="right", va="bottom", color=PLOT_TEXT, fontstyle="italic")
        ax.tick_params(colors=PLOT_TEXT, labelsize=11)
        ax.yaxis.grid(True, color=PLOT_GRID, alpha=0.5, linestyle=":")
        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cluster Selection Metrics (k = four to nine, k = 6 highlighted)",
                 color=PLOT_TEXT, fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "2a_cluster_selection_metrics.png"),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


def plot_dendrogram(Z, tickers, k):
    """Full-period dendrogram with cut line at k."""
    n = len(tickers)
    merge_idx = n - k
    if merge_idx > 0:
        threshold = (Z[merge_idx - 1, 2] + Z[merge_idx, 2]) / 2
    else:
        threshold = Z[merge_idx, 2] * 0.9

    set_link_color_palette(CLUSTER_COLORS[:max(k, 6)])

    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    scipy_dendro(Z, labels=tickers, ax=ax, leaf_rotation=90, leaf_font_size=14,
                 color_threshold=threshold, above_threshold_color="#AAAAAA")

    ax.axhline(y=threshold, linestyle="--", color=PLOT_ACCENT2, linewidth=2, alpha=0.85)
    ax.text(ax.get_xlim()[1] * 0.98, threshold, f"  k={k}",
            ha="right", va="bottom", fontsize=16, fontweight="bold",
            color=PLOT_ACCENT2, alpha=0.85)

    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward, k={k})",
                 color=PLOT_TEXT, fontsize=20, fontweight="bold", pad=15)
    ax.set_ylabel("Correlation Distance", color=PLOT_TEXT, fontsize=14)
    ax.tick_params(axis="x", colors=PLOT_TEXT, labelsize=14)
    ax.tick_params(axis="y", colors=PLOT_TEXT, labelsize=13)
    ax.yaxis.grid(True, color=PLOT_GRID, alpha=0.5, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)

    plt.subplots_adjust(bottom=0.18, top=0.92)
    fig.savefig(os.path.join(FIGURES_DIR, "2b_dendrogram.png"),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)




SECTOR_SHORT = {
    "Communication Services": "Comm Svcs",
    "Consumer Discretionary": "Cons Disc",
    "Consumer Staples":       "Cons Stpl",
    "Energy":                 "Energy",
    "Financials":             "Financials",
    "Health Care":            "Health",
    "Industrials":            "Industrials",
    "Information Technology": "Info Tech",
    "Materials":              "Materials",
    "Real Estate":            "Real Est",
    "Utilities":              "Utilities",
}


def _sector_sorted_tickers(tickers, sector_map):
    """Sort tickers by sector, return (sorted_tickers, sector_boundaries)."""
    st = sorted(tickers, key=lambda t: (sector_map.get(t, ""), t))
    boundaries = []   # (start_idx, end_idx, short_name)
    prev = sector_map.get(st[0], "")
    start = 0
    for i, t in enumerate(st):
        s = sector_map.get(t, "")
        if s != prev:
            boundaries.append((start, i - 1, SECTOR_SHORT.get(prev, prev)))
            start = i
            prev = s
    boundaries.append((start, len(st) - 1, SECTOR_SHORT.get(prev, prev)))
    return st, boundaries


def _year_xticks(dates):
    """Return (positions, labels) for January-only year ticks."""
    positions, labels = [], []
    prev_yr = None
    for j, d in enumerate(dates):
        if d.month <= 2 and d.year != prev_yr:   # first occurrence near Jan
            positions.append(j)
            labels.append(str(d.year))
            prev_yr = d.year
    return positions, labels


def plot_correlation_before_after(weekly_returns, tickers, sector_map, cluster_map_k6):
    """Side-by-side correlation matrices: GICS sector order vs cluster order."""
    SECTOR_SHORT_LOCAL = {
        "Communication Services": "Comm Svcs", "Consumer Discretionary": "Cons Disc",
        "Consumer Staples": "Cons Stpl", "Energy": "Energy", "Financials": "Financials",
        "Health Care": "Health", "Industrials": "Industrials",
        "Information Technology": "Info Tech", "Materials": "Materials",
        "Real Estate": "Real Est", "Utilities": "Utilities",
    }

    # Left: sorted by GICS sector
    sector_tickers = sorted(tickers, key=lambda t: (sector_map.get(t, ""), t))
    corr_sector = weekly_returns[sector_tickers].corr()

    # Right: sorted by cluster
    cluster_int = {t: int(cluster_map_k6[t][1:]) for t in tickers}
    clustered_tickers = sorted(tickers, key=lambda t: (cluster_int[t], sector_map.get(t, "")))
    corr_clustered = weekly_returns[clustered_tickers].corr()

    def _get_bounds(tick_list, group_fn, short_map):
        bounds = []
        prev = group_fn(tick_list[0])
        start = 0
        for i, t in enumerate(tick_list):
            g = group_fn(t)
            if g != prev:
                bounds.append((start, i - 1, short_map.get(prev, str(prev))))
                start = i
                prev = g
        bounds.append((start, len(tick_list) - 1, short_map.get(prev, str(prev))))
        return bounds

    sector_bounds = _get_bounds(sector_tickers, lambda t: sector_map.get(t, ""), SECTOR_SHORT_LOCAL)
    cluster_short = {i: f"C{i}" for i in range(1, 12)}
    cluster_bounds = _get_bounds(clustered_tickers, lambda t: cluster_int[t], cluster_short)

    fig = plt.figure(figsize=(26, 11))
    fig.patch.set_facecolor(PLOT_BG)

    ax1 = fig.add_axes([0.10, 0.08, 0.36, 0.80])
    ax2 = fig.add_axes([0.54, 0.08, 0.36, 0.80])

    for ax, corr_df, title, tick_list, bounds, side in [
        (ax1, corr_sector, "By GICS Sector", sector_tickers, sector_bounds, "left"),
        (ax2, corr_clustered, "By Cluster (k = 6)", clustered_tickers, cluster_bounds, "right"),
    ]:
        ax.set_facecolor(PLOT_BG)
        im = ax.imshow(corr_df.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=28, fontweight="bold", color=PLOT_TEXT, pad=14)
        ax.tick_params(colors=PLOT_TEXT)
        for s in ax.spines.values():
            s.set_color(PLOT_SPINE)

        for start, end, short in bounds:
            if start > 0:
                ax.axhline(y=start - 0.5, color="white", linewidth=2.5)
                ax.axvline(x=start - 0.5, color="white", linewidth=2.5)

        n = len(tick_list)
        for start, end, short in bounds:
            mid = (start + end) / 2
            y_mid = fig.transFigure.inverted().transform(ax.transData.transform((0, mid)))[1]
            y_s = fig.transFigure.inverted().transform(ax.transData.transform((0, start - 0.3)))[1]
            y_e = fig.transFigure.inverted().transform(ax.transData.transform((0, end + 0.3)))[1]
            x_base = fig.transFigure.inverted().transform(ax.transData.transform((0, 0)))[0]
            fig.text(x_base - 0.02, y_mid, short, fontsize=22,
                     color=PLOT_TEXT, ha="right", va="center")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Correlation", color=PLOT_TEXT, fontsize=20)
    cbar.ax.tick_params(colors=PLOT_TEXT, labelsize=18)

    fig.suptitle("Correlation Heatmap: GICS Sector Order vs Cluster Order",
                 fontsize=32, fontweight="bold", color=PLOT_TEXT, y=1.03)
    fig.savefig(os.path.join(FIGURES_DIR, "2d_correlation_before_after.png"),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


def plot_cluster_membership_heatmap(cluster_history, tickers, sector_map):
    """Rolling cluster membership heatmap — sector + stock labels on y, time on x."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    dates = sorted(cluster_history.keys())
    n_dates = len(dates)
    sorted_tickers, sec_bounds = _sector_sorted_tickers(tickers, sector_map)
    n_tickers = len(sorted_tickers)

    # Build matrix
    matrix = np.zeros((n_tickers, n_dates), dtype=int)
    for j, d in enumerate(dates):
        cm = cluster_history[d]
        for i, t in enumerate(sorted_tickers):
            matrix[i, j] = int(cm.get(t, "C0")[1:])

    k_vals = sorted(set(matrix.flatten()))
    n_colors = max(k_vals) if k_vals else 6
    cmap = ListedColormap(CLUSTER_COLORS[:n_colors])
    norm = BoundaryNorm(list(range(1, n_colors + 2)), cmap.N)

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(PLOT_BG)

    # Layout: [sector labels | bracket | tickers | heatmap | colorbar]
    ax = fig.add_axes([0.13, 0.06, 0.77, 0.86])  # main heatmap
    ax.set_facecolor(PLOT_BG)

    ax.imshow(matrix, aspect="auto", interpolation="nearest",
              cmap=cmap, norm=norm, origin="upper")

    # Y-axis: stock tickers tight to the heatmap
    ax.set_yticks(range(n_tickers))
    ax.set_yticklabels(sorted_tickers, fontsize=13, color=PLOT_TEXT)

    # Separator lines between sectors
    for start, end, short in sec_bounds:
        if start > 0:
            ax.axhline(y=start - 0.5, color=PLOT_SPINE, linewidth=1.2, alpha=0.7)

    # Sector labels + bracket bars drawn in figure coordinates
    for start, end, short in sec_bounds:
        mid = (start + end) / 2
        # Convert data y to figure y
        y_fig_mid = ax.transData.transform((0, mid))
        y_fig_start = ax.transData.transform((0, start - 0.3))
        y_fig_end = ax.transData.transform((0, end + 0.3))
        # Convert to figure fraction
        y_mid = fig.transFigure.inverted().transform(y_fig_mid)[1]
        y_s = fig.transFigure.inverted().transform(y_fig_start)[1]
        y_e = fig.transFigure.inverted().transform(y_fig_end)[1]
        # Sector name
        fig.text(0.01, y_mid, short, fontsize=17, fontweight="bold", color=PLOT_TEXT,
                 ha="left", va="center")
        # Bracket bar
        fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
            (0.09, y_s), (0.09, y_e),
            transform=fig.transFigure, arrowstyle="-",
            color=PLOT_SPINE, linewidth=2, clip_on=False))

    # X-axis: January of each year
    xpos, xlabels = _year_xticks(dates)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=14, color=PLOT_TEXT)

    ax.set_xlabel("Year", color=PLOT_TEXT, fontsize=15)
    ax.set_title("Rolling Cluster Membership Over Time (k = 6)",
                 color=PLOT_TEXT, fontsize=22, fontweight="bold", pad=14)
    ax.tick_params(colors=PLOT_TEXT, length=4)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)

    # Colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
        ticks=range(1, n_colors + 1), pad=0.015, shrink=0.7)
    cbar.set_label("Cluster", color=PLOT_TEXT, fontsize=20)
    cbar.ax.tick_params(colors=PLOT_TEXT, labelsize=18)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "2c_cluster_membership_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  PRINT FUNCTIONS
# ═══════════════════════════════════════════════════════════

def print_full_period_table(metrics):
    """Print full-period clustering metrics (static, entire sample)."""
    print(f"\n  {'k':>3}  {'Silhouette':>11}  {'Davies-Bouldin':>15}  {'Singletons':>11}")
    print(f"  {'---':>3}  {'-'*11}  {'-'*15}  {'-'*11}")
    for row in metrics:
        print(f"  {row['k']:>3}  {row['Silhouette']:>11.4f}  {row['Davies-Bouldin']:>15.3f}"
              f"  {row['Singletons']:>11}")


def print_rolling_k_table(metrics):
    """Print rolling metrics by k (singleton %)."""
    print(f"\n  {'k':>3}  {'Roll Sing%':>11}")
    print(f"  {'---':>3}  {'-'*11}")
    for row in metrics:
        print(f"  {row['k']:>3}  {row['Roll Sing%']:>10.1%}")


def print_cluster_compositions(cluster_map, sector_map, tickers, k):
    """Print cluster members and their sectors. Returns a DataFrame."""
    clusters = {}
    for t in tickers:
        clusters.setdefault(cluster_map[t], []).append(t)

    rows = []
    print(f"\n  Cluster compositions at k={k}:")
    print(f"  {'Cluster':<10} {'Size':>5}  {'Tickers':<40} {'Sectors'}")
    print(f"  {'-'*10} {'-'*5}  {'-'*40} {'-'*30}")

    for label in sorted(clusters, key=lambda x: int(x[1:])):
        members = clusters[label]
        sectors = [sector_map.get(t, "?") for t in members]
        sector_summary = ", ".join(sorted(set(sectors)))
        ticker_str = ", ".join(members)
        print(f"  {label:<10} {len(members):>5}  {ticker_str:<40} {sector_summary}")
        for t in members:
            rows.append({"Cluster": label, "Ticker": t, "Sector": sector_map.get(t, "?")})

    return pd.DataFrame(rows)


def print_rolling_summary(rolling_results):
    """Print rolling stability summary table."""
    ari = rolling_results["ari_series"]
    pers = rolling_results["persistence_series"]

    rows = [
        {"Metric": "Adjusted Rand Index",
         "Mean": f"{ari.mean():.3f}", "Std": f"{ari.std():.3f}",
         "Min": f"{ari.min():.3f}", "Max": f"{ari.max():.3f}"},
        {"Metric": "Cluster Persistence",
         "Mean": f"{pers.mean():.3f}", "Std": f"{pers.std():.3f}",
         "Min": f"{pers.min():.3f}", "Max": f"{pers.max():.3f}"},
    ]

    print(f"\n  {'Metric':<28} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in rows:
        print(f"  {r['Metric']:<28} {r['Mean']:>6} {r['Std']:>6} "
              f"{r['Min']:>6} {r['Max']:>6}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Cluster Analysis")
    print("  Full-Period + Rolling Stability")
    print("=" * 60)

    # ── Step 1: Load and resample ───────────────────────────
    print("\n[Step 1] Loading data and computing weekly returns ...")
    prices, tickers, sector_map = load_data()
    weekly_prices, weekly_returns = resample_weekly(prices)
    print(f"  Weekly returns : {weekly_returns.shape[0]} weeks x {weekly_returns.shape[1]} tickers")
    print(f"  Date range     : {weekly_returns.index[0].strftime('%Y-%m-%d')}"
          f" to {weekly_returns.index[-1].strftime('%Y-%m-%d')}")

    # ── Step 2: Full-period clustering for each k ───────────
    print(f"\n[Step 2] Full-period clustering for k = {{{min(K_RANGE)}..{max(K_RANGE)}}} ...")
    metrics, results = full_period_analysis(weekly_returns, tickers, sector_map, K_RANGE, LINKAGE)

    # ── Step 2b: Rolling singleton % for each k ──────────────
    print("\n[Step 2b] Computing rolling singleton % per k ...")
    roll_sing = _rolling_singleton_pct(weekly_returns, tickers, LOOKBACK_YEARS, LINKAGE, K_RANGE)
    for m in metrics:
        m["Roll Sing%"] = roll_sing.get(m["k"], 0)

    # ── Step 3: Print tables ────────────────────────────────
    print("\n[Step 3a] Full-period metrics")
    print_full_period_table(metrics)

    print("\n[Step 3b] Rolling metrics by k (5-year windows)")
    print_rolling_k_table(metrics)

    # ── Step 4: Cluster selection metrics (4-panel) ─────────
    print("\n[Step 4] Saving cluster selection metrics ...")
    plot_cluster_selection_metrics(results, metrics, K_RANGE, tickers)
    print("  -> figures/2a_cluster_selection_metrics.png")

    # ── Step 5: Dendrogram at k=6 ──────────────────────────
    print("\n[Step 5] Saving dendrogram (k=6) ...")
    plot_dendrogram(results[6]["Z"], tickers, k=6)
    print("  -> figures/2b_dendrogram.png")

    # ── Step 6: Cluster compositions for k=5,6,7,8 ──────────
    print("\n[Step 6] Cluster compositions")
    compositions_df = None
    for k_show in [5, 6, 7, 8]:
        if k_show in results:
            df = print_cluster_compositions(results[k_show]["cluster_map"], sector_map, tickers, k=k_show)
            if k_show == 6:
                compositions_df = df

    # ── Step 7: Rolling stability analysis ──────────────────
    print(f"\n[Step 7] Rolling cluster stability ({LOOKBACK_YEARS}-year lookback, k=6) ...")
    rolling = rolling_analysis(weekly_returns, tickers, sector_map, LOOKBACK_YEARS, LINKAGE, k=6)
    print(f"  Total windows: {rolling['n_windows']}")

    # ── Step 8: Rolling summary ─────────────────────────────
    print("\n[Step 8] Rolling stability summary")
    stability_df = print_rolling_summary(rolling)

    # ── Step 9: Cluster membership heatmap ───────────────
    print("\n[Step 9] Saving cluster membership heatmap ...")
    plot_cluster_membership_heatmap(rolling["cluster_history"], tickers, sector_map)
    print("  -> figures/2c_cluster_membership_heatmap.png")

    # ── Step 10: Correlation before/after clustering ───────
    print("\n[Step 10] Saving correlation before/after clustering ...")
    plot_correlation_before_after(weekly_returns, tickers, sector_map, results[6]["cluster_map"])
    print("  -> figures/2d_correlation_before_after.png")

    # ═════════════════════════════════════════════════════════
    #  SAVE OUTPUTS
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Saving outputs")
    print("=" * 60)

    pd.DataFrame(metrics).to_csv(os.path.join(DATA_DIR, "2_full_period_metrics.csv"), index=False)
    print("[Save 1] 2_full_period_metrics.csv")

    stability_df.to_csv(os.path.join(DATA_DIR, "2_rolling_stability.csv"), index=False)
    print("[Save 2] 2_rolling_stability.csv")

    compositions_df.to_csv(os.path.join(DATA_DIR, "2_cluster_compositions_k6.csv"), index=False)
    print("[Save 3] 2_cluster_compositions_k6.csv")

    print("\n" + "=" * 60)
    print("  Done. k=6 selected.")
    print("=" * 60)


if __name__ == "__main__":
    main()

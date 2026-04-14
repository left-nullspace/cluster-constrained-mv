"""
4_results.py

Backtest results for the submitted 33-stock universe using:
  CMVO (Max Sharpe), Plain MV (Max Sharpe),
  CMVO (Min Vol), Plain MV (Min Vol),
  CMVO + 5% stock cap, and Equal Weight (1/N).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE, "figures")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the submission-local config first so the parent engine/data loader use the
# submitted universe rather than the parent project's hard-coded one.
from config import BASELINE_CONFIG, TICKERS, CRISIS_PERIODS, RF_ANNUAL
from engine import run_rolling_backtest, run_ew_backtest, run_plain_mv_backtest
from metrics import compute_metrics, compute_turnover, oos_return_distribution_stats, trim_all_to_common_start, compute_es

# ── Plot style ──
PLOT_BG   = "#FFFFFF"
PLOT_TEXT  = "#3B3B3B"
PLOT_GRID  = "#D5D5D5"
PLOT_SPINE = "#AAAAAA"

STRATEGY_COLORS = {
    "CMVO": "#5B7553",
    "Plain MV":               "#C2785C",
    "CMVO (Min Vol)":         "#4A7C59",
    "Plain MV (Min Vol)":     "#A65E46",
    "CMVO (5% Stock Cap)":    "#D4A754",
    "Equal Weight (1/N)":     "#6B8EA7",
}

DISPLAY_NAMES = {
    "CMVO": "CMVO\nMax Sharpe",
    "Plain MV": "Plain MV\nMax Sharpe",
    "CMVO (Min Vol)": "CMVO\nMin Vol",
    "Plain MV (Min Vol)": "Plain MV\nMin Vol",
    "CMVO (5% Stock Cap)": "CMVO\n5% Cap",
    "Equal Weight (1/N)": "Equal Weight\n(1/N)",
}

STRATEGY_ORDER = [
    "CMVO",
    "Plain MV",
    "CMVO (Min Vol)",
    "Plain MV (Min Vol)",
    "CMVO (5% Stock Cap)",
    "Equal Weight (1/N)",
]


# ═══════════════════════════════════════════════════════════
#  RUN BACKTESTS
# ═══════════════════════════════════════════════════════════

def _backtest_summary(name, res):
    """Print one-line summary after a backtest finishes."""
    ret = res["portfolio_returns"]
    if ret.empty:
        print(f"           {name}: no OOS returns")
        return
    n_splits = len(res.get("weights", {}))
    first = ret.index[0].strftime("%Y-%m-%d")
    last = ret.index[-1].strftime("%Y-%m-%d")
    total = (1 + ret).prod() - 1
    print(f"           {name}: {n_splits} splits  |  {first} → {last}  |  total return {total*100:,.1f}%")


def _align_results(results, strategy_order, lookback_years):
    """Align all strategy return series to a common OOS start date."""
    returns_dict = {name: results[name]["portfolio_returns"] for name in strategy_order}
    trimmed, common_start = trim_all_to_common_start(returns_dict)
    print(f"\n  Aligned all series to {common_start.strftime('%Y-%m-%d')} "
          f"(optimized strategies require {lookback_years}-yr lookback before first rebalance)")
    for name in strategy_order:
        aligned = trimmed[name]
        results[name]["portfolio_returns"] = aligned
        results[name]["oos_weekly_returns"] = aligned.copy()
    return results, common_start


def run_all_backtests():
    """Run the submitted results backtests and return aligned results."""
    from data_loader import load_data
    daily_close, _, _ = load_data()
    start_str = str(daily_close.index.min().date())

    bc = BASELINE_CONFIG

    print(f"\n  Running backtests (data from {start_str}) ...")

    print("    [1/6] CMVO (Max Sharpe) ...")
    cmvo = run_rolling_backtest(
        start_str, bc["lookback"], bc["rebal"], bc["k"],
        bc["linkage"], bc["target"], bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    )
    _backtest_summary("CMVO", cmvo)

    print("    [2/6] Plain MV (Max Sharpe) ...")
    plain = run_plain_mv_backtest(
        start_str, bc["lookback"], bc["rebal"],
        cov_method=bc["cov"], opt_target="max_sharpe", rf_annual=bc["rf_annual"]
    )
    _backtest_summary("Plain MV", plain)

    print("    [3/6] CMVO (Min Vol) ...")
    cmvo_min_vol = run_rolling_backtest(
        start_str, bc["lookback"], bc["rebal"], bc["k"],
        bc["linkage"], "min_volatility", bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
    )
    _backtest_summary("CMVO MinVol", cmvo_min_vol)

    print("    [4/6] Plain MV (Min Vol) ...")
    plain_min_vol = run_plain_mv_backtest(
        start_str, bc["lookback"], bc["rebal"],
        cov_method=bc["cov"], opt_target="min_volatility", rf_annual=bc["rf_annual"]
    )
    _backtest_summary("Plain MinVol", plain_min_vol)

    print("    [5/6] Equal Weight (weekly rebalance) ...")
    ew = run_ew_backtest(start_str, bc["rebal"])
    _backtest_summary("Equal Weight", ew)

    print("    [6/6] CMVO + 5% stock cap ...")
    capped = run_rolling_backtest(
        start_str, bc["lookback"], bc["rebal"], bc["k"],
        bc["linkage"], bc["target"], bc["cov"],
        cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
        stock_cap=0.05, rf_annual=bc["rf_annual"],
    )
    _backtest_summary("CMVO+5%cap", capped)

    all_results = {
        "CMVO": cmvo,
        "Plain MV": plain,
        "CMVO (Min Vol)": cmvo_min_vol,
        "Plain MV (Min Vol)": plain_min_vol,
        "CMVO (5% Stock Cap)": capped,
        "Equal Weight (1/N)": ew,
    }
    all_results, common_start = _align_results(all_results, STRATEGY_ORDER, bc["lookback"])

    # Load weekly stock returns for turnover calculation (DeMiguel 2009, Eq.15)
    _, weekly_close, _ = load_data()
    weekly_returns = weekly_close[TICKERS].pct_change().dropna()

    return all_results, common_start, weekly_returns


# ═══════════════════════════════════════════════════════════
#  TABLE 1: Performance Metrics
# ═══════════════════════════════════════════════════════════

def _compute_covar_coes(strategy_returns, benchmark_returns, alpha=0.05):
    """CoVaR and CoES — strategy risk conditional on benchmark distress.

    CoVaR(α): α-quantile of strategy returns on dates when benchmark <= its VaR(α).
    CoES(α):  mean of strategy returns on those same distressed dates.
    """
    bench_var = benchmark_returns.quantile(alpha)
    distress_mask = benchmark_returns <= bench_var
    strat_distressed = strategy_returns[distress_mask]
    if len(strat_distressed) < 2:
        return np.nan, np.nan
    covar = float(strat_distressed.quantile(alpha))
    coes = float(strat_distressed.mean())
    return covar, coes


def performance_table(results, strategy_order, weekly_returns=None):
    """Build the comparison table with return, risk, and concentration metrics."""
    rows = []
    for name in strategy_order:
        res = results[name]
        ret = res["portfolio_returns"]
        m = compute_metrics(ret, label=name)

        # Turnover (DeMiguel 2009, Eq.15: drifted vs target)
        avg_turnover = np.nan
        if "weights" in res:
            to = compute_turnover(res["weights"], tickers=TICKERS, returns=weekly_returns)
            if len(to) > 0:
                avg_turnover = round(to.mean() * 100, 1)

        # ES at 5% (weekly, reported as %)
        es = round(compute_es(ret) * 100, 2)

        # HHI (sum of squared weights) — avg across rebalances
        hhi = np.nan
        if "weights" in res and res["weights"]:
            ws = res["weights"]
            hhis = [sum(v ** 2 for v in ws[d].values()) for d in sorted(ws.keys())]
            hhi = round(float(np.mean(hhis)), 4)

        rows.append({
            "Strategy":           name,
            "Ann. Return %":      m.get("Ann. Return %", np.nan),
            "Ann. Volatility %":  m.get("Ann. Volatility %", np.nan),
            "Ann. Sharpe":        m.get("Sharpe", np.nan),
            "Max DD %":           m.get("Max DD %", np.nan),
            "ES (5%) %":          es,
            "HHI":                hhi,
            "Avg Turnover %":     avg_turnover,
        })
    return pd.DataFrame(rows)


def print_performance_table(df, title="Performance Metrics Comparison"):
    """Pretty-print the performance table."""
    print(f"\n  {title}")
    print("  " + "=" * 100)
    print(f"  {'Strategy':<28} {'AnnRet%':>8} {'AnnVol%':>8} {'Sharpe':>7} {'MaxDD%':>8}"
          f" {'ES5%':>7} {'HHI':>7} {'TO%':>6}")
    print("  " + "-" * 100)
    for _, r in df.iterrows():
        print(f"  {r['Strategy']:<28} {r['Ann. Return %']:>7.1f}  {r['Ann. Volatility %']:>7.1f}"
              f"  {r['Ann. Sharpe']:>6.2f}  {r['Max DD %']:>7.1f}"
              f"  {r['ES (5%) %']:>6.2f}  {r['HHI']:>6.4f}  {r['Avg Turnover %']:>5.1f}")
    print("  " + "=" * 100)


# ═══════════════════════════════════════════════════════════
#  STATISTICAL SIGNIFICANCE: Jobson-Korkie (1981) / Memmel (2003)
# ═══════════════════════════════════════════════════════════

def _jobson_korkie_test(ret_a, ret_b):
    """Jobson-Korkie (1981) test with Memmel (2003) correction.

    Tests H0: Sharpe(A) = Sharpe(B).
    Returns (z-statistic, two-sided p-value).
    """
    from scipy.stats import norm
    # Align to common dates
    common = ret_a.index.intersection(ret_b.index)
    ra, rb = ret_a.loc[common], ret_b.loc[common]
    T = len(ra)
    if T < 10:
        return np.nan, np.nan

    mu_a, mu_b = ra.mean(), rb.mean()
    sig_a, sig_b = ra.std(ddof=1), rb.std(ddof=1)
    rho = ra.corr(rb)

    sr_a = mu_a / sig_a
    sr_b = mu_b / sig_b

    # Memmel (2003) corrected variance
    theta = (1.0 / T) * (2 * (1 - rho)
             + 0.5 * (sr_a ** 2 + sr_b ** 2) * (1 - rho ** 2)
             - sr_a * sr_b * (1 - rho) ** 2)

    if theta <= 0:
        return np.nan, np.nan

    z = (sr_a - sr_b) / np.sqrt(theta)
    p = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 3), round(p, 4)


def print_significance_tests(results, strategy_order):
    """Print pairwise Sharpe ratio significance tests."""
    print("\n  Sharpe Ratio Significance Tests (Jobson-Korkie / Memmel correction)")
    print("  " + "=" * 85)
    print(f"  {'Comparison':<45} {'z-stat':>8} {'p-value':>9} {'Sig':>6}")
    print("  " + "-" * 85)

    pairs = []
    strats = list(strategy_order)
    for i in range(len(strats)):
        for j in range(i + 1, len(strats)):
            pairs.append((strats[i], strats[j]))

    for name_a, name_b in pairs:
        ret_a = results[name_a]["portfolio_returns"]
        ret_b = results[name_b]["portfolio_returns"]
        z, p = _jobson_korkie_test(ret_a, ret_b)
        if np.isnan(z):
            sig = "N/A"
        elif p < 0.01:
            sig = "***"
        elif p < 0.05:
            sig = "**"
        elif p < 0.10:
            sig = "*"
        else:
            sig = ""
        label = f"{name_a}  vs  {name_b}"
        print(f"  {label:<45} {z:>8.3f} {p:>9.4f} {sig:>6}")

    print("  " + "=" * 85)
    print("  *** p<0.01  ** p<0.05  * p<0.10")


# ═══════════════════════════════════════════════════════════
#  FIGURE 4a + TABLE 2: OOS Weekly Returns
# ═══════════════════════════════════════════════════════════

def plot_oos_boxplot(results, strategy_order):
    """Side-by-side boxplot of OOS weekly returns."""
    data_list = []
    labels = []
    colors = []
    for name in strategy_order:
        res = results[name]
        oos = res.get("oos_weekly_returns", pd.Series(dtype=float))
        if oos.empty:
            # For Equal Weight, compute OOS returns from portfolio returns directly
            oos = res["portfolio_returns"]
        data_list.append(oos.dropna().values * 100)
        labels.append(DISPLAY_NAMES.get(name, name))
        colors.append(STRATEGY_COLORS[name])

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color(PLOT_TEXT)
        median.set_linewidth(2)
    for whisker in bp["whiskers"]:
        whisker.set_color(PLOT_SPINE)
    for cap in bp["caps"]:
        cap.set_color(PLOT_SPINE)

    ax.axhline(y=0, color=PLOT_SPINE, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Weekly Return (%)", color=PLOT_TEXT, fontsize=22)
    ax.set_title("OOS Weekly Return Distribution",
                 color=PLOT_TEXT, fontsize=28, fontweight="bold", pad=14)
    ax.tick_params(colors=PLOT_TEXT, labelsize=20)
    ax.yaxis.grid(True, color=PLOT_GRID, alpha=0.5, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "4a_oos_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)
    return path


def oos_summary_table(results, strategy_order):
    """OOS weekly return distribution summary for each strategy."""
    rows = []
    for name in strategy_order:
        res = results[name]
        oos = res.get("oos_weekly_returns", pd.Series(dtype=float))
        if oos.empty:
            oos = res["portfolio_returns"]
        stats = oos_return_distribution_stats(oos)
        stats["Strategy"] = name
        rows.append(stats)
    return pd.DataFrame(rows)


def print_oos_table(df):
    """Pretty-print the OOS summary."""
    cols = ["Strategy", "N obs", "Mean %", "Median %", "Std %",
            "Skew", "Hit Rate %", "VaR(5%) %", "CVaR(5%) %"]
    print("\n  OOS Weekly Return Distribution Summary")
    print("  " + "-" * 95)
    header = "  ".join(f"{c:>11}" for c in cols)
    print(f"  {header}")
    print("  " + "-" * 95)
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:>11.3f}")
            elif isinstance(v, (int, np.integer)):
                vals.append(f"{v:>11d}")
            else:
                vals.append(f"{str(v):>11}")
        print("  " + "  ".join(vals))
    print("  " + "-" * 95)


# ═══════════════════════════════════════════════════════════
#  TABLE 3: Crisis Period Performance
# ═══════════════════════════════════════════════════════════

def crisis_table(results, strategy_order):
    """Cumulative return during each crisis period for each strategy."""
    rows = []
    returns_dict = {name: results[name]["portfolio_returns"] for name in strategy_order}
    returns_dict, _ = trim_all_to_common_start(returns_dict)

    for crisis_name, start, end in CRISIS_PERIODS:
        row = {"Crisis": crisis_name, "Period": f"{start} to {end}"}
        for name in strategy_order:
            ret = returns_dict[name]
            crisis_ret = ret.loc[start:end]
            if len(crisis_ret) > 0:
                cum = (1 + crisis_ret).prod() - 1
                row[name] = round(cum * 100, 1)
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_crisis_table(df, strategy_order):
    """Pretty-print the crisis table (dynamic columns from STRATEGY_ORDER)."""
    strats = [s for s in strategy_order if s in df.columns]
    short = {
        "CMVO": "CMVO-MS",
        "Plain MV": "Plain-MS",
        "CMVO (Min Vol)": "CMVO-MV",
        "Plain MV (Min Vol)": "Plain-MV",
        "CMVO (5% Stock Cap)": "CMVO+5%",
        "Equal Weight (1/N)": "1/N",
    }
    header = f"  {'Crisis':<12} {'Period':<23}"
    for s in strats:
        header += f" {short.get(s, s):>9}"
    print(f"\n  Crisis Period Performance (%)")
    print("  " + "=" * len(header))
    print(header)
    print("  " + "-" * len(header))
    for _, row in df.iterrows():
        line = f"  {row['Crisis']:<12} {row['Period']:<23}"
        for s in strats:
            v = row.get(s, np.nan)
            line += f" {v:>9.1f}"
        print(line)
    print("  " + "=" * len(header))


# ═══════════════════════════════════════════════════════════
#  FIGURES 4c/4d: Sector Weight Allocation Over Time
# ═══════════════════════════════════════════════════════════

SECTOR_COLORS = {
    "Communication Services": "#6B8EA7",
    "Consumer Discretionary": "#C2785C",
    "Consumer Staples":       "#D4A754",
    "Energy":                 "#8E6C88",
    "Financials":             "#A3B18A",
    "Health Care":            "#C49A6C",
    "Industrials":            "#7A9EAF",
    "Information Technology": "#5B7553",
    "Materials":              "#B56B4F",
    "Real Estate":            "#8CB369",
    "Utilities":              "#4A7C59",
}


SECTOR_SHORT = {
    "Communication Services": "Comm Svcs", "Consumer Discretionary": "Cons Disc",
    "Consumer Staples": "Cons Stpl", "Energy": "Energy", "Financials": "Financials",
    "Health Care": "Health", "Industrials": "Industrials",
    "Information Technology": "Info Tech", "Materials": "Materials",
    "Real Estate": "Real Est", "Utilities": "Utilities",
}


def _sector_sorted_tickers():
    """Return (sorted tickers, sector boundary tuples)."""
    from config import GICS_SECTOR_MAP
    st = sorted(TICKERS, key=lambda t: (GICS_SECTOR_MAP.get(t, ""), t))
    bounds = []; prev = GICS_SECTOR_MAP.get(st[0], ""); start = 0
    for i, t in enumerate(st):
        s = GICS_SECTOR_MAP.get(t, "")
        if s != prev:
            bounds.append((start, i - 1, SECTOR_SHORT.get(prev, prev)))
            start = i; prev = s
    bounds.append((start, len(st) - 1, SECTOR_SHORT.get(prev, prev)))
    return st, bounds


def _year_xticks(dates):
    pos, lab = [], []; prev = None
    for j, d in enumerate(dates):
        if d.month <= 2 and d.year != prev:
            pos.append(j); lab.append(str(d.year)); prev = d.year
    return pos, lab


def _weights_matrix(weights_schedule, sorted_tickers):
    dates = sorted(weights_schedule.keys())
    mat = np.zeros((len(sorted_tickers), len(dates)))
    for j, d in enumerate(dates):
        w = weights_schedule[d]
        for i, t in enumerate(sorted_tickers):
            mat[i, j] = w.get(t, 0.0)
    return dates, mat


def _agg_sector_weights(ws):
    """Aggregate per-ticker weights into per-sector weights."""
    from config import GICS_SECTOR_MAP
    dates = sorted(ws.keys())
    sectors = sorted(set(GICS_SECTOR_MAP.values()))
    rows = []
    for d in dates:
        w = ws[d]
        row = {"date": d}
        for s in sectors:
            members = [t for t, sec in GICS_SECTOR_MAP.items() if sec == s]
            row[s] = sum(w.get(t, 0.0) for t in members)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _plot_sector_pair(ax1, ax2, sdf1, sdf2, title1, title2, fname):
    """Stacked area for two strategy panels, shared x-axis."""
    sectors = sorted(sdf1.columns)
    fig = ax1.figure
    for ax, sdf, title in [(ax1, sdf1, title1), (ax2, sdf2, title2)]:
        ax.set_facecolor(PLOT_BG)
        bottom = np.zeros(len(sdf))
        for s in sectors:
            vals = sdf[s].values * 100
            ax.fill_between(sdf.index, bottom, bottom + vals,
                            label=s, color=SECTOR_COLORS.get(s, "#888"),
                            alpha=0.85, linewidth=0.3, edgecolor="white")
            bottom += vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("Weight (%)", color=PLOT_TEXT, fontsize=16)
        ax.set_title(title, color=PLOT_TEXT, fontsize=18, fontweight="bold", pad=10)
        ax.tick_params(colors=PLOT_TEXT, labelsize=14)
        ax.yaxis.grid(True, color=PLOT_GRID, alpha=0.4, linestyle=":")
        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.set_xlabel("Year", color=PLOT_TEXT, fontsize=16)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=15,
               framealpha=0.9, edgecolor=PLOT_SPINE, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(top=0.93, bottom=0.20)
    fig.savefig(os.path.join(FIGURES_DIR, fname),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


def plot_sector_weights(results):
    """Combined stacked area: CMVO above, Plain MV below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.12})
    fig.patch.set_facecolor(PLOT_BG)
    _plot_sector_pair(
        ax1, ax2,
        _agg_sector_weights(results["CMVO"]["weights"]),
        _agg_sector_weights(results["Plain MV"]["weights"]),
        "CMVO \u2014 Sector Weight Allocation Over Time",
        "Plain MV \u2014 Sector Weight Allocation Over Time",
        "4b_sector_weights.png",
    )


def _green_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("greens", [
        "#FFFFFF", "#E8F0E4", "#B5D4A8", "#7CB56B", "#5B7553", "#2D4A24"
    ])


def _black_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blacks", [
        "#FFFFFF", "#E0E0E0", "#A0A0A0", "#606060", "#303030", "#000000"
    ])


def _plot_heatmap_pair(panels, dates, bounds, fname, sorted_tickers=None, show_tickers_top=False, show_tickers_bottom=False):
    """Draw two heatmap panels with sector labels. Optionally show ticker names on top/bottom panels.

    panels: list of (matrix, title, vmax, cbar_label) or
            (matrix, title, vmax, cbar_label, cmap_override)
    """
    default_cmap = _black_cmap()
    n_tickers = panels[0][0].shape[0]

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(PLOT_BG)

    ax1 = fig.add_axes([0.13, 0.53, 0.75, 0.40])
    ax2 = fig.add_axes([0.13, 0.06, 0.75, 0.40])
    axes_list = [ax1, ax2]

    for ax, panel in zip(axes_list, panels):
        mat, title, vmax, cblabel = panel[:4]
        cmap = panel[4] if len(panel) > 4 else default_cmap
        ax.set_facecolor(PLOT_BG)
        im = ax.imshow(np.clip(mat * 100, 0, vmax), aspect="auto",
                       interpolation="nearest", cmap=cmap,
                       vmin=0, vmax=vmax, origin="upper")

        # Show ticker names on requested panels
        if sorted_tickers and ((show_tickers_top and ax is axes_list[0]) or
                                (show_tickers_bottom and ax is axes_list[1])):
            ax.set_yticks(range(n_tickers))
            ax.set_yticklabels(sorted_tickers, fontsize=11, color=PLOT_TEXT)
        else:
            ax.set_yticks([])

        # Sector separator lines
        for start, end, short in bounds:
            if start > 0:
                ax.axhline(y=start - 0.5, color=PLOT_SPINE, linewidth=1.0, alpha=0.7)

        ax.set_title(title, color=PLOT_TEXT, fontsize=24, fontweight="bold", pad=12)
        ax.tick_params(colors=PLOT_TEXT, length=4, labelsize=18)
        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE)
        cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85, aspect=20)
        cb.set_label(cblabel, color=PLOT_TEXT, fontsize=18)
        cb.ax.tick_params(colors=PLOT_TEXT, labelsize=16)

        # Sector labels + bracket bars in figure coordinates
        for start, end, short in bounds:
            mid = (start + end) / 2
            y_fig_mid = fig.transFigure.inverted().transform(ax.transData.transform((0, mid)))[1]
            y_fig_s = fig.transFigure.inverted().transform(ax.transData.transform((0, start - 0.3)))[1]
            y_fig_e = fig.transFigure.inverted().transform(ax.transData.transform((0, end + 0.3)))[1]
            fig.text(0.01, y_fig_mid, short, fontsize=18, fontweight="bold", color=PLOT_TEXT,
                     ha="left", va="center")
            fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
                (0.09, y_fig_s), (0.09, y_fig_e),
                transform=fig.transFigure, arrowstyle="-",
                color=PLOT_SPINE, linewidth=2, clip_on=False))

    # X-axis on bottom only
    xpos, xlabels = _year_xticks(dates)
    ax1.set_xticklabels([])
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(xlabels, fontsize=20, color=PLOT_TEXT)
    ax2.set_xlabel("Year", color=PLOT_TEXT, fontsize=22)

    fig.savefig(os.path.join(FIGURES_DIR, fname),
                dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig)


def plot_stock_weight_heatmap(results):
    """Per-stock weight heatmap: Plain MV above, CMVO below. Same scale for both."""
    st, bounds = _sector_sorted_tickers()
    cmvo_dates, cmvo_mat = _weights_matrix(results["CMVO"]["weights"], st)
    _, plain_mat = _weights_matrix(results["Plain MV"]["weights"], st)

    # Use the actual max across both panels so the comparison is fair
    shared_vmax = max(float(plain_mat.max() * 100), float(cmvo_mat.max() * 100))
    shared_vmax = int(np.ceil(shared_vmax / 5) * 5)  # round up to nearest 5

    _plot_heatmap_pair([
        (plain_mat, "Plain MV \u2014 Per-Stock Weight (Unconstrained)", shared_vmax, "Weight (%)"),
        (cmvo_mat, "CMVO \u2014 Per-Stock Weight (30% Cluster Cap)", shared_vmax, "Weight (%)"),
    ], cmvo_dates, bounds, "4c_stock_weights_heatmap.png", sorted_tickers=st,
       show_tickers_top=True, show_tickers_bottom=True)


# ═══════════════════════════════════════════════════════════
#  CONCENTRATION + CAPPED VARIANT PLOTS
# ═══════════════════════════════════════════════════════════

def _active_stocks(weights_schedule, threshold=0.001):
    """Return array of number of stocks held (>threshold) at each rebalance."""
    return np.array([
        sum(1 for v in w.values() if v > threshold)
        for w in [weights_schedule[d] for d in sorted(weights_schedule.keys())]
    ])


def _capped_active_mean(res):
    return _active_stocks(res["weights"]).mean()


def _print_concentration(results, strategy_order):
    """Print concentration summary: median stocks held, median max weight, HHI."""
    print(f"\n  {'Strategy':<26} {'Med Stocks':>11} {'Med MaxWt%':>11} {'Med HHI':>9} {'MaxWt%':>8}")
    print("  " + "-" * 70)
    for name in strategy_order:
        res = results[name]
        if "weights" not in res:
            continue
        ws = res["weights"]
        counts = _active_stocks(ws)
        max_wts = []
        hhis = []
        for d in sorted(ws.keys()):
            w = ws[d]
            vals = np.array(list(w.values()))
            max_wts.append(vals.max() * 100)
            hhis.append(float((vals ** 2).sum()))
        max_wts = np.array(max_wts)
        hhis = np.array(hhis)
        print(f"  {name:<26} {np.median(counts):>10.0f}  {np.median(max_wts):>10.1f}"
              f"  {np.median(hhis):>8.3f}  {max_wts.max():>7.1f}")


def plot_sector_weights_capped(results, capped):
    """Combined stacked area: CMVO (uncapped) above, CMVO (5% Stock Cap) below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.12})
    fig.patch.set_facecolor(PLOT_BG)
    _plot_sector_pair(
        ax1, ax2,
        _agg_sector_weights(results["CMVO"]["weights"]),
        _agg_sector_weights(capped["weights"]),
        "CMVO \u2014 Sector Weights (30% cluster cap, no stock cap)",
        "CMVO \u2014 Sector Weights (30% cluster cap + 5% stock cap)",
        "4d_sector_weights_capped.png",
    )


def plot_stock_weight_heatmap_capped(results, capped):
    """Per-stock weight heatmap: compare uncapped and capped CMVO on a common 0-30% scale."""
    st, bounds = _sector_sorted_tickers()
    cmvo_dates, cmvo_mat = _weights_matrix(results["CMVO"]["weights"], st)
    _, capped_mat = _weights_matrix(capped["weights"], st)

    # Same color scale in both panels so the effect of the stock cap is directly comparable.
    _plot_heatmap_pair([
        (cmvo_mat, "CMVO \u2014 Per-Stock Weight (30% cluster cap, no stock cap)", 30,
         "Weight (%)"),
        (capped_mat, "CMVO \u2014 Per-Stock Weight (30% cluster cap + 5% stock cap)", 30,
         "Weight (%)"),
    ], cmvo_dates, bounds, "4e_stock_weights_capped.png", sorted_tickers=st,
       show_tickers_top=True, show_tickers_bottom=True)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Backtest Results")
    print("=" * 60)

    # ── Run backtests ──
    results, _, weekly_returns = run_all_backtests()

    # ── Table 1: Performance metrics ──
    print("\n[1] Performance Metrics Comparison")
    perf = performance_table(results, STRATEGY_ORDER, weekly_returns=weekly_returns)
    print_performance_table(perf)
    print_significance_tests(results, STRATEGY_ORDER)
    perf.to_csv(os.path.join(DATA_DIR, "4_performance_metrics.csv"), index=False)
    print("     -> data/4_performance_metrics.csv")

    # ── Concentration summary ──
    print("\n[2] Portfolio Concentration")
    _print_concentration(results, STRATEGY_ORDER)

    # ── Figure 4b: Sector weights (CMVO vs Plain MV) ──
    print("\n[5] Sector Weight Allocation (CMVO vs Plain MV) ...")
    plot_sector_weights(results)
    print("     -> figures/4b_sector_weights.png")

    # ── Figure 4c: Per-stock weight heatmap (CMVO vs Plain MV) ──
    print("[6] Per-Stock Weight Heatmap (CMVO vs Plain MV) ...")
    plot_stock_weight_heatmap(results)
    print("     -> figures/4c_stock_weights_heatmap.png")

    # ── Figure 4d: Sector weights (capped variant) ──
    print("[7] Sector Weight Allocation (CMVO vs 5% stock cap) ...")
    plot_sector_weights_capped(results, results["CMVO (5% Stock Cap)"])
    print("     -> figures/4d_sector_weights_capped.png")

    print("\n" + "=" * 60)
    print("  Done. 3 figures + 1 CSV saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()

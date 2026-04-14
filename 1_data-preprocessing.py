"""
1_data-preprocessing.py

S&P 500 universe filtering pipeline.
Starts with all current S&P 500 constituents, applies data quality
filters, and selects top 3 stocks per GICS sector by median daily
dollar volume over a three-month ranking window ending at the freeze date.

Outputs:
  data/closing_prices/         -> individual CSV per S&P 500 stock (Date + Close)
  data/universe_prices.csv     -> combined close prices for the final 33 tickers
  data/ticker_list.csv         -> final 33 selected tickers + sectors
  data/sectors.csv             -> full S&P 500 ticker-to-sector mapping
  data/descriptive_stats.csv   -> summary statistics table
  figures/1b_corr_{daily,weekly,monthly,quarterly}.png -> correlation heatmaps
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    MIN_HISTORY_YEARS, MAX_CONSECUTIVE_GAP, STOCKS_PER_SECTOR,
    DOLLAR_VOL_LOOKBACK_MONTHS, DATA_FREEZE_DATE, RAW_TICKER_DIR, DATA_DIR,
    CLOSING_DIR, FIGURES_DIR,
    PLOT_BG, PLOT_TEXT, PLOT_GRID, PLOT_ACCENT, PLOT_ACCENT2, PLOT_SPINE,
)

# create output folders
for d in [DATA_DIR, CLOSING_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_sp500_list():
    """Load S&P 500 constituents + GICS sectors.
    Primary source: Bloomberg Terminal export (data/bloomberg_sp500_sectors.csv).
    Falls back to Wikipedia scrape if Bloomberg file is missing.
    """
    bb_path = os.path.join(DATA_DIR, "bloomberg_sp500_sectors.csv")
    if os.path.exists(bb_path):
        df = pd.read_csv(bb_path)[["ticker", "sector"]]
        print("  Source: Bloomberg Terminal GICS sector mapping")
        return df

    local = os.path.join(DATA_DIR, "sp500_wikipedia.csv")
    if os.path.exists(local):
        print("  Source: Wikipedia (Bloomberg file not found)")
        return pd.read_csv(local)

    import requests, io
    headers = {"User-Agent": "Mozilla/5.0 (research project)"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.strip()
    df.to_csv(local, index=False)
    return df


def load_ticker_csv(ticker):
    """Load a ticker's OHLCV data from local CSV cache, falling back to yfinance download."""
    for name in [ticker, ticker.replace(".", "-")]:
        path = os.path.join(RAW_TICKER_DIR, f"{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            idx = pd.to_datetime(df.index, utc=True)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None)
            df.index = idx
            return df
    # Fallback: download from Yahoo Finance and cache locally
    try:
        import yfinance as yf
        yf_ticker = ticker.replace(".", "-")
        df = yf.download(yf_ticker, period="max", auto_adjust=True, progress=False)
        if df is not None and len(df) > 0:
            idx = pd.to_datetime(df.index, utc=True)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None)
            df.index = idx
            os.makedirs(RAW_TICKER_DIR, exist_ok=True)
            save_name = ticker.replace(".", "-")
            df.to_csv(os.path.join(RAW_TICKER_DIR, f"{save_name}.csv"))
            return df
    except ImportError:
        pass
    except Exception:
        pass
    return None


def longest_nan_streak(series):
    """Return the length of the longest consecutive NaN run in a series."""
    is_nan = series.isna()
    if not is_nan.any():
        return 0
    groups = (is_nan != is_nan.shift()).cumsum()
    return int(is_nan.groupby(groups).sum().max())


def freeze_timestamp():
    """Return the hardcoded freeze date as a naive calendar timestamp."""
    return pd.Timestamp(DATA_FREEZE_DATE)


# ═══════════════════════════════════════════════════════════
#  PIPELINE STEP FUNCTIONS
# ═══════════════════════════════════════════════════════════

def filter_by_history(ticker_data, min_years):
    """Drop tickers with less than min_years of close-price history."""
    dropped = []
    for t in list(ticker_data.keys()):
        close = ticker_data[t]["Close"].dropna()
        cutoff = freeze_timestamp() - pd.DateOffset(years=min_years)
        if len(close) == 0 or close.index.min() > cutoff:
            dropped.append(t)
            del ticker_data[t]
    return ticker_data, dropped


def filter_by_gaps(ticker_data, max_gap):
    """Drop tickers whose longest consecutive NaN run exceeds max_gap."""
    fails = []
    for t in list(ticker_data.keys()):
        close = ticker_data[t]["Close"]
        first = close.first_valid_index()
        if first is None:
            fails.append((t, "no data"))
            del ticker_data[t]
            continue
        streak = longest_nan_streak(close.loc[first:])
        if streak > max_gap:
            fails.append((t, streak))
            del ticker_data[t]
    return ticker_data, fails


def forward_fill_gaps(ticker_data):
    """Forward-fill any remaining small NaN gaps in close prices."""
    n_filled = 0
    for t in ticker_data:
        col = ticker_data[t]["Close"]
        if col.isna().any():
            ticker_data[t]["Close"] = col.ffill()
            n_filled += 1
    return ticker_data, n_filled


def remove_duplicate_classes(ticker_data):
    """Remove known duplicate share classes, keeping the more liquid one."""
    DUPLICATES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}
    removed = []
    for drop_t, keep_t in DUPLICATES.items():
        if drop_t in ticker_data and keep_t in ticker_data:
            del ticker_data[drop_t]
            removed.append(f"{drop_t} -> keeping {keep_t}")
    return ticker_data, removed


def select_top_per_sector(ticker_data, sector_lookup, n_per_sector, lookback_months):
    """Rank tickers by median daily dollar volume and pick the top n per GICS sector."""
    vol_cutoff = freeze_timestamp() - pd.DateOffset(months=lookback_months)
    sector_rankings = {}
    for t in ticker_data:
        sector = sector_lookup.get(t)
        if sector is None:
            continue
        df = ticker_data[t]
        recent = df.loc[df.index >= vol_cutoff]
        if "Volume" in recent.columns and "Close" in recent.columns:
            dv = (recent["Close"] * recent["Volume"]).dropna()
            med = float(dv.median()) if len(dv) > 0 else 0.0
        else:
            med = 0.0
        sector_rankings.setdefault(sector, []).append((t, med))

    selected = []
    for sector in sorted(sector_rankings):
        ranked = sorted(sector_rankings[sector], key=lambda x: x[1], reverse=True)
        top = ranked[:n_per_sector]
        selected.extend([t for t, _ in top])

    return selected, sector_rankings


# ═══════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════

def build_universe_prices(ticker_data, selected):
    """Combine close prices for the selected tickers into one aligned DataFrame."""
    frames = {}
    for t in selected:
        close = ticker_data[t]["Close"].dropna()
        frames[t] = close

    prices = pd.DataFrame(frames)

    # strip timezone info so resample/pct_change work cleanly
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    else:
        prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)

    # trim to common start (when all 33 tickers have data)
    common_start = prices.dropna().index.min()
    prices = prices.loc[common_start:]
    prices = prices.ffill()  # fill any straggling internal NaNs
    return prices


def compute_descriptive_stats(universe_prices, selected, sector_lookup):
    """Compute summary statistics on daily returns for each ticker.

    Inspired by Capponi & Rubtsov (2021) who report mean, std, skewness,
    and kurtosis for S&P 500 constituents, noting excess kurtosis >> 3
    as motivation for fat-tailed models.
    """
    daily_ret = universe_prices[selected].pct_change().dropna()
    rows = []

    for t in selected:
        r = daily_ret[t]
        rows.append({
            "Ticker": t,
            "Sector": sector_lookup[t],
            "Ann Mean%": round(r.mean() * 252 * 100, 2),
            "Ann Std%": round(r.std() * np.sqrt(252) * 100, 2),
            "Skew": round(r.skew(), 3),
            "Kurt": round(r.kurtosis(), 2),     # excess kurtosis (normal = 0)
            "Min%": round(r.min() * 100, 2),
            "Max%": round(r.max() * 100, 2),
            "Obs": len(r),
        })

    stats_df = pd.DataFrame(rows)
    return stats_df


def print_descriptive_stats(stats_df):
    """Pretty-print the descriptive stats table to the terminal."""
    header = (f"  {'Ticker':<7} {'Sector':<22} {'Mean%':>7} {'Std%':>7} "
              f"{'Skew':>7} {'Kurt':>7} {'Min%':>8} {'Max%':>7}")
    print(header)
    print(f"  {'-'*7} {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

    for _, row in stats_df.iterrows():
        print(f"  {row['Ticker']:<7} {row['Sector']:<22} {row['Ann Mean%']:>7.1f} "
              f"{row['Ann Std%']:>7.1f} {row['Skew']:>7.2f} {row['Kurt']:>7.1f} "
              f"{row['Min%']:>8.1f} {row['Max%']:>7.1f}")

    avg_kurt = stats_df["Kurt"].mean()
    print(f"\n  Avg excess kurtosis: {avg_kurt:.1f}  "
          f"(Capponi & Rubtsov 2021 report ~29.5 for US stocks)")


SECTOR_SHORT = {
    "Communication Services": "Comm Svcs",
    "Consumer Discretionary": "Cons Disc",
    "Consumer Staples":       "Cons Stpl",
    "Energy":                 "Energy",
    "Financials":             "Financials",
    "Health Care":            "Health",
    "Industrials":            "Industrials",
    "Information Technology":  "Info Tech",
    "Materials":              "Materials",
    "Real Estate":            "Real Est",
    "Utilities":              "Utilities",
}


def _sector_bounds(sorted_sel, sector_lookup):
    """Compute (start, end, short_label) for each sector run in sorted_sel."""
    bounds = []
    prev = sector_lookup.get(sorted_sel[0], "")
    start = 0
    for i, t in enumerate(sorted_sel):
        s = sector_lookup.get(t, "")
        if s != prev:
            bounds.append((start, i - 1, SECTOR_SHORT.get(prev, prev)))
            start = i
            prev = s
    bounds.append((start, len(sorted_sel) - 1, SECTOR_SHORT.get(prev, prev)))
    return bounds


def make_correlation_plots(universe_prices, selected, sector_lookup):
    """Save individual correlation heatmaps at daily/weekly/monthly/quarterly.

    Tickers are sorted by GICS sector. Sector labels replace individual ticker names.
    """
    sorted_sel = sorted(selected, key=lambda t: (sector_lookup.get(t, ""), t))
    bounds = _sector_bounds(sorted_sel, sector_lookup)

    daily_ret = universe_prices[sorted_sel].pct_change().dropna()
    weekly_ret = universe_prices[sorted_sel].resample("W-FRI").last().pct_change().dropna()
    monthly_ret = universe_prices[sorted_sel].resample("ME").last().pct_change().dropna()
    quarterly_ret = universe_prices[sorted_sel].resample("QE").last().pct_change().dropna()

    freq_data = [
        ("Daily", daily_ret),
        ("Weekly", weekly_ret),
        ("Monthly", monthly_ret),
        ("Quarterly", quarterly_ret),
    ]

    for label, returns in freq_data:
        corr = returns.corr()
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr = upper.mean()
        n_obs = len(returns)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)

        im = ax.imshow(corr.values, cmap="YlGnBu", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks([])
        ax.set_yticks([])

        # No colorbar — separate colorbar image generated below

        ax.set_title("", pad=0)

        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE)

        plt.tight_layout()
        fname = f"1b_corr_{label.lower()}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname),
                    dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)
        print(f"    {label:<12} n={n_obs:<6} avg corr={avg_corr:+.3f}  -> figures/{fname}")

    # Standalone colorbar image — very tall to span both heatmap rows
    fig_cb, ax_cb = plt.subplots(figsize=(2, 42))
    fig_cb.patch.set_facecolor(PLOT_BG)
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=Normalize(-1, 1))
    cbar = fig_cb.colorbar(sm, cax=ax_cb)
    cbar.set_label("Correlation", color=PLOT_TEXT, fontsize=80, labelpad=20)
    cbar.ax.tick_params(colors=PLOT_TEXT, labelsize=70)
    fig_cb.savefig(os.path.join(FIGURES_DIR, "1b_corr_colorbar.png"),
                   dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
    plt.close(fig_cb)
    print(f"    {'Colorbar':<12} -> figures/1b_corr_colorbar.png")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  S&P 500 Universe Selection")
    print("  Data Preprocessing Pipeline")
    print("=" * 60)

    # Check if outputs already exist — skip the heavy download if so
    csv_path = os.path.join(DATA_DIR, "universe_prices.csv")
    ticker_path = os.path.join(DATA_DIR, "ticker_list.csv")
    if os.path.exists(csv_path) and os.path.exists(ticker_path):
        print("\n  [INFO] universe_prices.csv and ticker_list.csv already exist.")
        print("  To rebuild from scratch, delete them and re-run.")
        print("  Skipping to correlation plots only ...\n")
        import pandas as _pd
        _prices = _pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        _tl = _pd.read_csv(ticker_path)
        _tickers = _tl["ticker"].tolist()
        _sector_lookup = dict(zip(_tl["ticker"], _tl["sector"]))
        make_correlation_plots(_prices, _tickers, _sector_lookup)
        print("\n  Done (correlation plots regenerated).")
        return

    # ── Step 0: S&P 500 constituent list ────────────────────
    print("\n[Step 0] Loading S&P 500 constituents ...")
    sp500 = get_sp500_list()
    N_START = len(sp500)
    sector_lookup = dict(zip(sp500["ticker"], sp500["sector"]))
    print(f"  Total constituents : {N_START}")
    print(f"  GICS sectors       : {sp500['sector'].nunique()}")

    # ── Step 1: Load raw CSVs ───────────────────────────────
    print("\n[Step 1] Loading price/volume data from local CSVs ...")
    ticker_data = {}
    no_csv = []
    for _, row in sp500.iterrows():
        df = load_ticker_csv(row["ticker"])
        if df is not None:
            ticker_data[row["ticker"]] = df
        else:
            no_csv.append(row["ticker"])
    print(f"  Loaded   : {len(ticker_data)}")
    print(f"  No CSV   : {len(no_csv)}  {no_csv}")

    # ── Step 2: >= 20 years of history ──────────────────────
    print(f"\n[Step 2] Filtering for >= {MIN_HISTORY_YEARS} years of history ...")
    ticker_data, too_short = filter_by_history(ticker_data, MIN_HISTORY_YEARS)
    print(f"  Dropped  : {len(too_short)}")
    if too_short:
        print(f"    {too_short[:10]}{'...' if len(too_short) > 10 else ''}")
    print(f"  Remaining: {len(ticker_data)}")

    # ── Step 3: No gap > 5 consecutive trading days ─────────
    print(f"\n[Step 3] Removing stocks with gaps > {MAX_CONSECUTIVE_GAP} consecutive days ...")
    ticker_data, gap_fails = filter_by_gaps(ticker_data, MAX_CONSECUTIVE_GAP)
    print(f"  Dropped  : {len(gap_fails)}")
    for t, g in gap_fails[:5]:
        print(f"    {t}: gap = {g} days")
    if len(gap_fails) > 5:
        print(f"    ... +{len(gap_fails) - 5} more")
    print(f"  Remaining: {len(ticker_data)}")

    # ── Step 4: Forward-fill small gaps ─────────────────────
    print(f"\n[Step 4] Forward-filling remaining small gaps ...")
    ticker_data, n_filled = forward_fill_gaps(ticker_data)
    print(f"  Stocks with gaps filled: {n_filled}")
    print(f"  Remaining: {len(ticker_data)}  (no stocks removed)")

    # ── Step 5: Remove duplicate share classes ──────────────
    print("\n[Step 5] Removing duplicate share classes ...")
    ticker_data, dup_removed = remove_duplicate_classes(ticker_data)
    print(f"  Removed  : {len(dup_removed)}")
    for r in dup_removed:
        print(f"    {r}")
    print(f"  Remaining: {len(ticker_data)}")

    # ── Step 6: Top 3 per sector by dollar volume ───────────
    print(f"\n[Step 6] Selecting top {STOCKS_PER_SECTOR} per sector "
          f"(median daily $ vol, last {DOLLAR_VOL_LOOKBACK_MONTHS}mo) ...")
    selected, rankings = select_top_per_sector(
        ticker_data, sector_lookup, STOCKS_PER_SECTOR, DOLLAR_VOL_LOOKBACK_MONTHS
    )
    print(f"\n  {'Sector':<28} {'Selected':<25} {'Med $/day'}")
    print(f"  {'-'*28} {'-'*25} {'-'*18}")
    for sector in sorted(rankings):
        ranked = sorted(rankings[sector], key=lambda x: x[1], reverse=True)
        top = ranked[:STOCKS_PER_SECTOR]
        tstr = ", ".join(t for t, _ in top)
        vstr = ", ".join(f"${v/1e6:,.0f}M" for _, v in top)
        print(f"  {sector:<28} {tstr:<25} {vstr}")
    print(f"\n  Final universe: {len(selected)} stocks "
          f"({len(rankings)} sectors x {STOCKS_PER_SECTOR})")

    # ── Build universe prices (common-date aligned) ─────────
    print("\n[Step 7] Building universe price matrix ...")
    universe_prices = build_universe_prices(ticker_data, selected)
    print(f"  Shape    : {universe_prices.shape[0]} days x {universe_prices.shape[1]} tickers")
    print(f"  Range    : {universe_prices.index.min().strftime('%Y-%m-%d')} "
          f"to {universe_prices.index.max().strftime('%Y-%m-%d')}")

    # ── Step 8: Descriptive statistics ──────────────────────
    print(f"\n[Step 8] Descriptive statistics (daily returns, annualized)")
    stats_df = compute_descriptive_stats(universe_prices, selected, sector_lookup)
    print_descriptive_stats(stats_df)

    # ── Step 9: Correlation matrices ─────────────────────────
    print(f"\n[Step 9] Saving correlation matrices ...")
    make_correlation_plots(universe_prices, selected, sector_lookup)

    # ═════════════════════════════════════════════════════════
    #  SAVE ALL OUTPUTS
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Saving outputs")
    print("=" * 60)

    # 1) closing prices for all filtered S&P 500 stocks
    print("\n[Save 1] Closing prices -> data/closing_prices/")
    saved = 0
    for t in ticker_data:
        close = ticker_data[t]["Close"].dropna()
        if len(close) == 0:
            continue
        fname = t.replace(".", "-") + ".csv"
        close.to_frame("Close").to_csv(os.path.join(CLOSING_DIR, fname))
        saved += 1
    print(f"  {saved} CSVs written")

    # 2) universe prices (the final 33, aligned)
    universe_prices.to_csv(os.path.join(DATA_DIR, "universe_prices.csv"))
    print(f"[Save 2] universe_prices.csv  -> {universe_prices.shape}")

    # 3) ticker list
    ticker_list_df = pd.DataFrame({
        "ticker": selected,
        "sector": [sector_lookup[t] for t in selected],
    })
    ticker_list_df.to_csv(os.path.join(DATA_DIR, "ticker_list.csv"), index=False)
    print(f"[Save 3] ticker_list.csv      -> {len(selected)} tickers")

    # 4) full sector map
    sp500[["ticker", "sector"]].to_csv(os.path.join(DATA_DIR, "sectors.csv"), index=False)
    print(f"[Save 4] sectors.csv          -> {len(sp500)} rows")

    # 5) descriptive stats
    stats_df.to_csv(os.path.join(DATA_DIR, "descriptive_stats.csv"), index=False)
    print(f"[Save 5] descriptive_stats.csv")

    # ── Done ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Done. {len(selected)} stocks selected.")
    print(f"  Universe prices : data/universe_prices.csv")
    print(f"  Correlation matrices : figures/1b_corr_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

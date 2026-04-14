"""
config.py
All parameters for the research project live here.
"""

import csv
import os

# ── Data filtering parameters ──
MIN_HISTORY_YEARS = 20
MAX_CONSECUTIVE_GAP = 5        # max missing trading days in a row
STOCKS_PER_SECTOR = 3
DOLLAR_VOL_LOOKBACK_MONTHS = 3  # median dollar volume ranking window
DATA_FREEZE_DATE = "2026-04-01"  # matches the manuscript-era universe and logged med-dollar-volume rankings

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
RAW_TICKER_DIR = os.path.join(DATA_DIR, "closing_prices")
CLOSING_DIR = RAW_TICKER_DIR
FIGURES_DIR = os.path.join(BASE, "figures")
CSV_PATH = os.path.join(DATA_DIR, "universe_prices.csv")
TICKER_LIST_PATH = os.path.join(DATA_DIR, "ticker_list.csv")
CLUSTER_COMPOSITION_PATH = os.path.join(DATA_DIR, "2_cluster_compositions_k6.csv")

# ── Clustering parameters ──
LOOKBACK_YEARS = 5
LINKAGE = "ward"
K_RANGE = range(4, 10)   # k = 4, 5, 6, 7, 8, 9

# ── Backtest parameters ──
RF_ANNUAL = 0.0
CRISIS_PERIODS = [
    ("GFC", "2007-10-01", "2009-03-31"),
    ("COVID", "2020-02-01", "2020-04-30"),
    ("2022 Bear", "2022-01-01", "2022-10-31"),
]

# ── Plot style (white background) ──
PLOT_BG = "#FFFFFF"
PLOT_TEXT = "#3B3B3B"
PLOT_GRID = "#D5D5D5"
PLOT_ACCENT = "#5B7553"      
PLOT_ACCENT2 = "#C2785C"      
PLOT_SPINE = "#AAAAAA"

CLUSTER_COLORS = [
    "#5B7553",   # sage green
    "#C2785C",   # terracotta
    "#6B8EA7",   # steel blue
    "#D4A754",   # goldenrod
    "#8E6C88",   # muted plum
    "#A3B18A",   # olive
    "#C49A6C",   # tan
    "#7A9EAF",   # dusty teal
    "#B56B4F",   # rust
    "#8CB369",   # fern
]


def _load_universe():
    """Load the submitted 33-stock universe from ticker_list.csv."""
    if not os.path.exists(TICKER_LIST_PATH):
        return [], {}

    with open(TICKER_LIST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tickers = [row["ticker"] for row in rows]
    sector_map = {row["ticker"]: row["sector"] for row in rows}
    return tickers, sector_map


def _load_fixed_clusters():
    """Load the chosen k=6 partition; fall back to sector buckets if missing."""
    if os.path.exists(CLUSTER_COMPOSITION_PATH):
        with open(CLUSTER_COMPOSITION_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and {"Ticker", "Cluster"} <= set(rows[0].keys()):
            return {row["Ticker"]: row["Cluster"] for row in rows}

    clusters = {}
    for i, sector in enumerate(sorted(set(GICS_SECTOR_MAP.values())), start=1):
        for ticker, ticker_sector in GICS_SECTOR_MAP.items():
            if ticker_sector == sector:
                clusters[ticker] = f"C{i}"
    return clusters


TICKERS, GICS_SECTOR_MAP = _load_universe()

GICS_SECTOR_TICKERS = {}
for _ticker, _sector in GICS_SECTOR_MAP.items():
    GICS_SECTOR_TICKERS.setdefault(_sector, []).append(_ticker)

FIXED_CLUSTERS = _load_fixed_clusters()

# ── Strategy names (used across all scripts and the paper) ──
STRATEGY_NAMES = {
    "cmvo":        "CMVO",
    "plain_mv":    "Plain MV",
    "ew":          "Equal Weight (1/N)",
    "cmvo_minvol": "CMVO (Min Vol)",
    "plain_minvol":"Plain MV (Min Vol)",
    "cmvo_capped": "CMVO (5% stock cap)",
}

BASELINE_CONFIG = {
    "lookback": 5,
    "rebal": "Weekly",
    "k": 6,
    "linkage": "ward",
    "target": "max_sharpe",
    "cov": "sample",
    "cluster_cap": 0.30,
    "cluster_floor": 0.0,
    "stock_cap": 1.0,
    "rf_annual": RF_ANNUAL,
}

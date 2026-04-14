"""
Data loading for the submission universe.
"""

import functools
import pandas as pd
from config import CSV_PATH, TICKERS


@functools.lru_cache(maxsize=1)
def load_data():
    """Load price data and return 20-year windows of daily, weekly, and monthly close prices."""
    data = pd.read_csv(CSV_PATH)
    data["Date"] = pd.to_datetime(data["Date"].str.split(" ").str[0])
    data = data.set_index("Date").sort_index()
    # Handle ticker name mismatches between CSV and ticker_list (e.g. BRK-B vs BRK.B)
    rename = {}
    for t in TICKERS:
        if t not in data.columns:
            alt = t.replace(".", "-")
            if alt in data.columns:
                rename[alt] = t
    if rename:
        data = data.rename(columns=rename)
    end = data.index.max()
    start = end - pd.DateOffset(years=20)
    close = data.loc[start:end, TICKERS].ffill()
    daily_close = close
    weekly_close = close.resample("W-FRI").last()
    monthly_close = close.resample("ME").last()
    return daily_close, weekly_close, monthly_close

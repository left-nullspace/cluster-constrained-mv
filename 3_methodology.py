"""
3_methodology.py

Visualizes the backtest methodology — no results, just the design.
Produces:
  figures/3a_rolling_window.png  -> walk-forward window architecture (VBT Pro Splitter)
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
    DATA_DIR, FIGURES_DIR, LOOKBACK_YEARS,
    PLOT_BG, PLOT_TEXT,
)

os.makedirs(FIGURES_DIR, exist_ok=True)

TRAIN_COLOR = "#3F6F93"   # deep slate blue
OOS_COLOR = "#D97A5B"     # muted terracotta


def _actual_backtest_window(weekly_prices):
    """Return the realized OOS start/end and split count used by the engine."""
    weekly_returns = weekly_prices.pct_change().dropna()
    if weekly_returns.empty:
        return None, None, 0

    min_obs = LOOKBACK_YEARS * 52
    rebal_dates = weekly_returns.index[1:]
    first_rebal = None
    valid_rebals = []

    for rd in rebal_dates:
        window_start = rd - pd.DateOffset(years=LOOKBACK_YEARS)
        wr = weekly_returns.loc[window_start:rd]
        if len(wr) >= min_obs:
            valid_rebals.append(rd)
            if first_rebal is None:
                first_rebal = rd

    if not valid_rebals:
        return None, None, 0

    first_oos = weekly_returns.index[weekly_returns.index > first_rebal][0]
    last_oos = weekly_returns.index.max()
    return first_oos, last_oos, len(valid_rebals)


# ═══════════════════════════════════════════════════════════
#  FIGURE 1: Rolling Walk-Forward Window Architecture
# ═══════════════════════════════════════════════════════════

def plot_rolling_windows():
    """Show the rolling walk-forward window design using VBT Pro's Splitter."""
    import vectorbtpro as vbt

    prices = pd.read_csv(os.path.join(DATA_DIR, "universe_prices.csv"),
                         parse_dates=["Date"], index_col="Date")
    weekly = prices.resample("W-FRI").last()

    # Real splitter: 5-year estimation + 1-week OOS, rolling weekly
    real_train = LOOKBACK_YEARS * 52   # 260 weeks
    real_test = 1
    real_splitter = vbt.Splitter.from_rolling(
        weekly.index,
        length=real_train + real_test,
        split=-real_test,
        offset=-real_test,
        offset_anchor_set=None,
        backwards="sorted",
        set_labels=["Estimation Window (5 yr)", "1 Week"],
    )
    first_oos, last_oos, actual_splits = _actual_backtest_window(weekly)

    # Stylized splitter for visualisation (not to scale so OOS is visible)
    vis_train, vis_test = 40, 5
    short_index = weekly.index[: vis_train + vis_test + 2 * vis_test + 5]
    vis_splitter = vbt.Splitter.from_rolling(
        short_index,
        length=vis_train + vis_test,
        split=-vis_test,
        offset=-vis_test,
        offset_anchor_set=None,
        backwards="sorted",
        set_labels=["Estimation Window (5 yr)", "1 Week"],
    )

    # Build a simple static matrix for the first 3 stylized splits:
    # 0 = empty, 1 = train, 2 = OOS.
    split_rows = []
    for i, mask_df in enumerate(vis_splitter.get_iter_split_masks()):
        if i >= 3:
            break
        row = np.zeros(len(short_index), dtype=int)
        row[mask_df["Estimation Window (5 yr)"].values] = 1
        row[mask_df["1 Week"].values] = 2
        split_rows.append(row)
    split_matrix = np.vstack(split_rows)

    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([PLOT_BG, TRAIN_COLOR, OOS_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    FIG_BG = "#FBFAF7"
    FIG_TEXT = "#2F2F2F"
    FIG_SPINE = "#B8B8B8"

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.imshow(split_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xlim(-0.5, split_matrix.shape[1] - 0.5)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Window 1", "Window 2", "Window 3"], fontsize=16, color=FIG_TEXT)
    ax.set_ylabel("", fontsize=1)
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_title("Rolling Walk-Forward Design",
                 fontsize=20, color=FIG_TEXT, fontweight="bold", pad=12)

    # Inline labels for the first split to explain the stylized windows.
    first_row = split_matrix[0]
    train_idx = np.where(first_row == 1)[0]
    test_idx = np.where(first_row == 2)[0]
    if len(train_idx) > 0:
        train_center = (train_idx[0] + train_idx[-1]) / 2
        ax.text(train_center, 0, "Estimation Window (5 yr)", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")
    if len(test_idx) > 0:
        test_center = (test_idx[0] + test_idx[-1]) / 2
        ax.text(test_center, 0, "1 Week", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")

    fig.text(
        0.015,
        0.01,
        f"Real OOS sample: {first_oos.strftime('%Y-%m-%d')} to {last_oos.strftime('%Y-%m-%d')}",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#888888",
    )

    for spine in ax.spines.values():
        spine.set_color(FIG_SPINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(FIGURES_DIR, "3a_rolling_window.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Methodology Visuals")
    print("=" * 60)

    try:
        import vectorbtpro as vbt
    except ImportError:
        print("\n  [SKIP] vectorbtpro not installed.")
        print("  Skipped: figures/3a_rolling_window.png")
        return

    print("\n[1] Rolling walk-forward windows ...")
    plot_rolling_windows()
    print("  -> figures/3a_rolling_window.png")

    print("\n" + "=" * 60)
    print("  Done. 1 figure saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
methodology_diagrams.py

Generates the two walk-forward methodology loop diagrams:
  figures/3b_mv_methodology_loop.png   -> plain MV (5 steps)
  figures/3c_cmvo_methodology_loop.png -> CMVO with clustering highlighted (6 steps)

Run from submission/:
  python utils/methodology_diagrams.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, FIGURES_DIR, LOOKBACK_YEARS

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette ──
BG    = "#FFFFFF"
TEXT  = "#3B3B3B"
SAGE  = "#5B7553"
TERRA = "#C2785C"
STEEL = "#6B8EA7"
GOLD  = "#D4A754"
PLUM  = "#8E6C88"
SPINE = "#AAAAAA"


def _actual_window_summary():
    """Return ('YYYY-YYYY, N splits', 'YYYY-MM-DD', 'YYYY-MM-DD', N).

    Uses the engine's data source (sp500_closes.csv via data_loader) to match
    the actual backtest OOS window exactly.
    """
    try:
        from data_loader import load_data
        from config import BASELINE_CONFIG
        from engine import run_rolling_backtest

        daily_close, _, _ = load_data()
        start_str = str(daily_close.index.min().date())
        bc = BASELINE_CONFIG

        res = run_rolling_backtest(
            start_str, bc["lookback"], bc["rebal"], bc["k"],
            bc["linkage"], bc["target"], bc["cov"],
            cluster_cap=bc["cluster_cap"], cluster_floor=bc["cluster_floor"],
            stock_cap=bc["stock_cap"], rf_annual=bc["rf_annual"],
        )
        ret = res["portfolio_returns"]
        n = len(res["weights"])
        first_oos = ret.index[0]
        last_oos = ret.index[-1]
        summary = f"{first_oos.year}\u2013{last_oos.year}, {n} splits"
        return summary, first_oos.strftime("%Y-%m-%d"), last_oos.strftime("%Y-%m-%d"), n
    except Exception:
        return "2011\u20132026, 784 splits", "2011-02-18", "2026-02-13", 784


def draw_loop(ax, steps, title, subtitle, highlight_idx=None):
    """Draw a circular methodology loop on the given axes.

    Parameters
    ----------
    steps : list of (label, color)
    highlight_idx : int or None
        Index of the step to emphasise with a yellow glow.
    """
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.axis("off")

    ax.text(5, 10.3, title, fontsize=32, fontweight="bold",
            ha="center", va="center", color=TEXT)
    ax.text(5, 9.55, subtitle, fontsize=20, ha="center", va="center", color=SPINE)

    n = len(steps)
    cx, cy = 5, 4.5
    rx, ry = 3.8, 3.3
    angles = [90 - i * (360 / n) for i in range(n)]
    box_w, box_h = 2.8, 1.3

    positions = []
    for angle in angles:
        rad = np.radians(angle)
        positions.append((cx + rx * np.cos(rad), cy + ry * np.sin(rad)))

    # ── Boxes ──
    for i, ((x, y), (label, color)) in enumerate(zip(positions, steps)):
        is_hl = (highlight_idx is not None and i == highlight_idx)

        if is_hl:
            for pad, a in [(0.28, 0.04), (0.20, 0.07), (0.12, 0.12)]:
                glow = FancyBboxPatch(
                    (x - box_w / 2 - pad, y - box_h / 2 - pad),
                    box_w + 2 * pad, box_h + 2 * pad,
                    boxstyle="round,pad=0.15",
                    facecolor="#FFD700", alpha=a, edgecolor="none",
                )
                ax.add_patch(glow)

        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor="#FFD700" if is_hl else color,
            alpha=0.25 if is_hl else 0.18,
            edgecolor="#DAA520" if is_hl else color,
            linewidth=5 if is_hl else 2,
        )
        ax.add_patch(box)

        ax.text(x, y, label, fontsize=18, fontweight="bold",
                ha="center", va="center", color=TEXT)
        ax.text(x - box_w / 2 + 0.2, y + box_h / 2 - 0.18, str(i + 1),
                fontsize=16, fontweight="bold",
                color="#DAA520" if is_hl else color,
                alpha=0.9, ha="center", va="center")

    # ── Arrows (routed outside the circle, clamped to box edges) ──
    def _box_edge(bx, by, dirx, diry, bw, bh):
        tx = (bw / 2) / abs(dirx) if abs(dirx) > 1e-6 else 1e6
        ty = (bh / 2) / abs(diry) if abs(diry) > 1e-6 else 1e6
        t = min(tx, ty)
        return bx + dirx * t, by + diry * t

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = positions[i]
        x2, y2 = positions[j]

        dx, dy = x2 - x1, y2 - y1
        d = np.sqrt(dx ** 2 + dy ** 2)
        ux, uy = dx / d, dy / d

        ox1 = (x1 - cx); oy1 = (y1 - cy)
        on1 = np.sqrt(ox1 ** 2 + oy1 ** 2); ox1 /= on1; oy1 /= on1
        ox2 = (x2 - cx); oy2 = (y2 - cy)
        on2 = np.sqrt(ox2 ** 2 + oy2 ** 2); ox2 /= on2; oy2 /= on2

        s_dir_x = 0.4 * ox1 + 0.6 * ux
        s_dir_y = 0.4 * oy1 + 0.6 * uy
        sn = np.sqrt(s_dir_x ** 2 + s_dir_y ** 2); s_dir_x /= sn; s_dir_y /= sn
        e_dir_x = 0.4 * ox2 - 0.6 * ux
        e_dir_y = 0.4 * oy2 - 0.6 * uy
        en = np.sqrt(e_dir_x ** 2 + e_dir_y ** 2); e_dir_x /= en; e_dir_y /= en

        sx, sy = _box_edge(x1, y1, s_dir_x, s_dir_y, box_w, box_h)
        ex, ey = _box_edge(x2, y2, e_dir_x, e_dir_y, box_w, box_h)

        ax.add_patch(FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle="->,head_width=8,head_length=6",
            color=TEXT, linewidth=2, alpha=0.5,
            connectionstyle="arc3,rad=-0.3",
        ))

    ax.text(cx, cy, f"\u00d7{split_count}\nsplits", fontsize=28, fontweight="bold",
            ha="center", va="center", color=TEXT, alpha=0.25)


# Module-level variable set by main() before draw_loop is called
split_count = 784  # default, overridden in main()


def main():
    global split_count
    period_summary, _, _, split_count = _actual_window_summary()

    print("=" * 60)
    print("  Methodology Diagrams")
    print("=" * 60)

    print("\n[1] CMVO methodology loop ...")
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_loop(ax, [
        ("Estimation Window\n(5-yr weekly returns)", SAGE),
        ("Correlation-Distance\nMatrix", STEEL),
        ("Hierarchical Clustering\n(Ward, k = 6)", GOLD),
        ("Mean-Variance\nOptimisation\n(cluster-constrained)", TERRA),
        ("Hold 1 Week\n(out-of-sample)", PLUM),
        ("Slide Forward\n& Collect Return", SPINE),
    ], "CMVO Methodology",
       f"Cluster-constrained mean-variance optimisation ({period_summary})",
       highlight_idx=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "3b_cmvo_methodology_loop.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  -> figures/3b_cmvo_methodology_loop.png")

    print("\n" + "=" * 60)
    print("  Done. 1 diagram saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()

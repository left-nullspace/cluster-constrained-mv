# SCI999 Project: Cluster-Constrained Mean-Variance Optimization

This repository contains the code and data for the SCI999 research project, which examines whether cluster-based diversification constraints can improve the out-of-sample performance of a standard mean-variance portfolio.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```
   python run_all.py
   ```
   This executes all 7 scripts in order and saves a timestamped log to `output_logs/`.

3. To run individual scripts:
   ```
   python 1_data-preprocessing.py
   python 2_cluster_analysis.py
   python 3_methodology.py
   python 4_results.py
   python 5_sensitivity.py
   python 6_cpcv.py
   python 7_nco.py
   ```

## Data Sources and Freeze Date

Both input data files are **included in this repository** so that all results can be reproduced exactly as submitted, without re-downloading or re-exporting anything.

- **`data/universe_prices.csv`** (3.2 MB) -- Daily adjusted close prices for the 33 selected S&P 500 stocks, originally derived from Yahoo Finance. This is the frozen price matrix used by scripts 2--7. It is included in the repo and should not be regenerated unless you are intentionally rebuilding the universe from scratch.

- **`data/bloomberg_sp500_sectors.csv`** (13 KB) -- GICS sector classifications for all S&P 500 constituents, exported from a Bloomberg Terminal in April 2026. This file maps each ticker to its GICS sector and is used by `1_data-preprocessing.py` during universe selection. It is included in the repo because Bloomberg data cannot be re-downloaded programmatically.

**Freeze date:** The universe was frozen as of **1 April 2026** (`DATA_FREEZE_DATE` in `config.py`). The three-month median dollar-volume ranking window ends in February 2026, and the available price history ends on 13 February 2026. These two CSV files represent the exact data snapshot used in the paper.

## Repository Structure

```
submission/
  config.py              Configuration and parameters
  data_loader.py         Loads the 33-stock price matrix
  engine.py              Clustering, optimization, and backtest engine
  metrics.py             Performance and distributional metrics
  run_all.py             Master runner script (runs all 7 scripts)
  1_data-preprocessing.py   Universe selection and descriptive statistics
  2_cluster_analysis.py     Cluster diagnostics and selection of k=6
  3_methodology.py          Walk-forward window visualization (requires vectorbtpro)
  4_results.py              Main backtest results and figures
  5_sensitivity.py          Sensitivity analysis across hyper-parameters
  6_cpcv.py                 Combinatorial Purged Cross-Validation (requires vectorbtpro)
  7_nco.py                  Nested Clustered Optimization comparison (requires vectorbtpro)
  utils/
    methodology_diagrams.py   CMVO workflow diagram generation
  data/                    Input data and script output CSVs
  figures/                 Generated figures referenced in the paper
```

## Reproducibility Notes

- **Scripts 3, 6, and 7 require `vectorbtpro`.** If it is unavailable, `run_all.py` will skip those steps and print a message. All other scripts run with standard open-source packages only.
- Script 1 requires either pre-existing price CSVs in `data/` or the `yfinance` package to download them. The submitted repository includes `data/universe_prices.csv` so scripts 2-7 can run immediately.

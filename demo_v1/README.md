# Variance Risk Premium Trading Strategy — Demo

> Demo version of the codebase for my Honours thesis. Uses trimmed data for portability. For full numerical reproduction, contact michael431329585@gmail.com.

## Requirements

```
joblib
matplotlib
numpy
pandas
plotly
scikit-learn
scipy
statsmodels
```

## Data

| Dataset | Description | Source | Location |
|---------|-------------|--------|----------|
| EOD Options | First 10 trading days of 2016-01 (trimmed for size; 2016 chosen because earlier data is much smaller) | ORATS | `data/2016/` |
| Underlying prices | EOD underlying data | ORATS | `data/300_underlyings.parquet` |
| Stock splits | Corporate action adjustments | ORATS | `data/stock_splits.csv` |
| Ticker universe | Precomputed from options volume | Precomputed | `data/ticker_universe.xlsx` |
| Precomputed signals | 2020–2025 (full range is 2010–2025, ~300 MB) | Precomputed | `vrp_signals_PRECOMPUTED/` |
| Precomputed strategy PnLs | 2010–2025 monthly trade results | Precomputed | `backtest_results_PRECOMPUTED/` |
| Pretrained models | Trained on full 2010–2024 data | Precomputed | `models_PRETRAINED/` |

All scripts use `BASE_DIR = Path(__file__).resolve().parent`, so paths are relative to the script directory. No path changes are needed as long as the directory structure is preserved.

## Pipeline

The scripts should be run in the following order:

### 1. Fetch Data (not in demo)

`fetch_underlying.py` and `fetch_stock_split.py` require an API key. Pre-fetched data is provided in `data/`.

### 2. Process Underlying

Run `process_underlying.py`. Reads `data/300_underlyings.parquet` and outputs `data/300_underlyings_processed.parquet`. Computes underlying-based signals that feed into the signal aggregation step.

### 3. Compute Ticker Universe (read-only)

`get_tickers_READ_ONLY.py` is included for reference. It requires the full options dataset to run. The precomputed universe is in `data/ticker_universe.xlsx`.

### 4. Compute VRP Signals

Run `vrp_signal.py`. It reads the 10-day demo options data from `data/2016/`, computes variance risk premium and related signals for all stocks, and outputs `signals_2016-01` in `vrp_signals/`. In the full backtest, this produces signals for every trading day across the entire sample period.

### 5. Run Backtest

Run `backtest.py`. It reads the same 10-day demo options data from `data/2016/`, simulates short straddle strategies for all stocks, and outputs strategy PnLs for 2016-01 in `backtest_results/`. 10 trading days is enough to see the backtest open and close 1-week straddle positions. In the full backtest, this produces PnLs for every trading day across the entire sample period.

The two scripts `vrp_signal.py` and `backtest.py` share a very similar code structure. They create new directories (`vrp_signals/` and `backtest_results/`) and write new files there for viewing purposes; the precomputed files in `vrp_signals_PRECOMPUTED/` and `backtest_results_PRECOMPUTED/` are not affected. The signals from step 4 and the PnLs from step 5 are later combined in the analysis notebooks to identify which signal-based filters produce the best strategies.

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `vrp_analysis.ipynb` | Theoretical VRP of stocks (uses precomputed signals 2020–2025) |
| `backtest_analysis.ipynb` | Combines raw returns with signals; applies filtering and ML models. Trained models are saved to `models/` |
| `backtest_analysis_2025.ipynb` | Out-of-sample test on 2025 data using pretrained models (results match thesis exactly) |

Note: `models/` contains models trained by the demo notebooks (on limited data). `models_PRETRAINED/` contains the actual models trained on the full 2010–2024 dataset, which are used by `backtest_analysis_2025.ipynb`.

## Utility Modules

These modules are imported by the pipeline scripts and notebooks — they are not run directly.

| Module | Description |
|--------|-------------|
| `options_reader.py` | Shared data loader for daily ORATS options zip files. Used by both `vrp_signal.py` and `backtest.py` |
| `vrp_signal_utils.py` | Signal computation logic (Black-Scholes, model-free IV, interpolation) |
| `backtest_utils.py` | Backtest helpers (straddle simulation, position management, date iteration, DTE filtering) |
| `analysis_utils.py` | Analysis helpers (data loading, VRP summaries, PnL distributions, ML evaluation, equity curves) |
| `helper_code/` | Additional utilities: `vol_surface_utils.py` (option chain slicing), `plot_utils.py` (straddle payoff diagrams), `view_parquet.py` (parquet-to-CSV conversion), and exploratory notebooks |

## Output Directories

| Directory | Description |
|-----------|-------------|
| `vrp_signals/` | Signals computed by `vrp_signal.py` (demo output) |
| `backtest_results/` | Strategy PnLs computed by `backtest.py` (demo output) |
| `filtered_trades/` | Filtered trade results from the analysis notebooks (ML and manual filters) |
| `tabs_and_figs/` | Thesis figures and summary tables generated by the analysis notebooks |
| `debug/` | Sample options data for manual inspection |

## Reproducibility Note

Intermediate results from the demo may differ from the thesis due to the trimmed data range (10 days vs. full 2010–2025). In particular, ML models trained on the demo's limited precomputed signals (2020–2025) will be weaker than those trained on the full dataset. The demo is intended to demonstrate code functionality. The 2025 out-of-sample results in `backtest_analysis_2025.ipynb` use pretrained models on full data and are exact.

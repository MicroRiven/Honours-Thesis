"""Main driver script for computing daily VRP (Variance Risk Premium) signals.

Iterates over trading dates, loads options + underlying data, computes signals
via vrp_signal_utils, and writes monthly parquet/csv files.
"""

import pandas as pd
import time
from pathlib import Path
import logging

from options_reader import read_options_data
from backtest_utils import iter_trading_dates, load_tickers
from vrp_signal_utils import compute_signal_daily

BASE_DIR = Path(__file__).resolve().parent
COLS = ["yte", "iRate", "divRate"]  # extra columns to load from options data


def run_vrp(
    start_date: str,
    end_date: str,
    out_dir: Path,
    volume_threshold: int = 10_000,
    ticker_threshold: int = 1000,
):
    """Compute VRP signals for every trading day in [start_date, end_date].

    Results are flushed to disk month-by-month as parquet + csv.
    start_date/end_date accept both "YYYY-MM" and "YYYY-MM-DD" formats.
    """
    
    # --- Load reference data ---
    tickers_dict = load_tickers(
        BASE_DIR / "data" / "ticker_universe.xlsx",
        start_year=2010,
        end_year=2025,
        volume_threshold=volume_threshold)

    underlying_path = BASE_DIR / "data" / "300_underlyings_processed.parquet"
    underlying = pd.read_parquet(underlying_path)
    dates = iter_trading_dates(underlying, start_date, end_date)
    underlying_grouped = underlying.groupby("trade_date")
    # NOTE: no need to filter underlying by tickers because there's only 300 tickers in total, we won't save much memory

    # --- Day-by-day loop, flushing results each month ---
    all_signals = []
    current_month = None

    def flush_month(month_key: str):
        """Write accumulated signals for one calendar month to disk and clear buffer."""
        if all_signals:
            signals_df = pd.concat(all_signals, ignore_index=True)
        else:
            signals_df = pd.DataFrame()
        signals_df.to_parquet(out_dir / f"signals_{month_key}.parquet", index=False)
        signals_df.to_csv(out_dir / f"signals_{month_key}.csv", index=False)
        logging.info(f"Flushed results for month {month_key}: {len(signals_df)} signals.")
        all_signals.clear()

    for dt in dates:
        t0 = time.perf_counter()

        # Detect month boundary and flush previous month
        month_key = dt.strftime("%Y-%m")  # e.g. "2024-02"
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            flush_month(current_month)
            current_month = month_key

        # Skip dates with no underlying data
        if dt not in underlying_grouped.groups:
            logging.info(f"Underlying data for {dt.date()} not found, skipping.")
            continue
        underlying_today = underlying_grouped.get_group(dt)
        
        tickers = tickers_dict[dt.year][:ticker_threshold]

        # Load options for this trading day
        options_today = read_options_data(dt, tickers=tickers, cols=COLS)
        if options_today.empty:
            continue  # warning already logged in read_options_data

        # SPX is required as market reference
        if "SPX" not in options_today["ticker"].unique():
            logging.warning(f"SPX not in options data for {dt.date()}, skipping signal computation.")
            continue

        t1 = time.perf_counter()
        signals_today = compute_signal_daily(tickers, underlying_today, options_today, target_dte=30)
        all_signals.append(signals_today)
        t2 = time.perf_counter()

        logging.info(f"Date: {dt.date()} | Load: {t1-t0:.2f}s | Compute: {t2-t1:.2f}s")

    # Flush the last remaining month
    if current_month is not None:
        flush_month(current_month)


if __name__ == "__main__":

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name).11s] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("vrp_signal_log.txt", mode="w", encoding="utf-8"),
            logging.StreamHandler(open(1, "w", encoding="utf-8", closefd=False))
        ]
    )
    logging.getLogger("compute_signal").setLevel(logging.WARNING)  # Only show warnings and errors from compute_signal

    t0 = time.perf_counter() # start timer

    out_dir = BASE_DIR / "vrp_signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_month = "2016-01"
    end_month   = "2016-01"

    logging.info(f"Computing signals from {start_month} to {end_month}")
    logging.info("=====================================================")

    run_vrp(start_date=start_month, end_date=end_month, out_dir=out_dir, volume_threshold=10_000)

    elapsed = time.perf_counter() - t0  # stop timer
    logging.info("=====================================================")
    logging.info("Signal computation completed")
    logging.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logging.info(f"Date range: {start_month} to {end_month}")
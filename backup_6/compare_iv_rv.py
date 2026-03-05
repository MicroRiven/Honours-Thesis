import pandas as pd
import time
from pathlib import Path

import logging

from helper_code.options_reader import read_options_data
from compute_signal     import compute_signal_daily
from compute_iv         import compute_iv

BASE_DIR = Path(__file__).resolve().parent
COLS =  ["ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "yte", "strike",
         "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol", "iRate", "divRate",]
# COLS = ["cVolu", "pVolu"]


def _iter_trading_dates(
    underlying_df: pd.DataFrame,
    start: str,
    end: str
) -> list[pd.Timestamp]:

    dates = pd.Index(underlying_df["trade_date"].unique()).sort_values()
    start_dt = pd.to_datetime(start).normalize()
    end_dt   = (pd.to_datetime(end) + pd.offsets.MonthEnd(0)).normalize()
    dates = dates[(dates >= start_dt) & (dates <= end_dt)]
    return list(dates)   # list of Timestamps


def run_vrp(
    tickers: list[str],
    start_date: str,
    end_date: str,
    out_dir: Path,
):
    '''
    start_date and end_date works for both month strings "YYYY-MM" or full dates "YYYY-MM-DD"
    '''
        
    # ---- load and upsert underlying once ----
    underlying_path = BASE_DIR / "data" / "300_underlyings_processed.parquet"
    underlying = pd.read_parquet(underlying_path)
    dates = _iter_trading_dates(underlying, start_date, end_date) # this is a list of Timestamps, which is not consistent with trade_date column dtype (Datetime64[ns])
    
    # Filter tickers once
    underlying = underlying[underlying["ticker"].isin(tickers)]
    underlying_grouped = underlying.groupby("trade_date")

    all_signals = []
    all_ivs = []
    current_month = None

    def flush_month(month_key: str):
        if all_signals:
            signals_df = pd.concat(all_signals, ignore_index=True)
        else:
            signals_df = pd.DataFrame()
        if all_ivs:
            ivs_df = pd.concat(all_ivs, ignore_index=True)
        else:
            ivs_df = pd.DataFrame()
        signals_df.to_parquet(out_dir / f"signals_{month_key}.parquet", index=False)
        ivs_df.to_parquet(out_dir / f"ivs_{month_key}.parquet", index=False)
        logging.info(f"Flushed results for month {month_key}: {len(signals_df)} signals, {len(ivs_df)} ivs.")
        all_signals.clear()
        all_ivs.clear()

    for dt in dates:

        print(f"Backtesting date: {dt.date()}", end=" | ")
        t0 = time.perf_counter()

        # if month changes, flush previous month first
        month_key = dt.strftime("%Y-%m") # e.g., "2024-02"
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            flush_month(current_month)
            current_month = month_key

        # Fast lookup
        if dt not in underlying_grouped.groups:
            logging.info(f"Underlying data for {dt.date()} not found, skipping.")
            continue
        underlying_today = underlying_grouped.get_group(dt)
        
        # Load options for the day
        y = dt.year
        ymd = dt.strftime("%Y%m%d")
        day_zip = BASE_DIR / "data" / str(y) / f"ORATS_SMV_Strikes_{ymd}.zip"
        if not day_zip.exists():
            logging.info(f"Options data for {dt.date()} not found at {day_zip}, skipping.")
            continue  # missing day file
        options_today = read_options_data(day_zip, tickers, cols=COLS) # NOTE: the time this takes is directly related to the number of columns we read
        if options_today.empty:
            continue

        t1 = time.perf_counter()
        print(f"{t1 - t0:.2f}s", end=" | ")

        signals_today = compute_signal_daily(underlying_today, options_today, tickers)
        all_signals.append(signals_today)

        t2 = time.perf_counter()
        print(f"{t2 - t1:.2f}s", end=" | ")

        iv_today = compute_iv(options_today, tickers, target_dte=30)
        all_ivs.append(iv_today)

        t3 = time.perf_counter()
        print(f"{t3 - t2:.2f}s")

    # flush last month
    if current_month is not None:
        flush_month(current_month)


if __name__ == "__main__":

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name).11s] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("backtest_log.txt", mode="w"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("compute_iv").setLevel(logging.INFO)  # Only show warnings and errors from compute_iv
    logging.getLogger("compute_signal").setLevel(logging.INFO)  # Only show warnings and errors from compute_signal


    t0 = time.perf_counter() # start timer

    out_dir = BASE_DIR / "vrp_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    start_month = "2020-12"
    end_month   = "2021-02"
    # NOTE: META has ticker change to META in 2022-06-09
    # NOTE: Some trades may span month-end boundaries, add buffers

    logging.info(f"Starting backtest from {start_month} to {end_month}")
    logging.info("=====================================================")

    tickers_df = pd.read_excel(BASE_DIR / "data" / "ticker_universe.xlsx", sheet_name="final tickers (random)")
    tickers = tickers_df["ticker"].tolist()
    tickers = tickers[:30]  # limit to 30 tickers for now, can adjust as needed
    print(tickers)

    run_vrp(start_date=start_month, end_date=end_month, out_dir=out_dir, tickers=tickers)

    elapsed = time.perf_counter() - t0  # stop timer
    logging.info(f"Backtest completed in {elapsed:.2f}s ({elapsed/60:.2f} min)\n")
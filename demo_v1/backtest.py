import pandas as pd
import time
import json
from pathlib import Path
import logging
from collections import Counter, defaultdict


from options_reader import read_options_data
from backtest_utils import (
    load_tickers,
    iter_trading_dates,
    open_positions_for_day,
    close_positions_for_day,
    Position
)

BASE_DIR = Path(__file__).resolve().parent
COLS = ["cVolu", "pVolu", "delta"]

# option specific signals like volume and spread?



def run_backtest(
    start_date: str,
    end_date: str,
    out_dir: Path,
    volume_threshold: int = 10_000,
):
    '''
    start_date and end_date works for both month strings "YYYY-MM" or full dates "YYYY-MM-DD"
    '''

    # get dates
    underlying_path = BASE_DIR / "data" / "300_underlyings_processed.parquet"
    underlying = pd.read_parquet(underlying_path)
    dates = iter_trading_dates(underlying, start_date, end_date) # this is a list of Timestamps, which is not consistent with trade_date column dtype (Datetime64[ns])
    
    # get tickers for the date range
    tickers_dict = load_tickers(BASE_DIR / "data" / "ticker_universe.xlsx", start_year=2010, end_year=2025, volume_threshold=volume_threshold)
    # pprint(tickers_dict, compact=True, width=140, sort_dicts=False)
    # exit(1)

    # get stock split informaiton
    stock_split_df = pd.read_csv(BASE_DIR / "data" / "stock_splits.csv")

    # month flush set up
    trades_buf: list[dict] = []
    open_positions: list[Position] = []
    current_month = None
    monthly_stats = defaultdict(Counter)
    def flush_month(month_key: str):
        if trades_buf:
            pd.DataFrame(trades_buf).to_parquet(out_dir / f"trades_{month_key}.parquet", index=False)
            pd.DataFrame(trades_buf).to_csv(out_dir / f"trades_{month_key}.csv", index=False)
        trades_buf.clear()
        # Log monthly stats
        month_stats = dict(monthly_stats.get(month_key, {}))
        pretty_stats = json.dumps(month_stats, indent=2, sort_keys=True)
        logging.info(f"Monthly stats {month_key}:\n{pretty_stats}")
    
    # BIG LOOP
    for dt in dates:

        t0 = time.perf_counter()

        # if month changes, flush previous month first
        month_key = dt.strftime("%Y-%m") # e.g., "2024-02"
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            flush_month(current_month)
            current_month = month_key

        t1 = time.perf_counter()
        time_setup = t1 - t0

        # get tickers for the day (union with prior year to avoid missing closes on year change)
        # [] is the default value if the year key is not found
        tickers_last_year = tickers_dict.get(dt.year - 1, [])
        tickers_this_year = tickers_dict.get(dt.year, [])
        tickers = list(set(tickers_this_year) | set(tickers_last_year))

        # Load options for the day
        options_today = read_options_data(dt, tickers=tickers, max_dte = 100, cols=COLS) # NOTE: the time this takes is directly related to the number of columns we read
        if options_today.empty:
            continue

        t2 = time.perf_counter()
        time_load = t2 - t1

        # close due positions
        open_positions, closed_trades = close_positions_for_day(
            dt,
            options_today,
            open_positions,
            stock_split_df=stock_split_df,
            monthly_stats=monthly_stats,
        )
        trades_buf.extend(closed_trades)

        t3 = time.perf_counter()
        time_close = t3 - t2

        # open new positions 
        new_positions = open_positions_for_day(
            dt,
            options_today,
            tickers_this_year,
            monthly_stats=monthly_stats,
        )
        open_positions.extend(new_positions)

        t4 = time.perf_counter()
        time_open = t4 - t3
        
        # Log all timing info on one line
        logging.info(
            f"Processed {dt.date()} | {len(tickers)} tickers | setup: {time_setup:.2f}s | load: {time_load:.2f}s | close: {time_close:.2f}s | open: {time_open:.2f}s | total: {t4-t0:.2f}s"
        )

    # flush last month
    if current_month is not None:
        flush_month(current_month)

    logging.info(f"Backtest complete.")
    logging.info(f"Remaining open position: {len(open_positions)}")
    return


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name).11s] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("backtest_log.txt", mode="w"),
            logging.StreamHandler()
        ]
    )

    t0 = time.perf_counter() # start timer

    out_dir = BASE_DIR / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: if something happened in the middle and have to restart, manually remove the buffer
    start_month = "2016-01"
    end_month   = "2016-01"

    logging.info(f"Starting backtest from {start_month} to {end_month}")
    logging.info("=====================================================")

    run_backtest(
        start_date=start_month,
        end_date=end_month,
        out_dir=out_dir,
        volume_threshold=10_000,
    )

    elapsed = time.perf_counter() - t0  # stop timer
    logging.info(f"Backtest completed in {elapsed:.2f}s ({elapsed/60:.2f} min)\n")
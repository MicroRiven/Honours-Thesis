import pandas as pd
import time
from pathlib import Path

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name).11s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("backtest_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)
logging.getLogger("get_tickers").setLevel(logging.WARNING)  # Only show warnings and errors from get_tickers
logging.getLogger("fetch_underlying").setLevel(logging.WARNING)  # Only show warnings and errors from fetch_underlying

from helper_code.options_reader import read_options_data
from get_tickers        import get_universe_for_date
from fetch_underlying   import upsert_underlying_data
from compute_signal     import compute_signal_daily
from compute_pnl        import open_positions_for_day, close_positions_for_day, Position

BASE_DIR = Path(__file__).resolve().parent
COLS = ["ticker", "stkPx", "trade_date", "expirDate", "strike", "cValue", "pValue", "smoothSmvVol"]
# COLS = ["cVolu", "pVolu"]



def load_tickers(path: str | Path) -> list[str]:
    with open(path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

def _iter_trading_dates(underlying_df: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    d = underlying_df["trade_date"]
    dates = pd.Index(d.unique()).sort_values()
    
    # Convert month strings (e.g., "2024-02") to first/last day of month
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.offsets.MonthEnd(0)  # Last day of the month
    
    dates = dates[(dates >= start_dt) & (dates <= end_dt)]
    return [ts.to_pydatetime() for ts in dates]


def run_backtest(
    start_date: str,
    end_date: str,
    out_dir: Path,
    n_tickers = 100
):
    '''
    start_date and end_date works for both month strings "YYYY-MM" or full dates "YYYY-MM-DD"
    '''
        
    # ---- load and upsert underlying once ----
    underlying_path = BASE_DIR / "data" / "all_underlyings.parquet"
    underlying = pd.read_parquet(underlying_path)
    dates = _iter_trading_dates(underlying, start_date, end_date) # this is a list of Timestamps, which is not consistent with trade_date column dtype (Datetime64[ns])

    trades_buf: list[dict] = []
    open_positions: list[Position] = []
    current_month = None

    def flush_month(month_key: str):
        if trades_buf:
            pd.DataFrame(trades_buf).to_parquet(out_dir / f"trades_{month_key}.parquet", index=False)
            pd.DataFrame(trades_buf).to_csv(out_dir / f"trades_{month_key}.csv", index=False)
        logging.info(f"Flushed results for month {month_key}: {len(trades_buf)} closed trades.")
        trades_buf.clear()

    for dt in dates:

        print(f"Backtesting date: {dt.date()}", end=" | ")
        t0 = time.perf_counter()

        # Get ticker for today
        tickers = get_universe_for_date(dt, n_tickers=n_tickers)
        underlying = upsert_underlying_data(underlying, tickers, underlying_path) # update underlying and ticker

        t1 = time.perf_counter()
        print(f"{t1 - t0:.2f}s", end=" | ")

        # if month changes, flush previous month first
        month_key = dt.strftime("%Y-%m") # e.g., "2024-02"
        if current_month is None:
            current_month = month_key
        elif month_key != current_month:
            flush_month(current_month)
            current_month = month_key

        # Load underlying slice for the day
        underlying_today = underlying.loc[underlying["trade_date"].eq(dt)]
        if underlying_today.empty:
            continue
        # Load options for the day
        y = dt.year
        ymd = dt.strftime("%Y%m%d")
        day_zip = BASE_DIR / "data" / str(y) / f"ORATS_SMV_Strikes_{ymd}.zip"
        if not day_zip.exists():
            logging.warning(f"Options data for {dt.date()} not found at {day_zip}, skipping.")
            continue  # missing day file
        options_today = read_options_data(day_zip, tickers, cols=COLS) # NOTE: the time this takes is directly related to the number of columns we read
        if options_today.empty:
            continue

        t2 = time.perf_counter()
        print(f"{t2 - t1:.2f}s", end=" | ")

        # close due positions
        open_positions, closed_trades = close_positions_for_day(dt, underlying, options_today, open_positions)
        trades_buf.extend(closed_trades)

        t3 = time.perf_counter()
        print(f"{t3 - t2:.2f}s", end=" | ")

        # signals for today
        # we no longer need to write this to a file separately, instead, we append the signals to each closed trade
        signals_today = compute_signal_daily(underlying_today, options_today, tickers)

        # open new positions 
        new_positions = open_positions_for_day(dt, underlying_today, options_today, tickers, signals_today)
        open_positions.extend(new_positions)

        t4 = time.perf_counter()
        print(f"{t4 - t3:.2f}s")

    # flush last month
    if current_month is not None:
        flush_month(current_month)


    logging.info(f"Backtest complete.")
    logging.info(f"Remaining open position: {len(open_positions)}")
    return


if __name__ == "__main__":

    t0 = time.perf_counter() # start timer

    out_dir = BASE_DIR / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_month = "2023-01"
    end_month   = "2023-12"
    n_tickers = 150
    # NOTE: META has ticker change to META in 2022-06-09
    # NOTE: Some trades may span month-end boundaries, add buffers

    logging.info(f"Starting backtest from {start_month} to {end_month}")
    logging.info("=====================================================")

    run_backtest(
        start_date=start_month,
        end_date=end_month,
        out_dir=out_dir,
        n_tickers=n_tickers,
    )

    elapsed = time.perf_counter() - t0  # stop timer
    logging.info(f"Backtest completed in {elapsed:.2f}s ({elapsed/60:.2f} min)\n")
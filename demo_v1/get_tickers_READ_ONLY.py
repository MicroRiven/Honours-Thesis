"""Build and maintain the ticker universe for backtesting.

Selects tickers by option volume over sampled trading dates and writes
yearly sheets to the ticker_universe Excel file.
"""

import numpy as np
import pandas as pd
import time
from options_reader import read_options_data

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "ticker_universe.xlsx"


def compute_option_volume(
    options_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily option volume (calls + puts) per ticker."""
    options_df = options_df.copy()
    options_df["contract_volume"] = (
        options_df["cVolu"].fillna(0) + options_df["pVolu"].fillna(0)
    )

    volu = (
        options_df
        .groupby("ticker", as_index=False)["contract_volume"]
        .sum()
        .rename(columns={"contract_volume": "todays_volume"})
    )

    return volu


def tickers_available_on_date(date, ticker_ranges) -> list[str]:
    """Return tickers whose data range covers the given date."""
    dt = pd.to_datetime(date)
    mask = (ticker_ranges["min"] <= dt) & (dt <= ticker_ranges["max"])
    return ticker_ranges.loc[mask, "ticker"].tolist()


def get_tickers_for_date(date: str) -> pd.DataFrame:
    """Load available tickers for a date and rank them by daily option volume."""
    dt = pd.to_datetime(date)

    # NOTE: we only keep tickers that are currently trading
    # logger.debug("Reading ticker ranges from cached Excel file...")
    ticker_ranges = pd.read_excel(FILE_PATH, sheet_name="ticker_range")
    available_today = tickers_available_on_date(dt, ticker_ranges)

    # Compute option volume for the day
    # logger.debug("Computing option volume from option data...")
    option_df = read_options_data(dt, tickers=available_today)  # already filtered by ticker
    volu = compute_option_volume(option_df)

    volu = volu.sort_values("todays_volume", ascending=False)
    return volu


def _trailing_dates(
    underlying_df: pd.DataFrame,
    as_of: str | pd.Timestamp,
    window: int = 30,
) -> list[pd.Timestamp]:
    """Return the last `window` trading dates on or before `as_of`."""
    dates = pd.Index(underlying_df["trade_date"].unique()).sort_values()
    as_of_dt = pd.to_datetime(as_of).normalize()
    dates = dates[dates <= as_of_dt]
    return list(dates[-window:])  # last `window` trading dates


def _random_dates(
    underlying_df: pd.DataFrame,
    n_dates: int = 30,
    start_date: str | pd.Timestamp = None,
    end_date: str | pd.Timestamp = None,
) -> list[pd.Timestamp]:
    """Sample `n_dates` random trading dates from the given range."""
    # Slice by date range if provided
    if start_date:
        start_dt = pd.to_datetime(start_date).normalize()
        underlying_df = underlying_df[underlying_df["trade_date"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date).normalize()
        underlying_df = underlying_df[underlying_df["trade_date"] <= end_dt]

    dates = pd.Index(underlying_df["trade_date"].unique()).sort_values()
    picked = np.random.choice(dates.to_numpy(), size=n_dates, replace=False)
    return [pd.Timestamp(d).normalize() for d in picked]


def simple_universe():
    """Select a fixed ticker universe via intersection over random dates (original VRP method)."""
    
    underlying = pd.read_parquet(BASE_DIR / "data" / "all_underlyings.parquet")
    msft_underlying = underlying[underlying["ticker"] == "MSFT"]  # MSFT used to get trading-date calendar

    window = 50
    n_tickers = 500

    # dates = _trailing_dates(msft_underlying, as_of="2025-12-30", window=window)
    dates = _random_dates(msft_underlying, n_dates=window, start_date="2020-01-01", end_date="2025-12-30")

    # Collect per-date volume rankings
    volu_dfs = []
    for date in dates:
        logger.debug(f"Processing date: {date.date()}")
        volu = get_tickers_for_date(date)
        logger.debug(volu.head())
        volu_dfs.append(volu)

    # Keep only tickers present on every sampled date
    intersection = []
    for ticker in volu_dfs[0]["ticker"]:
        if all(ticker in df["ticker"].values for df in volu_dfs[1:]):
            intersection.append(
                {
                    "ticker": ticker,
                    "avg30_option_volume": np.mean([df[df["ticker"] == ticker]["todays_volume"].values[0] for df in volu_dfs])
                }
            )

    intersection_df = pd.DataFrame(intersection)
    intersection_df = intersection_df.sort_values("avg30_option_volume", ascending=False)

    with pd.ExcelWriter(FILE_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        # intersection_df.to_excel(writer, sheet_name="final tickers (trailing)", index=False)
        intersection_df.to_excel(writer, sheet_name="final tickers (random)", index=False)



def backtest_universe():
    """Build per-year ticker universes ranked by average option volume."""
    underlying = pd.read_parquet(BASE_DIR / "data" / "300_underlyings_processed.parquet")
    msft_underlying = underlying[underlying["ticker"] == "MSFT"]  # MSFT used for trading-date calendar (SPX too frequent)
    available_tickers = set(underlying["ticker"].unique())

    # Parameters
    start_year = 2010
    end_year = 2025
    window = 50
    assert window > 1

    for year in range(start_year, end_year + 1):  # 2010-2025 inclusive
        
        t0 = time.perf_counter() # start timer
        logger.info(f"Processing year: {year}")

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        dates = _random_dates(msft_underlying, n_dates=window, start_date=start_date, end_date=end_date)

        # Collect volume data across sampled dates
        volu_dfs = []
        for date in dates:
            logger.info(f"Processing date: {date.date()}")
            volu = get_tickers_for_date(date)
            volu_dfs.append(volu)

        # Average option volume across the window per ticker
        combined = pd.concat(volu_dfs, keys=range(len(volu_dfs)))
        all_tickers = combined['ticker'].unique()
        avg_volume = (
            combined.groupby('ticker')['todays_volume'].sum() / window
        ).sort_values(ascending=False)
        avg_volume_df = avg_volume.to_frame(name='avg_option_volume').reset_index()
        print(avg_volume_df.head())

        with pd.ExcelWriter(FILE_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            avg_volume_df.to_excel(writer, sheet_name=f"{year}", index=False)

        elapsed = time.perf_counter() - t0
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    backtest_universe()

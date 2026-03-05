import requests
from pathlib import Path
import numpy as np
import pandas as pd
import time
import logging

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "ticker_universe.xlsx"
ORATS_TOKEN = "4c49de0a-cc9b-4b65-9ab3-15743195f9a0"
url = "https://api.orats.io/datav2/tickers"

# Create a logger for this module (won't configure output by default)
logger = logging.getLogger(__name__)

def fetch_ticker_ranges():
    sess = requests.Session()

    resp = sess.get(url, params={"token": ORATS_TOKEN})
    resp.raise_for_status()

    data = resp.json()['data'] # list of dicts

    df = pd.DataFrame(data)

    # Optional: ensure date columns are parsed as dates
    df["min"] = pd.to_datetime(df["min"], errors="coerce")
    df["max"] = pd.to_datetime(df["max"], errors="coerce")

    return df


def compute_option_volume(
    options_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: ticker, total_volume
    """
    options_df = options_df.copy()
    options_df["total_contract_volume"] = (
        options_df["cVolu"].fillna(0) + options_df["pVolu"].fillna(0)
    )

    volu = (
        options_df
        .groupby("ticker", as_index=False)
        .agg(total_volume=("total_contract_volume", "sum"))
    )
    return volu


def tickers_available_on_date(date, ticker_ranges) -> set[str]:
    date = pd.to_datetime(date)
    mask = (ticker_ranges["min"] <= date) & (date <= ticker_ranges["max"])
    return set(ticker_ranges.loc[mask, "ticker"])


def get_universe_for_date(date: str, n_tickers = 300, list_or_df = "list") -> list[str] | pd.DataFrame:

    date = pd.to_datetime(date)
    if not FILE_PATH.exists():
        logger.info("New excel file created for ticker universe caching.")
        pd.DataFrame().to_excel(FILE_PATH, index=False)

    sheets = pd.ExcelFile(FILE_PATH).sheet_names

    # if csv exist, read csv, else fetch and save
    # NOTE: we only keep tickers that are currently trading
    if "ticker_range" in sheets:
        logger.info("Reading ticker ranges from cached Excel file...")
        ticker_ranges = pd.read_excel(FILE_PATH, sheet_name="ticker_range")
    else:
        logger.info("Fetching ticker ranges from ORATS API...")
        ticker_ranges = fetch_ticker_ranges()
        ticker_ranges["min"] = pd.to_datetime(ticker_ranges["min"], errors="coerce")
        ticker_ranges["max"] = pd.to_datetime(ticker_ranges["max"], errors="coerce")
        ticker_ranges = ticker_ranges[ticker_ranges["max"] > pd.to_datetime("2025-12-20")] # only keep tickers that are still trading
        with pd.ExcelWriter(FILE_PATH, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            ticker_ranges.to_excel(writer, sheet_name="ticker_range", index=False)
    
    unwanted_list = [
        "F", "VXX", "UVXY", "SMH" # Changed strike
        "ARM", "RUT", # ORATS data issue
        "WULF", "UNG", "SQQQ", "CHPT",  # notional issue
        "FXI", # no common strikes
        ] 
    ticker_ranges_adjusted = ticker_ranges[~ticker_ranges["ticker"].isin(unwanted_list)].copy()
    available_today = tickers_available_on_date(date, ticker_ranges_adjusted)

    # compute option volume for the day
    logger.info("Computing option volume from option data...")
    y = date.year
    ymd = date.strftime("%Y%m%d")
    option_path = BASE_DIR / "data" / str(y) / f"ORATS_SMV_Strikes_{ymd}.zip"
    options_df = pd.read_csv(option_path, compression="zip")
    volu = compute_option_volume(options_df)
    volu = volu[volu["ticker"].isin(available_today)]
    volu = volu.sort_values("total_volume", ascending=False).head(n_tickers)
    return volu["ticker"].tolist() if list_or_df == "list" else volu


def _trailing_dates(
    underlying_df: pd.DataFrame,
    as_of: str | pd.Timestamp,
    window: int = 30,
) -> list[pd.Timestamp]:
    
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
    
    # slice by date range if provided
    if start_date:
        start_dt = pd.to_datetime(start_date).normalize()
        underlying_df = underlying_df[underlying_df["trade_date"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date).normalize()
        underlying_df = underlying_df[underlying_df["trade_date"] <= end_dt]

    dates = pd.Index(underlying_df["trade_date"].unique()).sort_values()
    picked = np.random.choice(dates.to_numpy(), size=n_dates, replace=False)
    return [pd.Timestamp(d).normalize() for d in picked]


if __name__ == "__main__":
    
    # Configure logging only when running this file directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = time.perf_counter() # start timer

    BASE_DIR = Path(__file__).resolve().parent
    underlying = pd.read_parquet(BASE_DIR / "data" / "all_underlyings.parquet")
    tsla_underlying = underlying[underlying["ticker"] == "TSLA"]

    window = 50
    n_tickers = 500

    # we use a specific stock because apparently SPX trade on public holidays
    # dates = _trailing_dates(tsla_underlying, as_of="2025-12-30", window=window)
    dates = _random_dates(tsla_underlying, n_dates=window, start_date="2020-01-01", end_date="2025-12-30")

    # iterate the past 50 days
    volu_dfs = []
    for date in dates:
        logger.info(f"Processing date: {date.date()}")
        volu = get_universe_for_date(date, n_tickers=n_tickers, list_or_df="df")
        logger.info(volu.head())
        volu_dfs.append(volu)

    # compute intersection of 500 tickers across all 50 days
    intersection = []
    for ticker in volu_dfs[0]["ticker"]:
        if all(ticker in df["ticker"].values for df in volu_dfs[1:]): # if ticker is in all dataframes
            intersection.append(
                {
                    "ticker": ticker,
                    "avg30_option_volume": np.mean([df[df["ticker"] == ticker]["total_volume"].values[0] for df in volu_dfs])
                }
            )

    intersection_df = pd.DataFrame(intersection)
    intersection_df = intersection_df.sort_values("avg30_option_volume", ascending=False)
    print(intersection_df)

    with pd.ExcelWriter(FILE_PATH, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        # intersection_df.to_excel(writer, sheet_name="final tickers (trailing)", index=False)
        intersection_df.to_excel(writer, sheet_name="final tickers (random)", index=False)

    elapsed = time.perf_counter() - t0  # stop timer
    logger.info(f"Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
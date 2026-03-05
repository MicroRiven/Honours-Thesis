import requests
from pathlib import Path
import pandas as pd
import time
import logging

BASE_DIR = Path(__file__).resolve().parent
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


def tickers_available_on_date(trade_date, ticker_ranges) -> set[str]:
    trade_date = pd.to_datetime(trade_date)
    mask = (ticker_ranges["min"] <= trade_date) & (trade_date <= ticker_ranges["max"])
    return set(ticker_ranges.loc[mask, "ticker"])


def get_universe_for_date(trade_date: str, n_tickers = 300) -> list[str]:

    trade_date = pd.to_datetime(trade_date)
    excel_path = BASE_DIR / "data" / "ticker_universe.xlsx"
    if not excel_path.exists():
        print("New excel file created for ticker universe caching.")
        pd.DataFrame().to_excel(excel_path, index=False)

    sheets = pd.ExcelFile(excel_path).sheet_names

    # if csv exist, read csv, else fetch and save
    # NOTE: we only keep tickers that are currently trading
    if "ticker_range" in sheets:
        logger.info("Reading ticker ranges from cached CSV...")
        ticker_ranges = pd.read_excel(excel_path, sheet_name="ticker_range")
    else:
        logger.info("Fetching ticker ranges from ORATS API...")
        ticker_ranges = fetch_ticker_ranges()
        ticker_ranges["min"] = pd.to_datetime(ticker_ranges["min"], errors="coerce")
        ticker_ranges["max"] = pd.to_datetime(ticker_ranges["max"], errors="coerce")
        ticker_ranges = ticker_ranges[ticker_ranges["max"] > pd.to_datetime("2025-12-20")] # only keep tickers that are still trading
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            ticker_ranges.to_excel(writer, sheet_name="ticker_range", index=False)

    
    available_today = tickers_available_on_date(trade_date, ticker_ranges)

    # if csv exist, read csv, else compute and save
    if "option_volume" in sheets:
        logger.info("Reading option volume from cached CSV...")
        volu = pd.read_excel(excel_path, sheet_name="option_volume")
    else:
        logger.info("Computing option volume from option data...")
        option_path = BASE_DIR / "data" / "2024" / "ORATS_SMV_Strikes_20240102.zip" # any date will do
        options_df = pd.read_csv(option_path, compression="zip")
        volu = compute_option_volume(options_df)
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            volu.to_excel(writer, sheet_name="option_volume", index=False)

    volu = volu[volu["ticker"].isin(available_today)]
    
    universe = (
        volu.sort_values("total_volume", ascending=False)
           .head(n_tickers)["ticker"]
           .tolist()
    )
    return universe


if __name__ == "__main__":
    
    # Configure logging only when running this file directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = time.perf_counter() # start timer

    # NOTE: The option data date is just an example, we look at what tickers are trading
    # The trade_date can be before or after it, doesn't matter
    trade_date = "2023-06-01"
    tickers = get_universe_for_date(trade_date, n_tickers=300)
    print(f"Top {len(tickers)} tickers by option volume on {trade_date}:")
    print(tickers)

    elapsed = time.perf_counter() - t0  # stop timer
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
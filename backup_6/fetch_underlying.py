from __future__ import annotations
from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import time
import random
import logging
from typing import Any, Dict
import requests

import logging
logger = logging.getLogger(__name__)


ORATS_BASE = "https://api.orats.io/datav2"  # REST base for /hist/dailies
ORATS_TOKEN = "4c49de0a-cc9b-4b65-9ab3-15743195f9a0"
BASE_DIR = Path(__file__).resolve().parent
REQUEST_TIMEOUT = 30              # Seconds for each HTTP request


class OratsError(Exception):
    """Custom exception for ORATS API issues (HTTP errors, schema mismatches, etc.)."""
    pass


def _get(
    endpoint: str,
    params: Dict[str, Any],
    *,
    max_attempts: int = 6,
    empty_on_404: bool = True,
    session: requests.Session | None = None,
) -> Dict[str, Any]:
    """HTTP GET with retries, exponential backoff, and jitter.

    - Appends the ORATS token to the query string.
    - Returns JSON as a dict; ensures a top-level "data" list is present.
    - Treats 404 as empty data when `empty_on_404=True` (handy for sparse tickers).
    - Retries 429 and 5xx with jittered backoff.
    """
    url = f"{ORATS_BASE}{endpoint}"
    params = {**params, "token": ORATS_TOKEN}

    sess = session or requests.Session()
    backoff = 0.5  # seconds (doubles per retry)

    for attempt in range(max_attempts):
        r = sess.get(url, params=params, timeout=REQUEST_TIMEOUT)

        # Fast-path: empty payload for missing resources
        if r.status_code == 404 and empty_on_404:
            return {"data": []}

        # Retryable conditions: rate limit or transient server issues
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == max_attempts - 1:
                raise OratsError(f"ORATS API error {r.status_code}: {r.text[:300]}")
            time.sleep(backoff + random.random() * 0.25)  # jitter helps herd-thundering
            backoff *= 2
            continue

        # Non-success that's not retryable
        if r.status_code != 200:
            raise OratsError(f"ORATS API error {r.status_code}: {r.text[:300]}")

        # Parse and validate schema
        try:
            js = r.json()
        except Exception as e:
            raise OratsError(f"JSON decode error: {e}") from e

        if "data" not in js or not isinstance(js["data"], list):
            raise OratsError(f"Unexpected ORATS response: {js}")
        return js  # {'data': [...]} expected

    # Defensive: we should have returned by now
    raise OratsError("Exhausted retries without a response.")


def fetch_underlying(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV for the requested range and write metrics to Excel.

    Steps:
      1) Fetch as few API calls as possible (auto‑split when capped).
      2) Rename columns to conventional OHLCV names; sort and de‑dupe by date.
      3) Compute avg_volume_30 and Yang–Zhang RV30.
      4) Save to `file_path` (Excel) and return the DataFrame.
    """

    with requests.Session() as sess:
        params = {
            "ticker": ticker,
            "tradeDate": f"{start_date},{end_date}",
        }
        js = _get("/hist/dailies", params, max_attempts=6, empty_on_404=True, session=sess)
        rows = js.get("data", [])
        underlying_df = pd.DataFrame(rows)

    if underlying_df.empty:
        logger.warning(f"No underlying data fetched for {ticker} in range {start_date} to {end_date}.")
        return underlying_df

    # Normalize column names and types
    underlying_df["tradeDate"] = pd.to_datetime(underlying_df["tradeDate"]).dt.normalize()  # ensure datetime dtype
    underlying_df = underlying_df.rename(
        columns={
            "clsPx": "close",
            "hiPx": "high",
            "loPx": "low",
            "stockVolume": "volume",
            "tradeDate": "trade_date",
        }
    )

    return underlying_df


def upsert_underlying_data(existing_df, tickers, underlying_path):
    
    excel_path = BASE_DIR / "data" / "ticker_universe.xlsx"
    ticker_ranges = pd.read_excel(excel_path, sheet_name="ticker_range")

    existing_tickers = set(existing_df["ticker"].unique()) # set of str

    underlying_dfs = []
    for ticker in tickers:
        
        if ticker in existing_tickers:
            continue
        else:
            logger.info(f"[{ticker}] not in existing underlying data, fetching...")
            row = ticker_ranges[ticker_ranges["ticker"] == ticker]
            start_date = row["min"].iloc[0]
            end_date = row["max"].iloc[0] # DO NOT USE values[0], it gives numpy datetime64, which causes issues later
            # to keep data size manageable, limit to post-2015 data
            if start_date < pd.to_datetime("2015-01-01"):
                start_date = pd.to_datetime("2015-01-01")
            underlying_df = fetch_underlying(ticker=ticker, start_date=start_date, end_date=end_date)
            if underlying_df.empty:
                continue
            underlying_dfs.append(underlying_df)

    if underlying_dfs == []:
        logger.info("No new underlying data to fetch.")
        return existing_df  # Return the existing data unchanged
    else:
        # Combine all underlying data into a single parquet file
        new_df = pd.concat(underlying_dfs, ignore_index=True)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True) if underlying_path.exists() else new_df
        combined_df.to_parquet(underlying_path, index=False)

        return combined_df


def get_tickers_list(excel_path):
    trailing = pd.read_excel(excel_path, sheet_name="final tickers (trailing)")
    random = pd.read_excel(excel_path, sheet_name="final tickers (random)")
    # verticlally stack to dataframe, then get unique tickers as a list
    df = pd.concat([trailing, random], ignore_index=True)
    tickers = df["ticker"].unique().tolist()
    return tickers


# This downloads all the underlying data from a fixed list of tickers
if __name__ == "__main__":
    
    underlying_path = BASE_DIR / "data" / "300_underlyings.parquet"
    excel_path = BASE_DIR / "data" / "ticker_universe.xlsx"
    ticker_ranges = pd.read_excel(excel_path, sheet_name="ticker_range")

    tickers = get_tickers_list(excel_path)
    print(tickers)
    print(len(tickers))

    dfs = []
    for ticker in tickers:
        print(f"Fetching underlying data for {ticker}...")

        row = ticker_ranges[ticker_ranges["ticker"] == ticker]
        start_date = row["min"].iloc[0]
        end_date = row["max"].iloc[0]

        underlying_df = fetch_underlying(ticker=ticker, start_date=start_date, end_date=end_date)
        dfs.append(underlying_df)

    all_underlying = pd.concat(dfs, ignore_index=True)
    all_underlying.to_parquet(underlying_path, index=False)
    print(f"Saved all underlying data to {underlying_path}.")
        






    
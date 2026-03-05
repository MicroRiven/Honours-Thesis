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
TICKERS = ['SPY', 'QQQ', 'SPX', 'TSLA', 'AAPL', 'IWM', 'NVDA', 'VIX', 'AMD', 'AMZN', 'MARA', 'MSFT', 'PFE', 'BABA', 'META', 'NIO', 'CSCO', 'TLT', 'TQQQ', 'C', 'COIN', 'SQQQ', 'PLTR', 'RIVN', 'HYG', 'SOFI', 'BAC', 'T', 'INTC', 'SOXL', 'GOOGL', 'UBER', 'RIOT', 'PYPL', 'KWEB', 'MRNA', 'AMC', 'CVNA', 'XOM', 'SNAP', 'FXI', 'NFLX', 'SLV', 'CCL', 'F', 'SQ', 'UVXY', 'EWZ', 'GLD', 'Z', 'EFA', 'ARKK', 'KRE', 'BA', 'DIS', 'ETRN', 'VZ', 'XLE', 'GOOG', 'NKE', 'FSR', 'LYFT', 'XSP', 'GDX', 'XBI', 'MU', 'AFRM', 'BITO', 'EEM', 'JD', 'XLF', 'SE', 'HOOD', 'VXX', 'DKNG', 'NKLA', 'WMT', 'OXY', 'CHWY', 'SHOP', 'KO', 'WBA', 'AI', 'MRK', 'WFC', 'JPM', 'IEF', 'SAVE', 'SOXS', 'AAL', 'PLUG', 'SBUX', 'NCLH', 'TNA', 'GME', 'MO', 'LVS', 'TLRY', 'UPST', 'ORCL', 'ABBV', 'JBLU', 'ZIM', 'MSTR', 'BMY', 'QCOM', 'ONON', 'SMH', 'RUT', 'USO', 'CELH', 'ARM', 'JNJ', 'RBLX', 'TSM', 'LI', 'DIA', 'PDD', 'GM', 'X', 'MPW', 'CVS', 'LCID', 'CYTK', 'CLSK', 'BITF', 'ROKU', 'WYNN', 'LQD', 'ADBE', 'AVGO', 'UNG', 'CLF', 'ENPH', 'CVX', 'SNOW', 'WBD', 'DAL', 'CCJ', 'CHPT', 'DVN', 'HUT', 'ET', 'NVAX', 'BIDU', 'AMAT', 'WULF', 'PARA', 'COST', 'CRM', 'LUMN', 'RILY', 'CRWD', 'BYND', 'XLV', 'NEE', 'AEO', 'AA', 'BLNK', 'FDX', 'CPB', 'UAL', 'VALE', 'XPEV', 'XLP', 'EOSE', 'RTX', 'PTON', 'VTNR', 'U', 'YINN', 'MS', 'TDOC', 'SPXS', 'IREN', 'ENVX', 'PBR', 'PG', 'GOLD', 'TGT', 'BTBT', 'XP', 'BTG', 'DDOG', 'SMCI', 'AYX', 'GE', 'XLU', 'CRSP', 'RIG', 'ZM', 'PEP', 'DASH', 'M', 'XLK', 'TEVA', 'SPWR', 'NET', 'JETS', 'LLY', 'ALT', 'CIFR', 'FCX', 'ABNB', 'DOCU', 'MRVL', 'PANW', 'MSOS', 'MNKD', 'FUBO', 'MDB', 'KVUE', 'TSEM', 'UNH', 'BX', 'AMGN', 'DG', 'FSLR', 'NLY', 'SCHW', 'GS', 'RCL', 'PINS', 'HE', 'THC', 'ASTS', 'LUV', 'OPEN', 'TTD', 'TZA', 'XHB', 'UCO', 'TMUS', 'GPS', 'LULU', 'PATH', 'HAL', 'NEM', 'UPS', 'CMCSA', 'KMI', 'ALB', 'BOIL', 'TAN', 'PENN', 'AVXL', 'W', 'URA', 'IBM', 'SPXU', 'VOO', 'PCT', 'TSN', 'CAT', 'HIVE', 'XLI', 'HD', 'ON', 'KHC', 'RKLB', 'BP', 'DOW', 'V', 'TFC', 'SGML', 'ATSG', 'CAN', 'CGC', 'TPR', 'HUM', 'TWLO', 'ZS', 'XLY', 'AXP', 'FUTU', 'ALGM', 'BILI', 'LMT', 'WDC', 'ETSY', 'CZR', 'SEDG', 'CLOV', 'LMND', 'XRT', 'HSY', 'MGM', 'KEY', 'KR', 'LABU', 'GILD', 'COP', 'RUN', 'TCOM', 'BRK_B', 'CVE', 'MRO', 'IOVA', 'EDR', 'GDXJ']
           

class OratsError(Exception):
    """Custom exception for ORATS API issues (HTTP errors, schema mismatches, etc.)."""
    pass


def yang_zhang_orats(df: pd.DataFrame, window: int = 30, trading_days: int = 252) -> float:
    """Compute Yang–Zhang realized volatility (annualized) over a trailing window.

    Assumes `df` contains columns: open, high, low, close (lowercase) and is
    sorted by ascending calendar date. The function internally takes the last
    `window` rows and applies the YZ estimator using:
      • overnight open/close terms (oc, co)
      • Rogers–Satchell term for intraday range

    Returns the *annualized* volatility using sqrt(trading_days) scaling.
    """
    d = df.sort_values("trade_date").tail(window).copy()

    # Extract as float numpy arrays (avoids dtype surprises)
    o = d["open"].astype(float).values
    h = d["high"].astype(float).values
    l = d["low"].astype(float).values
    c = d["close"].astype(float).values

    if len(d) < window:
        raise ValueError(f"Not enough rows for window={window} (got {len(d)})")

    # Overnight & open-close log returns
    oc = np.log(o[1:] / c[:-1])  # open[t] / close[t-1]
    co = np.log(c[1:] / o[1:])   # close[t] / open[t]

    # Rogers–Satchell term (skip first day, as oc/co start from index 1)
    u   = np.log(h[1:] / o[1:])
    dwn = np.log(l[1:] / o[1:])
    rs  = u * (u - co) + dwn * (dwn - co)

    # Sample variances/means
    oc_var  = np.var(oc, ddof=1)
    co_var  = np.var(co, ddof=1)
    rs_mean = np.mean(rs)

    n = len(d) - 1  # oc/co/rs length
    k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34 / 1.34

    yz_var   = oc_var + k * co_var + (1 - k) * rs_mean
    rv_annual = np.sqrt(max(yz_var, 0.0)) * np.sqrt(trading_days)
    return rv_annual


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


def compute_avg_vol(df: pd.DataFrame, days = 30) -> pd.DataFrame:
    df["avg_volume_30"] = df["volume"].rolling(days, min_periods=days).mean()
    return df


def compute_rv(df: pd.DataFrame, days = 30) -> pd.DataFrame:

    rv30_vals: list[float] = []
    for i in range(len(df)):
        if i < days - 1:
            rv30_vals.append(np.nan)
        else:
            rv30_vals.append(yang_zhang_orats(df.iloc[i - days + 1 : i + 1], window=days))
    df["rv30"] = rv30_vals
    return df


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
    underlying_df["tradeDate"] = pd.to_datetime(underlying_df["tradeDate"])  # ensure datetime dtype
    underlying_df = underlying_df.rename(
        columns={
            "clsPx": "close",
            "hiPx": "high",
            "loPx": "low",
            "stockVolume": "volume",
            "tradeDate": "trade_date",
        }
    )

    underlying_df = compute_avg_vol(underlying_df, days=30)
    underlying_df = compute_rv(underlying_df, days=30)
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
            row = ticker_ranges[ticker_ranges["ticker"] == ticker]
            start_date = row["min"].iloc[0]
            end_date = row["max"].iloc[0] # DO NOT USE values[0], it gives numpy datetime64, which causes issues later
            # to keep data size manageable, limit to post-2019 data
            if start_date < pd.to_datetime("2019-01-01"):
                start_date = pd.to_datetime("2019-01-01")
            logger.info(f"Fetching underlying data for {ticker:6s} from {start_date.date()} to {end_date.date()}...")
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


if __name__ == "__main__":
    
    underlying_path = BASE_DIR / "data" / "all_underlyings.parquet"    
    combined_df = upsert_underlying_data(pd.read_parquet(underlying_path) if underlying_path.exists() else pd.DataFrame(), ["SQ"], underlying_path)
    print(combined_df.dtypes)

    
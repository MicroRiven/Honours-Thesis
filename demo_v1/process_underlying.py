"""Process raw underlying price data: compute rolling realised volatility and volume.

Reads the raw parquet, fixes known bad data, computes Yang-Zhang and
close-to-close realised variance/vol, and writes the processed parquet.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def process_underlying(underlying_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 30-day average volume and realised variance/vol columns.

    Computes Yang-Zhang and close-to-close realised variance over a
    21-trading-day window (~1 month), then derives annualised vol.
    """
    days = 21  # 21 trading days ~ 1 month

    # Rolling average underlying volume
    underlying_df["avg_volume_30"] = underlying_df["volume"].rolling(days, min_periods=days).mean()

    # Yang-Zhang realised variance (trailing window)
    rv30_vals: list[float] = []
    for i in range(len(underlying_df)):  # start from index 0
        if i < days:
            rv30_vals.append(np.nan)
        else:  # the first valid index is i = 21
            rv30_vals.append(_yang_zhang(underlying_df.iloc[i - days : i], days=days, squared=True))
    underlying_df["rvar30_yz"] = rv30_vals

    # Close-to-close realised variance
    log_return = np.log(underlying_df["close"] / underlying_df["close"].shift(1))
    underlying_df["rvar30_cc"] = log_return.rolling(days, min_periods=days).var(ddof=1) * 252.0

    # Annualised vol = sqrt(variance)
    underlying_df["rvol30_yz"] = np.sqrt(underlying_df["rvar30_yz"])
    underlying_df["rvol30_cc"] = np.sqrt(underlying_df["rvar30_cc"])

    return underlying_df


def _yang_zhang(df: pd.DataFrame, days: int = 21, trading_days: int = 252, squared=True) -> float:
    """Compute Yang-Zhang realised variance (annualised) over a trailing window.

    Combines overnight (open/close), close-to-open, and Rogers-Satchell
    intraday range components. Returns variance if squared=True, else vol.
    """
    assert days >= 2, "Need at least 2 days to compute Yang-Zhang volatility"
    d = df.sort_values("trade_date").tail(days+1).copy()

    # Extract as float numpy arrays (avoids dtype surprises)
    o = d["open"].astype(float).values
    h = d["high"].astype(float).values
    l = d["low"].astype(float).values
    c = d["close"].astype(float).values

    # Overnight and close-to-open log returns
    oc = np.log(o[1:] / c[:-1])  # open[t] / close[t-1]
    co = np.log(c[1:] / o[1:])   # close[t] / open[t]

    # Rogers-Satchell intraday range term
    u   = np.log(h[1:] / o[1:])
    dwn = np.log(l[1:] / o[1:])
    rs  = u * (u - co) + dwn * (dwn - co)

    # Sample variances/means - small mathematical differences from paper
    oc_var  = np.var(oc, ddof=1)
    co_var  = np.var(co, ddof=1)
    rs_mean = np.mean(rs)

    # YZ weighting parameter
    k = 0.34 / (1.34 + (days + 1) / (days - 1)) if days > 1 else 0.34 / 1.34

    # Combined YZ variance, annualised
    yz_var   = oc_var + k * co_var + (1 - k) * rs_mean
    yz_var = max(yz_var, 0.0)
    yz_var_annual = yz_var * trading_days
    return yz_var_annual if squared else np.sqrt(yz_var_annual)


def main():
    """Load raw underlying data, fix known issues, compute features, and save."""

    underlying_path = BASE_DIR / "data" / "300_underlyings.parquet"
    out_path = BASE_DIR / "data" / "300_underlyings_processed.parquet"
    underlying_df = pd.read_parquet(underlying_path)
    underlying_df = underlying_df.sort_values(["ticker", "trade_date"])
    print(underlying_df.head())
    print()

    # --- Fix known bad data on 2021-01-13 for SPX/VIX ---
    # (likely missing decimal point; high/low ratio > 100x)
    for ticker in ["SPX", "VIX"]:
        mask = (underlying_df["ticker"] == ticker) & (underlying_df["trade_date"] == pd.Timestamp("2021-01-13"))
        print("Before:")
        print(underlying_df.loc[mask])
        difference = (underlying_df.loc[mask, "close"].iloc[0] / underlying_df.loc[mask, "low"].iloc[0])
        if difference > 3 or difference < 1/3:
            underlying_df.loc[mask, ["open", "high", "low", "unadjHiPx", "unadjLoPx", "unadjOpen"]] *= 100
            print("After:")
            print(underlying_df.loc[mask])
            print()
    underlying_df.to_parquet(underlying_path, index=False)
    print("Finished fixing bad data for 2021-01-13.")
    print()

    # --- Compute rolling features per ticker ---
    underlying_df.drop(columns=["unadjClsPx", "unadjHiPx", "unadjLoPx", "unadjOpen", "unadjStockVolume", "updatedAt"], inplace=True)
    print(underlying_df.head())
    underlying_grouped = underlying_df.groupby("ticker")
    underlying_dfs = []
    for ticker, underlying_df in underlying_grouped:
        print(f"Processing underlying for {ticker} with {len(underlying_df)} rows...")
        processed_df = process_underlying(underlying_df)
        underlying_dfs.append(processed_df)
    combined_underlying_df = pd.concat(underlying_dfs, ignore_index=True)
    combined_underlying_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()




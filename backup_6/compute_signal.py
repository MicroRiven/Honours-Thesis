"""
This code is based on backtest.ipynb
It has been refactored into functions for better modularity and reusability.

Fetching and filtering must be done separately, because fetching is slow, so we only do it once.

But should be filter and compute at the same time?
Since everything is offline and fast, seperating them shouldn't matter much.

"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import interp1d
from helper_code.options_reader import read_options_data

import logging
logger = logging.getLogger(__name__)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#=======================================================================


def get_signals(atm_today, underlying_row):
    """
    Given the filtered options data for a specific day and the underlying data,
    compute the signals based on the criteria.

    Args:
        atm_today (pd.DataFrame): At-the-money options data for the day.
        underlying_row (pd.Series): Series containing the underlying stock price data.

    Returns:
        dict: A dictionary containing the computed signals.
    """
    signals = {}

    if atm_today.empty:
        logger.info(f"No at-the-money options data for {underlying_row['date']}.")
        return signals

    # keep just what we need, ensure numeric & drop NaNs
    d = atm_today.copy()
    d["dte"] = pd.to_numeric(d["dte"], errors="coerce")
    d["vol"] = pd.to_numeric(d["vol"], errors="coerce")
    d = d.dropna(subset=["dte","vol"])
    days = d["dte"].to_numpy()
    ivs  = d["vol"].to_numpy()
    term_spline = build_term_structure(days, ivs)

    # Evaluate the term structure curve
    iv30 = term_spline(30.0)
    iv60 = term_spline(60.0)

    # compute forward factor
    fwd_var_30_60 = (iv60**2 * 60.0 - iv30**2 * 30.0) / 30.0
    fwd_var_30_60 = max(float(fwd_var_30_60), 0.0) # clamp tiny negatives from interpolation / rounding
    forward_iv_30_60 = float(np.sqrt(fwd_var_30_60))
    forward_factor_30_60 = np.nan
    if forward_iv_30_60 > 0:
        forward_factor_30_60 = (iv30 - forward_iv_30_60) / forward_iv_30_60

    # Slope between the earliest expiry and 45
    dte0 = float(days.min())
    ts_slope_0_45 = (term_spline(45.0) - term_spline(dte0)) / (45.0 - dte0)

    # Usage with your ORATS df:
    rv30_yz = underlying_row["rv30_yz"]
    rv30_cc = underlying_row["rv30_cc"]

    rv30 = rv30_yz # we can choose which one to use

    if rv30 <= 0 or not np.isfinite(rv30) or not np.isfinite(iv30):
        iv30_rv30 = np.nan
        logger.warning(f"Invalid RV30 ({rv30}) or IV30 ({iv30}) for ticker {underlying_row.name} on {underlying_row['trade_date']}. Setting iv30_rv30 to NaN.")
    else:
        iv30_rv30 = iv30 / rv30

    # Average volume check
    avg_underlying_volume = underlying_row["avg_volume_30"]

    # Expected move - need to look into this
    TARGET_DTE = 30
    row_30d = atm_today.loc[(atm_today["dte"] - TARGET_DTE).abs().idxmin()] # pick the expiry closest to 30D
    straddle = float(row_30d["cValue"]) + float(row_30d["pValue"])
    underlying_price = float(row_30d["stkPx"])
    expected_move_pct = round(straddle / underlying_price * 100, 2) # ??

    # Populate the signals dictionary
    signals = {
        "ticker": underlying_row.name,
        "trade_date": underlying_row["trade_date"],
        "rv30_yz": rv30_yz,
        "rv30_cc": rv30_cc,
        "atm_iv30": iv30,
        "atm_iv60": iv60,
        "forward_iv_30_60": forward_iv_30_60,
        "forward_factor_30_60": forward_factor_30_60,
        "ts45": ts_slope_0_45,
        "iv30_rv30": iv30_rv30,
        "avg_volume": avg_underlying_volume,
        "expected_move_pct": expected_move_pct
    }
    return signals


# Constructs a term structure spline from IVs at different expiration days
def build_term_structure(days, ivs):
    days = np.array(days, dtype=float)
    ivs  = np.array(ivs,  dtype=float)

    # sort
    idx  = np.argsort(days)
    days = days[idx]
    ivs  = ivs[idx]

    # linear interpolator (we’ll clamp outside)
    spline = interp1d(days, ivs, kind="linear", fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return float(ivs[0])
        elif dte > days[-1]:
            return float(ivs[-1])
        else:
            return float(spline(dte))
    return term_spline


def compute_signal_daily(
    underlying_today: pd.DataFrame,
    option_today: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:

    result = []
    tickers_set = set(tickers)

    # filter and group
    underlying_today = underlying_today[underlying_today["ticker"].isin(tickers_set)]
    underlying_indexed = underlying_today.set_index("ticker")
    option_today = option_today[option_today["ticker"].isin(tickers_set)]
    option_grouped = option_today.groupby("ticker")
    

    for ticker in tickers:
        if ticker not in underlying_indexed.index:
            logger.info(f"Ticker {ticker} not found in underlying data, skipping.")
            continue
        if ticker not in option_grouped.groups:
            logger.info(f"Ticker {ticker} not found in options data, skipping.")
            continue

        chain = option_grouped.get_group(ticker)
        urow = underlying_indexed.loc[ticker]
        
        assert len(urow.shape) == 1, "urow should be a Series"
        close = float(urow["close"])  # scalar!

        chain = chain.copy()
        chain["strike_diff"] = (chain["strike"].astype(float) - close).abs()

        # drop rows with missing or non-numeric values in critical columns
        chain_len_before = len(chain)
        chain = chain.dropna(subset=["expirDate","dte","vol","cValue","pValue","stkPx","strike_diff"])
        if len(chain) < chain_len_before:
            logger.warning(f"Dropped {chain_len_before - len(chain)} rows with missing values for ticker {ticker}.")
        if chain.empty:
            continue

        idx = chain.groupby("expirDate")["strike_diff"].idxmin()
        atm_today = chain.loc[idx]

        # pass a row/Series/dict, not a 1-row DataFrame
        sig = get_signals(atm_today, urow)
        if sig:
            result.append(sig)

    return pd.DataFrame(result)
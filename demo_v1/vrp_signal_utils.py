"""Utility functions for VRP signal computation.

Refactored from backtest.ipynb for modularity and reusability.

Notes:
- Fetching and filtering are done separately because fetching is slow.
- Since everything is offline and fast, separating filter and compute is fine.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d

from backtest_utils import filter_by_dte
from options_reader import read_options_data
from helper_code.vol_surface_utils import slice_chain_by_volume

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# ======================================================================


def build_otm_Q(chain, F):
    """Build OTM option mid-price array Q(K) for model-free IV calculation.

    For a single (ticker, trade_date, expiry) chain:
      K < K0 => put mid,  K > K0 => call mid,  K == K0 => avg(call, put).
    Returns the augmented chain and the reference strike K0.
    """
    chain = slice_chain_by_volume(chain, low_volume_threshold=0)

    # Convert columns to numpy arrays
    strikes = chain["strike"].to_numpy()
    c = chain["cValue"].to_numpy(dtype=float)
    p = chain["pValue"].to_numpy(dtype=float)

    # K0 = highest strike at or below the forward price
    strikes = np.asarray(strikes)
    below = strikes[strikes <= F]
    if below.size == 0:
        K0 = strikes.min()
        logger.warning(f"No strikes below forward price F={F:.2f}, using min strike K0={K0:.2f}")
    else:
        K0 = below.max()
    # Assign Q: puts below K0, calls above K0, average at K0
    Q = np.full(len(chain), np.nan, dtype=float)
    idx0 = np.where(strikes == K0)[0][0]

    for i in range(len(chain)):
        if strikes[i] < strikes[idx0]:
            Q[i] = p[i]
        elif strikes[i] > strikes[idx0]:
            Q[i] = c[i]
        else:
            Q[i] = np.nanmean([c[i], p[i]])

    chain["Q"] = Q
    return chain, K0


def get_signals(ticker, urow, opt_ticker, target_dte=30):
    """Compute all VRP-related signals for one ticker on one trading day.

    Combines model-free implied variance (MFIV) near target_dte with
    ATM IV term-structure interpolation and realised-vol from the underlying.

    Returns a dict of signals, or None if data is insufficient.
    """
    
    # --- Pre-processing ---
    # compute column before grouping
    chain = opt_ticker.reset_index(drop=True).copy()
    chain["contract_volume"] = (
        chain["cVolu"].fillna(0) + chain["pVolu"].fillna(0)
    )
    total_daily_volume = chain["contract_volume"].sum()
    # NOTE: we don't have to filter out any low volume here, we can do it in analysis
    chain["strike_diff"] = (chain["strike"] - chain["stkPx"]).abs()
    chain_grouped = chain.groupby("expirDate")

    # ==========================================
    # Model-free implied variance (MFIV) for expiry closest to target_dte
    # ==========================================

    # Filter to the expiry closest to target_dte
    chain_dte30 = filter_by_dte(chain, target_dte)
    chain_dte30 = chain_dte30.sort_values("strike")
    exp = chain_dte30["expirDate"].iloc[0]

    S = float(chain_dte30["stkPx"].iloc[0])
    r = float(chain_dte30["iRate"].median()) if "iRate" in chain_dte30.columns else 0.0
    q = float(chain_dte30["divRate"].median()) if "divRate" in chain_dte30.columns else 0.0
    T = float(chain_dte30["yte"].iloc[0])
    if T <= 0:
        print(chain_dte30.head())
        exit(1)
    # Forward price under cost-of-carry model
    F = S * np.exp((r - q) * T)

    # Build OTM Q(K) and compute model-free implied variance
    chain_dte30, K0 = build_otm_Q(chain_dte30, F)
    chain_dte30 = chain_dte30.sort_values("strike")
    K = chain_dte30["strike"].to_numpy()
    Q = chain_dte30["Q"].to_numpy()

    if len(K) < 2 or not np.isfinite(K0) or K0 <= 0:
        logger.warning(f"Not enough valid strikes to compute IV for ticker {ticker} on expiry {exp}.")
        return None

    # Trapezoidal-style strike spacing (dK)
    dK = np.empty_like(K)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0

    mask = np.isfinite(Q) & np.isfinite(dK) & np.isfinite(K) & (K > 0)
    if mask.sum() < 3:
        logger.warning(f"Not enough valid strikes to compute IV for ticker {ticker} on expiry {exp}. Valid points: {mask.sum()}")
        return None  # not enough valid strikes to compute IV
    
    # CBOE-style model-free implied variance
    integral = np.sum(Q[mask] * dK[mask] / (K[mask] ** 2))
    market_ivar = (2.0 / T) * np.exp(r * T) * integral - (1.0 / T) * (F / K0 - 1.0) ** 2
    if market_ivar < 0:
        logger.error(f"Computed negative implied variance for ticker {ticker} on expiry {exp}.")
        return None
    
    # ==========================================
    # ATM IV term structure signals
    # ==========================================

    # Select the ATM strike for each expiry
    idx = chain_grouped["strike_diff"].idxmin()  # index of nearest-ATM strike per expiry
    atm_today = chain.loc[idx]  # one row per expiry

    # Clean ATM data
    d = atm_today.copy()
    d["dte"] = pd.to_numeric(d["dte"], errors="coerce")
    d["vol"] = pd.to_numeric(d["vol"], errors="coerce")
    d = d.dropna(subset=["dte","vol"])

    # Build term structure spline and evaluate at key DTEs
    days = d["dte"].to_numpy()
    ivs  = d["vol"].to_numpy()
    spline = _build_term_structure(days, ivs)

    # Evaluate at key DTEs
    dte0 = float(days.min())
    iv0 = float(spline(dte0))  # IV at shortest available expiry
    iv30 = float(spline(30))
    iv45 = float(spline(45))
    iv60 = float(spline(60))

    # Realised vol from underlying data
    rvol30_yz = urow["rvol30_yz"]
    rvol30_cc = urow["rvol30_cc"]

    # Expected move - need to look into this
    # idx = (atm_today["dte"] - target_dte).abs().idxmin()
    # orow = atm_today.loc[idx]
    # straddle = float(orow["cValue"]) + float(orow["pValue"])
    # underlying_price = float(orow["stkPx"])
    # expected_move = round(straddle / underlying_price * 100, 2)

    # --- Assemble output ---
    signals = {
        "ticker": ticker,
        "trade_date": urow["trade_date"],
        "rvol30_yz": rvol30_yz,
        "rvol30_cc": rvol30_cc,
        "ivar30": market_ivar,

        # atm term-structure signals
        "dte0": dte0,
        "atm_iv0": iv0,
        "atm_iv30": iv30,
        "atm_iv45": iv45,
        "atm_iv60": iv60,

        "avg_underlying_volume": urow["avg_volume_30"],
        "daily_option_volume": total_daily_volume,

        # the following signals will be computed later in the analysis notebook
        # "market_iv30": np.sqrt(market_ivar),
        # "iv30_rv30": iv30 / rvol30_yz
        # "fwd_factor_30_60"
        # "ts_slope_30", "ts_slope_45", "ts_slope_60"
    }

    return signals


# Constructs a term structure spline from IVs at different expiration days
def _build_term_structure(days, ivs):
    """Build a linear interpolator for ATM IV as a function of DTE.

    Clamps to boundary values outside the observed range.
    """
    days = np.array(days, dtype=float)
    ivs  = np.array(ivs,  dtype=float)

    # sort
    idx  = np.argsort(days)
    days = days[idx]
    ivs  = ivs[idx]

    # Linear interpolator, clamped outside observed range
    spline = interp1d(days, ivs, kind="linear",
                    bounds_error=False,
                    fill_value=(ivs[0], ivs[-1]))
    return spline


def compute_signal_daily(
    tickers: list[str],
    underlying_today: pd.DataFrame,
    option_today: pd.DataFrame,
    target_dte: int = 30,
    dte_window: tuple[int, int] = None,  # optional DTE range filter
) -> pd.DataFrame:
    """Compute VRP signals for all tickers on a single trading day.

    Filters and joins underlying + options data, then calls get_signals()
    per ticker. Returns a DataFrame of signal rows (float32).
    """

    result = []
    tickers_set = set(tickers)

    # --- Filter and index data for fast per-ticker lookup ---
    underlying_today = underlying_today[underlying_today["ticker"].isin(tickers_set)]
    underlying_indexed = underlying_today.set_index("ticker")
    option_today = option_today[option_today["ticker"].isin(tickers_set)]
    if dte_window is not None:
        option_today = option_today[option_today["dte"].between(dte_window[0], dte_window[1])]
    option_grouped = option_today.groupby("ticker")
    

    for ticker in tickers:
        if ticker not in underlying_indexed.index:
            logger.debug(f"Ticker {ticker} not found in underlying data, skipping.")  # frequent with fixed universe
            continue
        if ticker not in option_grouped.groups:
            logger.info(f"Ticker {ticker} not found in options data, skipping.")
            continue

        # Retrieve per-ticker data
        opt_ticker = option_grouped.get_group(ticker)  # already filtered by trade_date and ticker
        urow = underlying_indexed.loc[ticker]

        # Drop rows with critical missing fields
        assert len(urow.shape) == 1, "urow should be a Series"
        len_before = len(opt_ticker)
        opt_ticker = opt_ticker.dropna(subset=["expirDate","dte","vol","cValue","pValue","stkPx","strike"])
        if len(opt_ticker) < len_before:
            logger.warning(f"Dropped {len_before - len(opt_ticker)} rows with missing values for ticker {ticker}.")
        if opt_ticker.empty:
            continue

        sig = get_signals(ticker, urow, opt_ticker, target_dte=target_dte)
        if sig:  # skip None results
            result.append(sig)

    # Downcast floats to save memory
    df = pd.DataFrame(result)
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df
    

# --- Local testing ---
if __name__ == "__main__":

    # Read underlying data
    underlying_path = BASE_DIR / "data" / "300_underlyings_processed.parquet"
    underlying = pd.read_parquet(underlying_path)
    underlying_today = underlying[underlying["trade_date"] == pd.to_datetime("2024-01-02")]

    # Read options data
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    dt = pd.to_datetime("2024-01-02")
    cols = ["yte", "iRate", "divRate"]
    chain = read_options_data(dt, tickers=tickers, cols=cols)

    ivs = compute_signal_daily(tickers, underlying_today, chain)
    print(ivs)

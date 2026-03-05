"""
Straddle simulation with ORATS Historical Data API.

- Sells 1x ATM call + 1x ATM put on trade_date
- Holds to selected expiration (or nearest if not specified)
- Prices entry at mid (bid/ask), falls back to model "Value" if quotes are missing
- Uses unadjusted close at expiration for P&L
"""

from __future__ import annotations
import os
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# configures logging with logging.basicConfig(),
# all loggers in imported modules (like compute_pnl) automatically inherit that configuration.
# This is the standard Python logging pattern.
import logging
logger = logging.getLogger(__name__)

from helper_code.options_reader import read_options_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    ticker: str
    strategy: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    short_expiry: pd.Timestamp
    long_expiry: Optional[pd.Timestamp]
    strike: float
    entry_credit: float
    entry_debit: float
    signals: dict  # snapshot at entry


def choose_expiry_by_dte(opt: pd.DataFrame, target_dte: int) -> pd.Timestamp:
    df = opt[["expirDate", "trade_date"]].drop_duplicates().copy()
    df["dte"] = (df["expirDate"] - df["trade_date"]).dt.days
    df["abs_err"] = (df["dte"] - target_dte).abs()
    return df.sort_values("abs_err").iloc[0]["expirDate"]


def choose_common_atm_strike(
    opt: pd.DataFrame,
    short_exp: pd.Timestamp,
    long_exp: pd.Timestamp,
    spot: float,
) -> float:
    # Pull unique strikes for each expiry
    s_short = opt.loc[opt["expirDate"].eq(short_exp), "strike"].dropna().unique()
    s_long  = opt.loc[opt["expirDate"].eq(long_exp),  "strike"].dropna().unique()

    if len(s_short) == 0 or len(s_long) == 0:
        logging.warning("Missing strikes for one of the expiries.")
        return None

    # Intersection (common strikes)
    common = np.intersect1d(s_short, s_long, assume_unique=False)
    if common.size == 0:
        logging.warning(f"No common strikes between expiries {short_exp.date()} and {long_exp.date()} for ticker {opt['ticker'].iloc[0] if not opt.empty else 'unknown'}.")
        return None

    # Choose closest to spot (ATM) among common strikes
    i = np.abs(common - spot).argmin()
    return float(common[i])


def option_price(
    opt: pd.DataFrame, # already filtered for ticker
    expiry: pd.Timestamp,
    strike: float,
    ticker: str, # only for logging purposes
    call: bool = True,
) -> float | None:
    '''
    No point using a max strike diff because options for a liquid ticker have small strike gaps.
    '''
    key = "cValue" if call else "pValue"

    # slice for expiry
    if opt.empty:
        return None
    sub = opt.loc[opt["expirDate"].eq(expiry), ["strike", key]].copy()
    sub = sub.dropna(subset=["strike"])
    if sub.empty:
        return None

    # Exact strike
    row = sub.loc[sub["strike"].eq(strike)] # pd.DataFrame
    if not row.empty:
        price = row[key].iloc[0]
        return None if pd.isna(price) else price

    # Closest strike
    diffs = (sub["strike"] - strike).abs()
    idx = diffs.idxmin()
    row = sub.loc[idx] # pd.Series, don't need iloc[0] later
    logger.warning(
        f"[{ticker}] option price substituted @ {expiry.date()} "
        f"{strike:.4f} -> {row['strike']:.4f}"
    )
    price = row[key]
    return None if pd.isna(price) else price


def close_on_or_before(
    underlying_ticker: pd.DataFrame,
    date: pd.Timestamp,
) -> float:
    """
    Returns the close price on `date` if available,
    otherwise the most recent close strictly before `date`.

    Logs a warning if a fallback date is used.
    """
    idx = underlying_ticker.index
    assert idx.is_monotonic_increasing, "Index must be sorted by trade_date"

    # Find position of last index <= date
    pos = idx.searchsorted(date, side="right") - 1

    if pos < 0:
        ticker = underlying_ticker["ticker"].iloc[0]
        raise ValueError(f"[{ticker}] No underlying data on or before {date.date()}")

    used_date = idx[pos]
    price = underlying_ticker.iloc[pos]["close"]

    if used_date != date:
        ticker = underlying_ticker["ticker"].iloc[0]
        logger.warning(
            f"[{ticker}] underlying close substituted: requested={date.date()} used={used_date.date()}"
        )

    return float(price)


def log_position_details(pos: Position, extra_info: dict = None):
    """Log detailed position information for debugging."""
    info = {
        "ticker": pos.ticker,
        "strategy": pos.strategy,
        "open_date": pos.open_date.date(),
        "close_date": pos.close_date.date(),
        "short_expiry": pos.short_expiry.date(),
        "long_expiry": pos.long_expiry.date() if pos.long_expiry else None,
        "strike": pos.strike,
        "entry_credit": pos.entry_credit,
        "entry_debit": pos.entry_debit,
    }
    if extra_info:
        info.update(extra_info)
    logger.error(f"Position details: {info}")


def open_positions_for_day(
    dt: pd.Timestamp,
    underlying_today: pd.DataFrame,
    options_today: pd.DataFrame,
    tickers: list[str],
    signals_today: pd.DataFrame,
) -> list[Position]:
    """
    Create new positions opened on dt for each ticker.
    Returns a list of Position objects to be appended to open_positions.
    """
    new_positions: list[Position] = []

    # Ensure index for fast price lookup
    assert "ticker" in underlying_today.columns
    assert "ticker" in signals_today.columns
    u = underlying_today.set_index("ticker", drop=False)
    s = signals_today.set_index("ticker", drop=False)


    for ticker in tickers:

        # Setup signal, underlying, option slices for ticker
        if ticker not in s.index or ticker not in u.index:
            logger.warning(f"Ticker {ticker} missing in signals or underlying data on {dt.date()}, skipping.")
            continue 
        sig = s.loc[ticker].to_dict() # get the pd.Series then convert to dict
        spot = float(u.loc[ticker, "close"])
        opt = options_today.loc[options_today["ticker"].eq(ticker)].copy()
        if opt.empty:
            logger.warning(f"Ticker {ticker} not found in options data on {dt.date()}, skipping.")
            continue


        # NOTE: short and long means short the near-term expiry, long the further expiry
        short_exp = choose_expiry_by_dte(opt, 30)
        long_exp = choose_expiry_by_dte(opt, 60)
        strike = choose_common_atm_strike(opt, short_exp, long_exp, spot)
        if strike is None:
            logger.warning(f"Could not determine common ATM strike for {ticker} on {dt.date()}, skipping.")
            continue

        short_c =  option_price(opt, short_exp, strike, ticker, call=True)
        short_p = option_price(opt, short_exp, strike, ticker, call=False)
        long_c = option_price(opt, long_exp, strike, ticker, call=True)
        long_p = option_price(opt, long_exp, strike, ticker, call=False)

        if None in [short_c, short_p, long_c, long_p]:
            logger.warning(f"Could not open position for {ticker} on {dt.date()}: missing leg price.")
            continue

        # Short straddle to expiry
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="short_straddle_hold",
                open_date=dt,
                close_date=short_exp, # close on expiry
                short_expiry=short_exp,
                long_expiry=None,
                strike=strike,
                entry_credit=short_c + short_p,
                entry_debit=0.0,
                signals=sig,
            )
        )

        # Short straddle, close after 1 week (next trading day after +5 calendar)
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="short_straddle_1w",
                open_date=dt,
                close_date=dt + pd.Timedelta(days=5),
                short_expiry=short_exp,
                long_expiry=None,
                strike=strike,
                entry_credit=short_c + short_p,
                entry_debit=0.0,
                signals=sig
            )
        )

        # Long call calendar 30/60 (short 30D, long 60D) close on short expiry
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="long_call_calendar_30_60",
                open_date=dt,
                close_date=short_exp,
                short_expiry=short_exp,
                long_expiry=long_exp,
                strike=strike,
                entry_credit=short_c,
                entry_debit=long_c,
                signals=sig,
            )
        )

        # Long straddle calendar 30/60 (short 30D, long 60D) close on short expiry
        # NOTE: we can add a long put calendar and derive the straddle from there
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="long_straddle_calendar_30_60",
                open_date=dt,
                close_date=short_exp,
                short_expiry=short_exp,
                long_expiry=long_exp,
                strike=strike,
                entry_credit=short_c + short_p,
                entry_debit=long_c + long_p,
                signals=sig,
            )
        )

    return new_positions


def close_positions_for_day(
    dt: pd.Timestamp,
    underlying_full: pd.DataFrame,
    options_today: pd.DataFrame,
    open_positions: list[Position],
) -> tuple[list[Position], list[dict]]:
    """
    Close any positions whose close_date <= dt using today's option chain.
    Returns:
      - still_open_positions
      - closed_trades (list of dict rows)
    """

    still_open: list[Position] = []
    closed_trades: list[dict] = []

    for pos in open_positions:
        if dt < pos.close_date:
            still_open.append(pos)
            continue
                
        underlying_ticker = (
            underlying_full.loc[underlying_full["ticker"].eq(pos.ticker)]
            .set_index("trade_date", drop=False)
            .sort_index()
        )
        opt = options_today.loc[options_today["ticker"].eq(pos.ticker)].copy()
        if opt.empty:
            logger.error(f"Ticker {pos.ticker} not found in options data on {dt.date()} when closing position opened on {pos.open_date.date()}. Deleting position.")
            continue

        # Compute PnL based on strategy
        # NOTE: Repeating code doesn't hurt since we are using if/elif
        if pos.strategy == "short_straddle_hold":
            S_T = close_on_or_before(underlying_ticker, pos.close_date)
            exit_short_val = max(0, S_T - pos.strike) + max(0, pos.strike - S_T)
            pnl = pos.entry_credit - exit_short_val
            notional = pos.entry_credit
        
        elif pos.strategy == "short_straddle_1w":
            c = option_price(opt, pos.short_expiry, pos.strike, pos.ticker, call=True)
            p = option_price(opt, pos.short_expiry, pos.strike, pos.ticker, call=False)
            if c is None or p is None:
                logger.error(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                continue
            exit_short_val = c + p
            pnl = pos.entry_credit - exit_short_val
            notional = pos.entry_credit

        elif pos.strategy == "long_call_calendar_30_60":
            S_T = close_on_or_before(underlying_ticker, pos.short_expiry)
            short_c = max(0, S_T - pos.strike)
            long_c = option_price(opt, pos.long_expiry, pos.strike, pos.ticker, call=True)
            if long_c is None:
                logger.error(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                continue
            pnl = pos.entry_credit - pos.entry_debit - short_c + long_c
            notional = pos.entry_debit

        elif pos.strategy == "long_straddle_calendar_30_60":
            S_T = close_on_or_before(underlying_ticker, pos.short_expiry)
            short_c = max(0, S_T - pos.strike)
            short_p = max(0, pos.strike - S_T)
            # NOTE: maybe we can check how much these differ from option prices at expiry
            long_c = option_price(opt, pos.long_expiry, pos.strike, pos.ticker, call=True)
            long_p = option_price(opt, pos.long_expiry, pos.strike, pos.ticker, call=False)
            if None in [short_c, short_p, long_c, long_p]:
                logger.error(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                continue
            pnl = pos.entry_credit - pos.entry_debit - (short_c + short_p) + (long_c + long_p)
            notional = pos.entry_debit

        pnl *= 100.0 # Dollar PnL for 1-lot
        notional *= 100.0
        assert notional is not None and np.isfinite(notional)
        if notional <= 0:
            # this happens if option prices are 0, maybe due to rounding or data issues
            logger.error(f"Notional = {notional} for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
            continue
        pnl_norm = pnl / notional # computed at the end

        closed_trades.append({
            "ticker": pos.ticker,
            "strategy": pos.strategy,
            "open_date": pos.open_date,
            "close_date": dt,
            "short_expiry": pos.short_expiry,
            "long_expiry": pos.long_expiry,
            "strike": pos.strike,
            "entry_credit": pos.entry_credit,
            "entry_debit": pos.entry_debit,

            # we will add the exit value if needed later
            # it can be computed from pnl - entry_credit + entry_debit

            "pnl": pnl,
            "pnl_norm": pnl_norm,
            "win": bool(pnl > 0),
            **pos.signals,
        })

    return still_open, closed_trades
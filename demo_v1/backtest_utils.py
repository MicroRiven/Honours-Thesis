"""
Straddle simulation with ORATS Historical Data API.

- Sells 1x ATM call + 1x ATM put on trade_date
- Holds to selected expiration (or nearest if not specified)
- Prices entry at mid (bid/ask), falls back to model "Value" if quotes are missing
- Uses unadjusted close at expiration for P&L
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import asdict, dataclass
from typing import Optional

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


def _inc_monthly_stat(monthly_stats, dt: pd.Timestamp, key: str, amount: int = 1) -> None:
    if monthly_stats is None:
        return
    month_key = pd.Timestamp(dt).strftime("%Y-%m")
    if month_key not in monthly_stats:
        monthly_stats[month_key] = {}
    bucket = monthly_stats[month_key]
    bucket[key] = bucket.get(key, 0) + amount


@dataclass
class Position:
    ticker: str
    strategy: str
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    near_expiry: pd.Timestamp
    far_expiry: Optional[pd.Timestamp] = None
    spot_at_open: Optional[float] = None
    spot_at_close: Optional[float] = None
    strike_15d_30dte: Optional[float] = None
    strike_50d_30dte: Optional[float] = None
    strike_85d_30dte: Optional[float] = None
    strike_50d_60dte: Optional[float] = None
    entry_cashflow: Optional[float] = None
    exit_cashflow: Optional[float] = None
    notional: Optional[float] = None


def load_tickers(
    path: str | Path,
    start_year: int,
    end_year: int,
    volume_threshold: int = 10_000
) -> dict[int, list[str]]:

    tickers_by_year: dict[int, list[str]] = {}
    for year in range(start_year, end_year + 1):
        df = pd.read_excel(path, sheet_name=f"{year}")
        df = df[df["avg_option_volume"] > volume_threshold] # filter avg_option_volume > volume_threshold
        tickers = df["ticker"].tolist()
        tickers_by_year[year] = tickers

    return tickers_by_year


def iter_trading_dates(
    underlying_df: pd.DataFrame,
    start: str,
    end: str
) -> list[pd.Timestamp]:

    dates = pd.Index(underlying_df["trade_date"].unique()).sort_values()
    # Convert month strings (e.g., "2024-02") to first/last day of month
    start_dt = pd.to_datetime(start) # no need to normalise since pd.to_datetime("2024-01") already produces 2024-01-01 00:00:00 with no time component
    end_dt   = (pd.to_datetime(end) + pd.offsets.MonthEnd(0)).normalize()
    dates = dates[(dates >= start_dt) & (dates <= end_dt)]

    return list(dates)   # list of Timestamps



def filter_by_dte(
    opt: pd.DataFrame,
    target_dte: int | None = None,
    target_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    
    df = opt.copy()
    # prioritise target_dte
    if target_date is not None:
        df["abs_err"] = (df["expirDate"] - target_date).abs().dt.days
    elif target_dte is not None:
        df["abs_err"] = (df["dte"] - target_dte).abs() # dte should already exist
    else:
        raise ValueError("Either target_dte or target_date must be provided.")

    # return the dataframe with the minimum abs_err
    # if multiple rows have the same abs_err, return all of them
    min_err = df["abs_err"].min() # type(min_err) = numpy.int64
    df = df[df["abs_err"] == min_err].drop(columns=["abs_err"])
    # if two expiries are equidistance from target_dte, we take the further expiry
    if df["expirDate"].nunique() > 1:
        logger.debug(f"Multiple expiries equidistant from target DTE={target_dte}. Choosing further expiry for ticker {df['ticker'].iloc[0]}.")
        expiry = df["expirDate"].max()
        df = df[df["expirDate"] == expiry]

    return df


def choose_strikes(near_df, far_df,):

    # find the 15/50/85 delta strikes for near expiry
    # NOTE: delta is monotonic; idxmin only returns one anyway
    idx_15d = (near_df["delta"].abs() - 0.15).abs().idxmin()
    idx_50d = (near_df["delta"].abs() - 0.5).abs().idxmin()
    idx_85d = (near_df["delta"].abs() - 0.85).abs().idxmin()
    strike_15d_row = near_df.loc[idx_15d]
    strike_50d_row = near_df.loc[idx_50d]
    strike_85d_row = near_df.loc[idx_85d]

    idx_50d_60dte = (far_df["delta"].abs() - 0.5).abs().idxmin()
    strike_50d_60dte_row  = far_df.loc[idx_50d_60dte]
    
    return strike_15d_row, strike_50d_row, strike_85d_row, strike_50d_60dte_row


def option_price(
    opt: pd.DataFrame, # filtered for trade_date and ticker already, but not by expiry
    expiry: pd.Timestamp,
    strike: float,
    call: bool = True,
) -> float | None:
    '''
    No point using a max strike diff because options for a liquid ticker have small strike gaps.
    '''
    key = "cValue" if call else "pValue"
    volu_key = "cVolu" if call else "pVolu"
    
    # slice for expiry
    row = opt.loc[opt["expirDate"].eq(expiry) & opt["strike"].eq(strike)]
    if row.empty:
        return None
    
    price = row[key].iloc[0]
    volu = row[volu_key].iloc[0]
    if pd.isna(price) or volu < 5:
        return None
    
    return price


def _validate_legs(
    ticker: str,
    dt: pd.Timestamp,
    strike_15d_row: pd.Series,
    strike_50d_row: pd.Series,
    strike_85d_row: pd.Series,
    strike_50d_60dte_row: pd.Series,
    monthly_stats=None,
) -> bool:
    '''
    Return True if there's an issue, False if all good
    '''
    amount=5 

    # validate volume for atm leg
    if strike_50d_row["cVolu"] + strike_50d_row["pVolu"] < 10:
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: low volume at atm leg.")
        logger.debug(f"Expiry: {strike_50d_row['expirDate'].date()}, Strike: {strike_50d_row['strike']}, cVolu: {strike_50d_row['cVolu']}, pVolu: {strike_50d_row['pVolu']}")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, f"open_fail_low_volume_near_atm_leg", amount=amount)
        return True

    for row in [strike_15d_row, strike_50d_row, strike_85d_row, strike_50d_60dte_row]:
        
        assert type(row) == pd.Series

        # validate price
        call_price = row["cValue"]
        put_price = row["pValue"]
        
        if pd.isna(call_price) or pd.isna(put_price) or call_price <= 0 or put_price <= 0:
            logger.debug(f"Could not open position for {ticker} on {dt.date()}: NaN price.")
            logger.debug(f"Expiry: {row['expirDate'].date()}, Strike: {row['strike']}, cValue: {row['cValue']}, pValue: {row['pValue']}")
            _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
            _inc_monthly_stat(monthly_stats, dt, f"open_fail_invalid_leg_price", amount=amount)
            return True
    
    return False



def _validate_butterfly_legs(
    ticker: str,
    dt: pd.Timestamp,
    strike_15d_row: pd.Series,
    strike_50d_row: pd.Series,
    strike_85d_row: pd.Series,
    monthly_stats=None,
) -> bool:
    
    amount = 1

    # validate volume for 15d and 85d legs
    if strike_15d_row["cVolu"] < 5 or strike_85d_row["cVolu"] < 5:
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: low volume at 15d or 85d leg.")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, f"open_fail_low_volume_15d_85d_leg", amount=amount)
        return True

    # validate delta
    if not (0.08 <= strike_15d_row["delta"] <= 0.22):
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: 15 delta out of range. Delta: {strike_15d_row['delta']:.4f}")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_invalid_15_delta", amount=amount)
        return True
    if not (0.78 <= strike_85d_row["delta"] <= 0.92):
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: 85 delta out of range. Delta: {strike_85d_row['delta']:.4f}")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_invalid_85_delta", amount=amount)
        return True
    
    # validate symmetry of strikes
    strike_15d = strike_15d_row["strike"]
    strike_50d = strike_50d_row["strike"]
    strike_85d = strike_85d_row["strike"]
    lower_diff = strike_50d - strike_85d
    upper_diff = strike_15d - strike_50d

    flag = False
    if lower_diff == 0: 
        flag = True
    elif not (0.67 <= upper_diff / lower_diff <= 1.5):
        flag = True
    if flag:
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: butterfly legs not symmetric. Strikes: {strike_15d:.2f}, {strike_50d:.2f}, {strike_85d:.2f}")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_anti_symmetric_legs", amount=amount)
        return True
    
    # validate entry cashflow
    entry_cashflow = - strike_15d_row["cValue"] + 2 * strike_50d_row["cValue"] - strike_85d_row["cValue"]
    if entry_cashflow >= 0:
        logger.warning(f"Unexpected negative entry cashflow for long call butterfly. Ticker: {ticker}, Strikes: {strike_15d_row['strike']}, {strike_50d_row['strike']}, {strike_85d_row['strike']}, Entry cashflow: {entry_cashflow:.2f}. Skipping.")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_negative_entry_cashflow_butterfly", amount=amount)
        return True

    return False



def _validate_calendar_leg(
    ticker: str,
    dt: pd.Timestamp,
    near_exp: pd.Timestamp,
    far_exp: pd.Timestamp,
    strike_50d_60dte_row: pd.Series,
    max_moneyness_deviation: float = 0.1,
    monthly_stats=None,
) -> bool:
    '''
    check moneyness deviation
    if this fails, something weird is happening with the option chain, so we skip the ticker for the day
    Return True if there's an issue, False if all good
    '''

    amount = 2

    # validate volume for far leg
    if strike_50d_60dte_row["cVolu"] + strike_50d_60dte_row["pVolu"] < 10:
        logger.debug(f"Could not open position for {ticker} on {dt.date()}: low volume at far leg.")
        logger.debug(f"Expiry: {strike_50d_60dte_row['expirDate'].date()}, Strike: {strike_50d_60dte_row['strike']}, cVolu: {strike_50d_60dte_row['cVolu']}, pVolu: {strike_50d_60dte_row['pVolu']}")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=amount)
        _inc_monthly_stat(monthly_stats, dt, f"open_fail_low_volume_far_leg", amount=amount)
        return True


    # validate expiry gap
    if (far_exp - near_exp).days < 15 or (far_exp - near_exp).days > 45: # sanity check to avoid weird cases where far expiry is too close to near expiry
        logger.debug(f"Ticker {ticker} on {dt.date()}: Far expiry {far_exp.date()} is not within 15-45 days after near expiry {near_exp.date()}, skipping.")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount = amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_far_expiry_too_close", amount = amount)
        return True

    assert type(strike_50d_60dte_row) == pd.Series
    spot        = strike_50d_60dte_row["stkPx"]
    far_strike  = strike_50d_60dte_row["strike"]
    if abs(far_strike - spot) / spot > max_moneyness_deviation:
        logger.debug(f"Could not open calendar position for {ticker} on {dt.date()}: far leg strike too far from spot.")
        _inc_monthly_stat(monthly_stats, dt, "open_failed", amount = amount)
        _inc_monthly_stat(monthly_stats, dt, "open_fail_invalid_calendar_leg", amount = amount)
        return True
    
    return False


def open_positions_for_day(
    dt: pd.Timestamp,
    options_today: pd.DataFrame,
    tickers: list[str],
    monthly_stats=None,
) -> list[Position]:
    """
    Create new positions opened on dt for each ticker.
    Returns a list of Position objects to be appended to open_positions.
    """
    new_positions: list[Position] = []
    opt_grouped = options_today.groupby("ticker", sort=False)

    for ticker in tickers:
        _inc_monthly_stat(monthly_stats, dt, "open_attempted", amount=5)
    
        # Setup option slices for ticker
        try:
            opt = opt_grouped.get_group(ticker).copy()
        except KeyError:
            logger.debug(f"Ticker {ticker} missing in options data on {dt.date()}, skipping.")
            _inc_monthly_stat(monthly_stats, dt, "open_failed", amount=5)
            _inc_monthly_stat(monthly_stats, dt, "open_fail_missing_ticker", amount=5)
            continue
        assert not opt.empty
        
        # NOTE: short and long means short the near-term expiry, long the further expiry
        near_df = filter_by_dte(opt, target_dte = 30)
        near_exp = near_df["expirDate"].iloc[0]
        far_df = filter_by_dte(opt, target_date = near_exp + pd.Timedelta(days=30)) # target far expiry 30D after near expiry
        far_exp = far_df["expirDate"].iloc[0]

        strike_15d_row, strike_50d_row, strike_85d_row, strike_50d_60dte_row = choose_strikes(near_df, far_df)
        spot = strike_50d_row["stkPx"]

        if _validate_legs(
            ticker,
            dt,
            strike_15d_row,
            strike_50d_row,
            strike_85d_row,
            strike_50d_60dte_row,
            monthly_stats,
        ):
            continue

        # Short straddle to expiry
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="short_straddle_hold",
                open_date=dt,
                close_date=near_exp, # close on expiry
                spot_at_open=spot,
                near_expiry=near_exp,
                strike_50d_30dte = strike_50d_row["strike"],
                entry_cashflow = strike_50d_row["cValue"] + strike_50d_row["pValue"], # credit
                notional = strike_50d_row["cValue"] + strike_50d_row["pValue"],
            )
        )
        _inc_monthly_stat(monthly_stats, dt, "open_success")

        # Short straddle, close after 1 week (next trading day after +7 calendar)
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="short_straddle_1w",
                open_date=dt,
                close_date=dt + pd.Timedelta(days=7),
                spot_at_open=spot,
                near_expiry=near_exp,
                strike_50d_30dte= strike_50d_row["strike"],
                entry_cashflow = strike_50d_row["cValue"] + strike_50d_row["pValue"], # credit
                notional = strike_50d_row["cValue"] + strike_50d_row["pValue"],
            )
        )
        _inc_monthly_stat(monthly_stats, dt, "open_success")

        if not _validate_butterfly_legs(
            ticker,
            dt,
            strike_15d_row,
            strike_50d_row,
            strike_85d_row,
            monthly_stats=monthly_stats,
        ):
            
            # more checks
            entry_cashflow = - strike_15d_row["cValue"] + 2 * strike_50d_row["cValue"] - strike_85d_row["cValue"]
            min_payoff = min(0, 2 * strike_50d_row["strike"] - strike_85d_row["strike"] - strike_15d_row["strike"])
            assert min_payoff <= 0
            notional = - entry_cashflow - min_payoff # positive value of max loss
            assert notional > 0

            # Long call butterfly, to expiry (buy 15d, sell 50d, buy 85d)
            new_positions.append(
                Position(
                    ticker=ticker,
                    strategy="long_call_butterfly_hold",
                    open_date=dt,
                    close_date=near_exp,
                    spot_at_open=spot,
                    near_expiry=near_exp,
                    strike_15d_30dte= strike_15d_row["strike"],
                    strike_50d_30dte= strike_50d_row["strike"],
                    strike_85d_30dte= strike_85d_row["strike"],
                    entry_cashflow = entry_cashflow, # usually negative: net debit paid
                    notional = notional, # positive value of max loss
                )
            )
            _inc_monthly_stat(monthly_stats, dt, "open_success")

        if _validate_calendar_leg(
            ticker,
            dt,
            near_exp,
            far_exp,
            strike_50d_60dte_row,
            monthly_stats = monthly_stats,
        ):
            continue

        # Long call calendar 30/60 (short 30D, long 60D) close on short expiry
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="long_call_calendar_30_60",
                open_date=dt,
                close_date=near_exp,
                spot_at_open=spot,
                near_expiry=near_exp,
                far_expiry=far_exp,
                strike_50d_30dte= strike_50d_row["strike"],
                strike_50d_60dte= strike_50d_60dte_row["strike"],
                entry_cashflow = strike_50d_row["cValue"] - strike_50d_60dte_row["cValue"], # usually negative: net debit paid
                notional = strike_50d_row["cValue"] + strike_50d_60dte_row["cValue"],
            )
        )
        _inc_monthly_stat(monthly_stats, dt, "open_success")

        # Long straddle calendar 30/60 (short 30D, long 60D) close on short expiry
        # NOTE: we can add a long put calendar and derive the straddle from there
        new_positions.append(
            Position(
                ticker=ticker,
                strategy="long_straddle_calendar_30_60",
                open_date=dt,
                close_date=near_exp,
                spot_at_open=spot,
                near_expiry=near_exp,
                far_expiry=far_exp,
                strike_50d_30dte= strike_50d_row["strike"],
                strike_50d_60dte= strike_50d_60dte_row["strike"],
                entry_cashflow = strike_50d_row["cValue"] + strike_50d_row["pValue"] - strike_50d_60dte_row["cValue"] - strike_50d_60dte_row["pValue"], # usually negative: net debit paid
                notional = strike_50d_row["cValue"] + strike_50d_row["pValue"] + strike_50d_60dte_row["cValue"] + strike_50d_60dte_row["pValue"],
            )
        )
        _inc_monthly_stat(monthly_stats, dt, "open_success")

    return new_positions


def close_positions_for_day(
    dt: pd.Timestamp,
    options_today: pd.DataFrame,
    open_positions: list[Position],
    stock_split_df: pd.DataFrame,
    monthly_stats=None,
) -> tuple[list[Position], list[dict]]:
    """
    Close any positions whose close_date <= dt using today's option chain.
    Returns:
      - still_open_positions
      - closed_trades (list of dict rows)
    """

    still_open: list[Position] = []
    closed_trades: list[dict] = []

    opt_grouped = options_today.groupby("ticker", sort=False)

    for pos in open_positions:
        if dt < pos.close_date:
            still_open.append(pos)
            continue

        # remaining positions should be due today or overdue by a few days
        if dt > pos.close_date:
            logger.debug(f"Position for {pos.ticker} opened on {pos.open_date.date()} is overdue probably due to weekend or holiday. Attempting to close with today's data on {dt.date()}.")
            _inc_monthly_stat(monthly_stats, dt, "close_attempted_overdue")
            pos.close_date = dt

        _inc_monthly_stat(monthly_stats, dt, "close_attempted")

        
        try:
            opt = opt_grouped.get_group(pos.ticker).copy()
        except KeyError:
            logger.error(f"Options data for ticker {pos.ticker} not found on {dt.date()}. Deleting position.")
            _inc_monthly_stat(monthly_stats, dt, "close_fail_missing_ticker")
            continue
        assert not opt.empty

        # if stock split detected, adjust spot
        stock_splited = False
        spot = opt["stkPx"].iloc[0]
        stock_split_df_ticker = stock_split_df[stock_split_df["ticker"] == pos.ticker]
        for row in stock_split_df_ticker.itertuples():
            # row is a namedtuple with fields corresponding to the columns of stock_split_df
            if pos.open_date < pd.Timestamp(row.splitDate) and pos.close_date >= pd.Timestamp(row.splitDate):
                stock_splited = True
                spot = spot * row.divisor
                logger.info(f"Stock split detected for {pos.ticker} on {row.splitDate}. Adjusted spot price from {opt['stkPx'].iloc[0]} to {spot}")

        # Compute PnL based on strategy
        # NOTE: Repeating code doesn't hurt since we are using if/elif
        if pos.strategy == "short_straddle_hold":
            pos.exit_cashflow = - max(0, spot - pos.strike_50d_30dte) - max(0, pos.strike_50d_30dte - spot)
        
        elif pos.strategy == "short_straddle_1w":
            if stock_splited:
                _inc_monthly_stat(monthly_stats, dt, "close_fail_stock_split")
                continue
            c = option_price(opt, pos.near_expiry, pos.strike_50d_30dte, call=True)
            p = option_price(opt, pos.near_expiry, pos.strike_50d_30dte, call=False)
            if c is None or p is None:
                logger.debug(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                _inc_monthly_stat(monthly_stats, dt, "close_fail_missing_option_price")
                continue
            pos.exit_cashflow = - c - p

        elif pos.strategy == "long_call_butterfly_hold":
            c1 = max(0, spot - pos.strike_15d_30dte)
            c2 = max(0, spot - pos.strike_50d_30dte)
            c3 = max(0, spot - pos.strike_85d_30dte)
            # NOTE: c1 and c3 are symmetric in the calculation, so no need to worry about which one is which
            exit_cashflow = c1 - 2 * c2 + c3
            assert exit_cashflow / pos.notional >= -1, f"Butterfly loss exceeds notional.\nTicker: {pos.ticker}\nK1: {pos.strike_85d_30dte}\nK2: {pos.strike_50d_30dte}\nK3: {pos.strike_15d_30dte}\nExit cashflow: {exit_cashflow:.2f}\nNotional: {pos.notional:.2f}"
            pos.exit_cashflow = exit_cashflow

        elif pos.strategy == "long_call_calendar_30_60":
            if stock_splited:
                _inc_monthly_stat(monthly_stats, dt, "close_fail_stock_split")
                continue
            near_c = max(0, spot - pos.strike_50d_30dte)
            far_c = option_price(opt, pos.far_expiry, pos.strike_50d_60dte, call=True)
            if far_c is None:
                logger.debug(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                _inc_monthly_stat(monthly_stats, dt, "close_fail_missing_option_price")
                continue
            pos.exit_cashflow = - near_c + far_c

        elif pos.strategy == "long_straddle_calendar_30_60":
            if stock_splited:
                _inc_monthly_stat(monthly_stats, dt, "close_fail_stock_split")
                continue
            near_c = max(0, spot - pos.strike_50d_30dte)
            near_p = max(0, pos.strike_50d_30dte - spot)
            far_c = option_price(opt, pos.far_expiry, pos.strike_50d_60dte, call=True)
            far_p = option_price(opt, pos.far_expiry, pos.strike_50d_60dte, call=False)
            if None in [far_c, far_p]:
                logger.debug(f"Missing option prices for {pos.strategy} [{pos.ticker}] opened on {pos.open_date.date()} when closing on {dt.date()}. Deleting position.")
                _inc_monthly_stat(monthly_stats, dt, "close_fail_missing_option_price")
                continue
            pos.exit_cashflow = - near_c - near_p + far_c + far_p

        # Update actual close date if it differs from planned
        pos.spot_at_close = spot
        pos.close_date = dt
        closed_trades.append(asdict(pos))
        _inc_monthly_stat(monthly_stats, dt, "close_success")

    return still_open, closed_trades
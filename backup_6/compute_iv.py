import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from helper_code.options_reader import read_options_data

BASE_DIR = Path(__file__).resolve().parent

def compute_forward(S, r, q, T):
    return S * np.exp((r - q) * T)


def build_otm_Q(chain, F):
    """
    Given a single (ticker, trade_date, expiry) chain with cMid/pMid, build Q(K):
      K < K0 => put mid
      K > K0 => call mid
      K = K0 => avg(call, put)
    """
    chain = chain.sort_values("strike").copy()
    
    # Convert columns to numpy arrays
    strikes = chain["strike"].to_numpy()
    c = chain["cValue"].to_numpy(dtype=float)
    p = chain["pValue"].to_numpy(dtype=float)

    # Compute K0
    strikes = np.asarray(strikes)
    below = strikes[strikes <= F]
    if below.size == 0:
        K0 = strikes.min()
        logger.warning(f"No strikes below forward price F={F:.2f}, using min strike K0={K0:.2f}")
        print(chain)
    else:
        K0 = below.max()
    Q = np.full(len(chain), np.nan, dtype=float)
    idx0 = np.where(strikes == K0)[0][0] # if K0 in strikes else 0

    for i in range(len(chain)):
        if strikes[i] < strikes[idx0]:
            Q[i] = p[i]
        elif strikes[i] > strikes[idx0]:
            Q[i] = c[i]
        else:
            Q[i] = np.nanmean([c[i], p[i]])

    # Apply truncation on put wing: walk downward from idx0-1
    zeros = 0
    cut_put = -1  # all indices <= cut_put will be excluded (set to NaN)
    for i in range(idx0 - 1, -1, -1):
        if Q[i] == 0:
            zeros += 1
        else:
            zeros = 0
        if zeros >= 2:
            cut_put = i  # this is the second zero; everything below (<= i) excluded
            break
    if cut_put >= 0:
        Q[:cut_put + 1] = np.nan

    # Apply truncation on call wing: walk upward from idx0+1
    zeros = 0
    cut_call = len(chain)  # all indices >= cut_call will be excluded
    for i in range(idx0 + 1, len(chain)):
        if Q[i] == 0:
            zeros += 1
        else:
            zeros = 0
        if zeros >= 2:
            cut_call = i  # second zero; everything above (>= i) excluded
            break
    if cut_call < len(chain):
        Q[cut_call:] = np.nan

    chain["Q"] = Q
    return chain, K0


def compute_iv_ticker(
    chain: pd.DataFrame,
    ticker,
    target_dte=30, # target days to expiry
):
    
    expiry_stats = chain.groupby("expirDate")["dte"].median()
    exp = (expiry_stats - target_dte).abs().idxmin()
    chain = chain[chain["expirDate"] == exp].copy()
    chain = chain.sort_values("strike")

    S = float(chain["stkPx"].iloc[0])
    r = float(chain["iRate"].median()) if "iRate" in chain.columns else 0.0
    q = float(chain["divRate"].median()) if "divRate" in chain.columns else 0.0
    T = float(chain["yte"].iloc[0])
    F = compute_forward(S, r, q, T)

    # compute atm iv
    row = chain.iloc[(chain["strike"] - S).abs().argsort()[:1]]
    atm_iv = float(row["vol"].iloc[0])

    # compute iv30 using model-free formula
    chain, K0 = build_otm_Q(chain, F)
    chain = chain.sort_values("strike")
    K = chain["strike"].to_numpy()
    Q = chain["Q"].to_numpy()

    if len(K) < 2 or not np.isfinite(K0) or K0 <= 0:
        logger.warning(f"Not enough valid strikes to compute IV for ticker {ticker} on expiry {exp}.")
        return None


    # Compute integral
    dK = np.empty_like(K)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0

    mask = np.isfinite(Q) & np.isfinite(dK) & np.isfinite(K) & (K > 0)
    if mask.sum() < 3:
        logger.warning(f"Not enough valid strikes to compute IV for ticker {ticker} on expiry {exp}. Valid points: {mask.sum()}")
        return None  # not enough valid strikes to compute IV
    
    integral = np.sum(Q[mask] * dK[mask] / (K[mask] ** 2))
    var = (2.0 / T) * np.exp(r * T) * integral - (1.0 / T) * (F / K0 - 1.0) ** 2

    return {
        "ticker": ticker,
        "trade_date": chain["trade_date"].iloc[0],
        "expiry": exp,
        "yte": T,
        "stkPx": S,
        "implied_var30": var,
        "iv30": np.sqrt(max(var, 0.0)),
        "atm_iv30": atm_iv,
    }


def compute_iv(
    option_today: pd.DataFrame,
    tickers: list[str],
    target_dte=30,
    dte_window=(25, 35),
):
    
    option_today = option_today[option_today["dte"].between(dte_window[0], dte_window[1])]
    options_grouped = option_today.groupby("ticker")

    ivs = []
    
    for ticker in tickers:
        if ticker not in options_grouped.groups:
            logger.info(f"Ticker {ticker} not found in options data, skipping.")
            continue
        
        chain = options_grouped.get_group(ticker)
        iv = compute_iv_ticker(chain, ticker, target_dte=target_dte)
        if iv:
            ivs.append(iv)
    
    return pd.DataFrame(ivs)
    

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    day_zip = BASE_DIR / "data" / "2024" / f"ORATS_SMV_Strikes_{20240102}.zip"
    cols = [
        "ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "yte", "strike",
            "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol", "iRate", "divRate",
    ]
    df = read_options_data(day_zip, tickers=tickers, cols=cols)
    ivs = compute_iv(df, tickers, target_dte=30, dte_window=(25, 35))
    print(ivs)

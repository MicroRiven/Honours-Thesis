import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from helper_code.options_reader import read_options_data


def compute_forward(S, r, q, T):
    return S * np.exp((r - q) * T)

def compute_k0(strikes, F):
    strikes = np.asarray(strikes)
    below = strikes[strikes <= F]
    return below.max() if below.size else strikes.min()

def delta_k(strikes):
    k = np.asarray(strikes)
    dk = np.zeros_like(k, dtype=float)
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    return dk

def build_otm_Q(chain, F):
    """
    Given a single (ticker, trade_date, expiry) chain with cMid/pMid, build Q(K):
      K < K0 => put mid
      K > K0 => call mid
      K = K0 => avg(call, put)
    """
    chain = chain.sort_values("strike").copy()
    strikes = chain["strike"].to_numpy()
    K0 = compute_k0(strikes, F)

    Q = np.full(len(chain), np.nan, dtype=float)
    for i, row in enumerate(chain.itertuples(index=False)):
        K = row.strike
        c = row.cValue
        p = row.pValue
        if K < K0:
            Q[i] = p
        elif K > K0:
            Q[i] = c
        else:
            Q[i] = np.nanmean([c, p])
    chain["Q"] = Q
    return chain, K0

def plot_vix_integrand_from_orats(
    options_df,
    ticker,
    target_dte=30,           # target days to expiry
    dte_window=(20, 40),     # acceptable DTE window to pick an expiry
):
    
    sub = options_df[options_df["ticker"] == ticker]

    # choose expiry closest to target_dte within window
    cand = sub[sub["dte"].between(dte_window[0], dte_window[1])]
    if cand.empty:
        raise ValueError(f"No expiries in DTE window {dte_window} for {ticker}")
    expiry_stats = cand.groupby("expirDate")["dte"].median()
    exp = (expiry_stats - target_dte).abs().idxmin()
    chain = cand[cand["expirDate"] == exp].copy()

    # spot/rates/T
    S = float(chain["stkPx"].iloc[0])
    r = float(chain["iRate"].iloc[0]) if "iRate" in chain.columns else 0.0
    q = float(chain["divRate"].iloc[0]) if "divRate" in chain.columns else 0.0
    T = float(chain["yte"].median())

    F = compute_forward(S, r, q, T)

    # Build Q(K) and K0
    chain, K0 = build_otm_Q(chain, F)

    # clean
    chain = chain.dropna(subset=["Q"]).sort_values("strike")
    chain["integrand"] = chain["Q"] / (chain["strike"] ** 2)
    
    # ========= Plot =========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    # Top subplot: Q(K)
    ax1.plot(chain["strike"], chain["Q"], marker='o', label="Q(K)")
    ax1.axvline(F, linestyle="--", color='red', label=f"Forward F={F:.2f}")
    ax1.axvline(K0, linestyle="--", label=f"K0={K0:.2f}")
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Q(K)")
    ax1.legend()
    ax1.set_title(f"[{ticker}] expiry {exp.date()} (DTE~{int(chain['dte'].median())})")
    # Bottom subplot: Integrand
    ax2.plot(chain["strike"], chain["integrand"], marker='o', label="Q(K)/K²")
    ax2.axvline(F, linestyle="--", color='red', label=f"Forward F={F:.2f}")
    ax2.axvline(K0, linestyle="--", label=f"K0={K0:.2f}")
    ax2.set_title(f"[{ticker}] expiry {exp.date()} (DTE~{int(chain['dte'].median())})")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Q(K) / K²")
    ax2.legend()
    plt.show()

    return chain, {"ticker": ticker, "expiry": exp, "S": S, "F": F, "K0": K0, "T": T}

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    day_zip = BASE_DIR / "data" / "2024" / f"ORATS_SMV_Strikes_{20240102}.zip"
    cols = [
        "ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "yte", "strike",
            "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol", "iRate", "divRate",
    ]
    df = read_options_data(day_zip, tickers = ["SPY", "NFLX"], cols=cols)

    # Example calls (these will only work if your CSV contains the tickers)
    chain_spy, meta_spy = plot_vix_integrand_from_orats(df, ticker="SPY")
    chain_nflx, meta_nflx = plot_vix_integrand_from_orats(df, ticker="NFLX")
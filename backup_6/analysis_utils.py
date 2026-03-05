import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm



def _month_range(start_month: str, end_month: str) -> list[pd.Period]:
    """
    start_month, end_month: 'YYYY-MM'
    """
    start = pd.Period(start_month, freq="M")
    end   = pd.Period(end_month, freq="M")
    return list(pd.period_range(start, end, freq="M"))


def load_span(file_path: Path, start_ym: str, end_ym: str, prefix: str):
    
    dfs = []
    for ym in _month_range(start_ym, end_ym):
        file_name = file_path / f"{prefix}_{ym.strftime('%Y-%m')}.parquet"
        assert file_name.exists(), f"File not found: {file_name}"
        
        df = pd.read_parquet(file_name)
        dfs.append(df)

    if not dfs:
        raise ValueError("No parquet files loaded")

    return pd.concat(dfs, ignore_index=True)

# ==================================
# VRP Analysis
# ==================================

def combine_iv_rv(signals_df, ivs_df):
    """
    Compare realized volatility (RV) with implied volatility (IV).
    Args:
        signals_df: DataFrame containing rv30 column
        ivs_df: DataFrame containing iv column
    Returns:
        DataFrame with ticker, trade_date, rv30, iv30, and forward_rv30 (rv30 from 30 days ahead)
    """
    # Calculate rvar30 if not already present
    if "rvar30" not in signals_df.columns:
        signals_df["rvar30"] = signals_df["rv30_yz"] ** 2

    # Merge the dataframes on ticker and trade_date, keeping only specified columns
    merged = signals_df[["ticker", "trade_date", "rv30_yz", "rv30_cc", "rvar30"]].merge(
        ivs_df[["ticker", "trade_date", "iv30", "implied_var30"]],
        on=["ticker", "trade_date"],
        how="inner"
    )
    
    # Sort by ticker and trade_date to ensure proper ordering for shift operation
    merged = merged.sort_values(["ticker", "trade_date"]).reset_index(drop=True) # grouped by ticker, sorted by trade_date within each ticker
    merged["forward_rvar30"] = merged.groupby("ticker")["rvar30"].shift(-21)
    merged["forward_rv30"] = np.sqrt(merged["forward_rvar30"])
    merged["vrp"] = merged["implied_var30"] - merged["forward_rvar30"]

    return merged

def summarise_vrp_ticker(vrp_df_ticker, ticker, hac_lags=21):
    """
    nonoverlap:
      - "every_21d": take every 21st trading-day observation (approx 1m, no overlap)
      - "month_start": take first observation in each calendar month
    """
    df = vrp_df_ticker.copy()

    # keep and clean
    df = df[["trade_date", "implied_var30", "forward_rvar30", "forward_rv30", "vrp"]].dropna()
    df = df.sort_values("trade_date")

    mean_ivar = df["implied_var30"].mean()
    mean_rvar = df["forward_rvar30"].mean()
    mean_vrp = df["vrp"].mean()
    var_vrp = df["vrp"].var()
    skew_vrp = df["vrp"].skew()

    # -----------------------
    # 1) Naive IID t-stat
    # -----------------------
    n = len(df)
    assert n > 1, "Need at least 2 data points for t-test"
    se_naive = df["vrp"].std(ddof=1) / np.sqrt(n)
    t_naive = mean_vrp / se_naive
    p_naive_one_sided = 1 - stats.t.cdf(t_naive, df=n - 1)

    # -----------------------
    # 2) HAC (Newey–West) t-stat via OLS-on-constant
    # -----------------------
    y = df["vrp"].values
    X = np.ones((len(y), 1))  # constant only
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    t_hac = float(res.tvalues[0])
    p_hac_two_sided = float(res.pvalues[0])
    # convert to one-sided test mean>0
    p_hac_one_sided = p_hac_two_sided / 2 if t_hac > 0 else 1 - p_hac_two_sided / 2

    # -----------------------
    # 3) Non-overlapping monthly-ish sample t-stat
    # -----------------------
    df_no = df.iloc[::21].copy()
    n_no = len(df_no)
    if n_no > 1:
        mean_no = df_no["vrp"].mean()
        se_no = df_no["vrp"].std(ddof=1) / np.sqrt(n_no)
        t_no = mean_no / se_no
        p_no_one_sided = 1 - stats.t.cdf(t_no, df=n_no - 1)
    else:
        mean_no, se_no, t_no, p_no_one_sided = (np.nan, np.nan, np.nan, np.nan)

    return {
        "ticker": ticker,
        "mean_implied_var": round(mean_ivar, 3),
        "mean_rvar": round(mean_rvar, 3),
        "mean_vrp": round(mean_vrp, 4),
        "var_vrp": round(var_vrp, 4),
        "skew_vrp": round(skew_vrp, 4),

        # Naive
        "t_naive": round(float(t_naive), 2),
        "p_naive_one_sided": round(float(p_naive_one_sided), 4),

        # HAC
        "t_hac": round(float(t_hac), 2),
        "p_hac_one_sided": round(float(p_hac_one_sided), 4),

        # Non-overlapping
        "t_nonoverlap": round(float(t_no), 2) if np.isfinite(t_no) else np.nan,
        "p_nonoverlap_one_sided": round(float(p_no_one_sided), 4) if np.isfinite(p_no_one_sided) else np.nan,
    }

def plot_vrp(vrp_df, ticker: str, start_date = None, end_date = None, path = None):

    vrp_df["volrp"] = vrp_df["iv30"] - vrp_df["forward_rv30"]
    mean_VarRP = vrp_df["vrp"].mean()
    mean_VolRP = vrp_df["volrp"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    ax1, ax3, ax2, ax4 = axes.flatten()

    # Subplot 1: implied_var30 vs forward_rvar30
    ax1.plot(vrp_df["trade_date"], vrp_df["implied_var30"], label="Implied Var", linewidth=2)
    ax1.plot(vrp_df["trade_date"], vrp_df["forward_rvar30"], label="Forward Realised Var (Yang-Zhang)", linewidth=2)
    ax1.set_ylabel("Value")
    ax1.set_title(f"Implied Variance vs Realised Variance (Forward) for {ticker}")
    ax1.legend()

    # Subplot 2: Variance Risk Premium
    ax2.plot(vrp_df["trade_date"], vrp_df["vrp"], label="Variance Risk Premium", linewidth=2)
    ax2.axhline(0, linestyle="-", linewidth=1, color = "black")
    ax2.axhline(mean_VarRP, linestyle="--", linewidth=0.5, label=f"Mean: {mean_VarRP:.4f}", color="black")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Variance Risk Premium for {ticker}")
    ax2.legend()

    # Subplot 3: iv30 vs rv30
    ax3.plot(vrp_df["trade_date"], vrp_df["iv30"], label="IV", linewidth=2, color="orange")
    ax3.plot(vrp_df["trade_date"], vrp_df["forward_rv30"], label="Forward RV (Yang-Zhang)", linewidth=2, color="green")
    ax3.set_ylabel("Value")
    ax3.set_title(f"Implied Vol vs Realised Vol (Forward) for {ticker}")
    ax3.legend()

    # Subplot 4: Volatility Risk Premium
    ax4.plot(vrp_df["trade_date"], vrp_df["volrp"], label="Volatility Risk Premium", linewidth=2, color="purple")
    ax4.axhline(0, linestyle="-", linewidth=1, color = "black")
    ax4.axhline(mean_VolRP, linestyle="--", linewidth=0.5, label=f"Mean: {mean_VolRP:.4f}", color="black")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Value")
    ax4.set_title(f"Volatility Risk Premium for {ticker}")
    ax4.legend()

    # Slice the dataframe for the specified date range
    if start_date is not None and end_date is not None:
        ax1.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    else:
        ax1.set_xlim(pd.to_datetime("2019-01-01"), pd.to_datetime("2025-11-30"))

    plt.tight_layout()

    if path is not None:
        plt.savefig(path / f"vrp_{ticker}.png", bbox_inches="tight")

    plt.show()


# this function is archived
def plot_vrp_pnl(vrp_df_ticker, ticker, margin = 0.5, start_date = None, end_date = None):

    vrp_df_ticker["return"] = vrp_df_ticker["vrp"] / margin
    vrp_df_ticker["cum_return"] = (1 + vrp_df_ticker["return"]).cumprod() * 100
   
    # plot the return and cumulative return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(vrp_df_ticker["trade_date"], vrp_df_ticker["return"], label="Return", linewidth=2)
    ax1.set_ylabel("Return")
    ax1.set_title(f"Return and Cumulative Return of VRP Strategy for {ticker}")
    ax1.legend()

    ax2.plot(vrp_df_ticker["trade_date"], vrp_df_ticker["cum_return"], label="Cumulative Return", linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title(f"Cumulative Returns of VRP Strategy for {ticker}")
    ax2.legend()

    # compute sharpe of return and cumulative return
    sharpe = vrp_df_ticker["cum_return"].mean() / vrp_df_ticker["cum_return"].std() * np.sqrt(252)
    print(f"Sharpe Ratio of Cumulative Return for {ticker}: {sharpe:.2f}")

    # Slice the dataframe for the specified date range
    if start_date is not None and end_date is not None:
        ax1.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    else:
        ax1.set_xlim(pd.to_datetime("2019-01-01"), pd.to_datetime("2025-11-30"))

    plt.tight_layout()
    plt.show()


# ==================================
# Trade Analysis
# ==================================


def analyse_trades(df, log = False):
    # Summarise the pnl column
    summary = df["pnl_norm"].describe()
    print("PnL Summary:")
    print(summary)

    # Additional analysis
    print("\nAdditional Analysis:")
    print(f"Total PnL: {df["pnl_norm"].sum():.2f}")
    print(f"Average PnL: {df["pnl_norm"].mean():.2f}")
    print(f"Median PnL: {df["pnl_norm"].median():.2f}")
    print(f"Number of positive trades: {(df["pnl_norm"] > 0).sum()}")
    print(f"Number of negative trades: {(df["pnl_norm"] < 0).sum()}")
    print(f"Max PnL: {df["pnl_norm"].max():.2f}")
    print(f"Min PnL: {df["pnl_norm"].min():.2f}")

    if log:
        plt.figure(figsize=(10, 5))
        # Create two subplots: left for normal, right for log scale
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        # Normal histogram
        ax1.hist(df["pnl_norm"], bins=30, color='skyblue', edgecolor='black')
        ax1.set_title("PnL Distribution (Normal)")
        ax1.set_xlabel("Normalised PnL")
        ax1.set_ylabel("Frequency")
        ax1.grid(True)

        # Logged histogram
        ax2.hist(df["pnl_norm"], bins=30, color='skyblue', edgecolor='black')
        ax2.set_title("PnL Distribution (Log Frequency)")
        ax2.set_xlabel("Normalised PnL")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale("log")
        ax2.grid(True)
    else:
        plt.figure(figsize=(8, 5))
        plt.hist(df["pnl_norm"], bins=30, color='skyblue', edgecolor='black')
        plt.title("PnL Distribution")
        plt.xlabel("Normalised PnL")
        plt.ylabel("Frequency")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_pnl_over_time(df):
    plt.figure(figsize=(10, 5))
    df_sorted = df.sort_values(by="trade_date")
    plt.plot(df_sorted["trade_date"], df_sorted["pnl_norm"].cumsum(), marker='o', linestyle='-')
    plt.title("Cumulative PnL Over Time")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
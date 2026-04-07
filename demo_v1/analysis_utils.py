"""Shared analysis and plotting utilities for VRP and backtest notebooks.

Sections:
  - Data loading helpers
  - VRP analysis (composite signals, per-ticker VRP summary, VRP plots)
  - Trade analysis (PnL distributions, signal histograms, scatter plots)
  - ML evaluation (classification report, ROC curves)
  - Portfolio analytics (equity curves, drawdowns, Monte Carlo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates



def _month_range(start_month: str, end_month: str) -> list[pd.Period]:
    """Generate a list of monthly periods between start_month and end_month (inclusive)."""
    start = pd.Period(start_month, freq="M")
    end   = pd.Period(end_month, freq="M")
    return list(pd.period_range(start, end, freq="M"))


def load_span(file_path: Path, start_ym: str, end_ym: str, prefix: str):
    """Load and concatenate monthly parquet files for [start_ym, end_ym]."""
    dfs = []
    for ym in _month_range(start_ym, end_ym):
        file_name = file_path / f"{prefix}_{ym.strftime('%Y-%m')}.parquet"
        assert file_name.exists(), f"File not found: {file_name}"
        
        df = pd.read_parquet(file_name)
        dfs.append(df)

    if not dfs:
        raise ValueError("No parquet files loaded")

    return pd.concat([df for df in dfs if not df.empty], ignore_index=True)

# ==================================
# VRP Analysis
# ==================================


def compute_composite_signal(df: pd.DataFrame):
    """Derive composite VRP signals in-place (market IV, IV/RV ratio, forward factor, term-structure slopes)."""

    df["market_iv30"] = np.sqrt(df["ivar30"])
    df["iv30_rv30"] = df["atm_iv30"] / df["rvol30_yz"]

    # Forward factor: implied vol premium of near-term over forward vol
    df["fwd_var_30_60"] = (df["atm_iv60"]**2 * 60.0 - df["atm_iv30"]**2 * 30.0) / 30.0
    df["fwd_var_30_60"] = df["fwd_var_30_60"].clip(lower=0.0)  # clamp tiny negatives from interpolation / rounding
    df["fwd_iv_30_60"] = np.sqrt(df["fwd_var_30_60"])
    df["fwd_factor_30_60"] = np.where(
        df["fwd_iv_30_60"] > 0,
        (df["atm_iv30"] - df["fwd_iv_30_60"]) / df["fwd_iv_30_60"],
        np.nan
    )
    # Term-structure slopes (change in ATM IV per day of DTE)
    df["ts_slope_30"] = (df["atm_iv30"] - df["atm_iv0"]) / (30.0 - df["dte0"])
    df["ts_slope_45"] = (df["atm_iv45"] - df["atm_iv0"]) / (45.0 - df["dte0"])
    df["ts_slope_60"] = (df["atm_iv60"] - df["atm_iv0"]) / (60.0 - df["dte0"])


def summarise_vrp_ticker(vrp_df_ticker, ticker, ticker_volume, hac_lags=21):
    """Compute per-ticker VRP summary statistics with naive, HAC, and non-overlapping t-tests."""
    df = vrp_df_ticker.copy()

    # keep and clean
    df = df[["trade_date", "ivar30", "forward_rvar30", "forward_rvol30", "vrp"]].dropna()
    df = df.sort_values("trade_date")

    mean_ivar = df["ivar30"].mean()
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
    # 2) HAC (Newey-West) t-stat via OLS-on-constant
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
        "mean_ivar": round(mean_ivar, 4),
        "mean_rvar": round(mean_rvar, 4),
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

        "average_option_volume": round(float(ticker_volume), 0) if np.isfinite(ticker_volume) else np.nan,

    }


def plot_vrp(vrp_df, ticker: str, path=None, axes=None, show=True, save=True):
    """Two-panel chart: implied vs forward realised variance (top), variance risk premium (bottom)."""

    vrp_df = vrp_df.copy()
    vrp_df["volrp"] = vrp_df["market_iv30"] - vrp_df["forward_rvol30"]
    mean_VarRP = vrp_df["vrp"].mean()
    mean_VolRP = vrp_df["volrp"].mean()

    created_figure = axes is None
    if created_figure:
        fig, subplot_axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1, ax2 = subplot_axes.flatten()
    else:
        fig = axes[0].figure
        ax1, ax2 = axes

    # Subplot 1: ivar30 vs forward_rvar30
    ax1.plot(vrp_df["trade_date"], vrp_df["ivar30"], label="Implied Var", linewidth=2)
    ax1.plot(vrp_df["trade_date"], vrp_df["forward_rvar30"], label="Forward Realised Var (Yang-Zhang)", linewidth=2)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("Realised and Implied Variance")
    ax1.set_title(f"Implied Variance vs Forward Realised Variance for {ticker}")
    ax1.legend()

    # Subplot 2: Variance Risk Premium
    ax2.plot(vrp_df["trade_date"], vrp_df["vrp"], label="Variance Risk Premium", linewidth=2)
    ax2.axhline(0, linestyle="-", linewidth=1, color="black")
    ax2.axhline(mean_VarRP, linestyle="--", linewidth=0.3, label=f"Mean: {mean_VarRP:.4f}", color="red")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Variance Risk Premium")
    ax2.legend()

    # Use the data boundaries directly for the x-axis.
    trade_dates = pd.to_datetime(vrp_df["trade_date"]).dropna()
    if not trade_dates.empty:
        ax1.set_xlim(trade_dates.min(), trade_dates.max())

    if created_figure:
        fig.tight_layout()

    if path is not None and save:
        fig.savefig(path / f"ch4_vrp_{ticker}.png", bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax1, ax2)


# NOTE: this function is archived
def plot_vrp_pnl(vrp_df_ticker, ticker, margin=0.5, start_date=None, end_date=None):
    """[Archived] Plot simple VRP strategy return and cumulative return."""

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

def plot_monthly_pnl(df, target_col="pnl_norm", ax_month=None, ax_day=None):
    """Plot cross-sectional average PnL aggregated by month (top) and by day (bottom)."""

    df = df.copy()

    df["month"] = df["open_date"].dt.to_period("M")
    monthly_cs = (
        df.groupby("month")[target_col]
        .mean()
        .reset_index()
    )
    monthly_cs["month_ts"] = monthly_cs["month"].dt.to_timestamp()

    daily_cs = (
        df.groupby(df["open_date"].dt.floor("D"))[target_col]
        .mean()
        .reset_index(name=target_col)
        .rename(columns={"open_date": "day_ts"})
    )

    # Top subplot: monthly average
    ax_month.plot(monthly_cs["month_ts"], monthly_cs[target_col])
    ax_month.set_ylabel("Average normalised PnL")
    ax_month.set_title("Cross-sectional average monthly PnL")

    # Bottom subplot: daily average
    ax_day.plot(daily_cs["day_ts"], daily_cs[target_col], color="tab:orange", linewidth=1.0, alpha=0.8)
    ax_day.set_xlabel("Date")
    ax_day.set_ylabel("Average normalised PnL")
    ax_day.set_title("Cross-sectional average daily PnL")

    # Limit x-axis to first and last available day
    if not daily_cs.empty:
        x_min = daily_cs["day_ts"].min()
        x_max = daily_cs["day_ts"].max()
        ax_day.set_xlim(x_min, x_max)

    # choose "year" or "quarter"
    label_freq = "year"
    if label_freq == "year":
        ax_day.xaxis.set_major_locator(mdates.YearLocator())
        ax_day.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax_day.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        def _quarter_formatter(x, pos):
            dt = mdates.num2date(x)
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{quarter}"

        ax_day.xaxis.set_major_formatter(FuncFormatter(_quarter_formatter))

    plt.setp(ax_day.get_xticklabels(), rotation=45, ha="right")


def analyse_trades(df, file_path=None, axes=None, print_results=False):
    """Summarise PnL distribution and plot histogram (normal + log-scale)."""
    summary = df["pnl_norm"].describe()
    result = {
        **{k: v for k, v in summary.items()},
        "positive_trades": int((df["pnl_norm"] > 0).sum()),
        "negative_trades": int((df["pnl_norm"] < 0).sum()),
    }

    owns_figure = axes is None
    if owns_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    else:
        ax1, ax2 = axes

    # Normal histogram
    ax1.hist(df["pnl_norm"], bins=40, rwidth=0.9)
    ax1.set_title("PnL Distribution")
    ax1.set_xlabel("Normalised PnL")
    ax1.set_ylabel("Frequency")
    stats_text = (
        f"Number of Trades: {len(df)}\n"
        f"Mean: {df['pnl_norm'].mean():.4f}\n"
        f"Median: {df['pnl_norm'].median():.4f}"
    )
    ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, verticalalignment='top')

    # Logged histogram
    ax2.hist(df["pnl_norm"], bins=40, rwidth=0.9)
    ax2.set_title("PnL Distribution (Log Frequency)")
    ax2.set_xlabel("Normalised PnL")
    ax2.set_ylabel("Frequency")
    ax2.set_yscale("log")

    if print_results:
        print("PnL Summary:")
        for k, v in result.items():
            if isinstance(v, (int, np.integer)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}")

    if owns_figure:
        plt.tight_layout()
        if file_path is not None:
            plt.savefig(file_path, dpi=300)
        plt.show()

    return result


def plot_pnl_over_time(df):
    """Simple cumulative PnL line chart over trade dates."""
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


# NOTE: FORWARD LOOKING!
def plot_rv_vs_pnl(df, name, ax_rv=None, ax_vrp=None):
    """Scatter PnL vs forward realised vol (left) and vs VRP (right) with best-fit lines."""

    # Trim outliers at 0.5% tails
    low_q, high_q = (0.005, 0.995)
    low_pnl = df["pnl_norm"].quantile(low_q)
    high_pnl = df["pnl_norm"].quantile(high_q)
    low_rv = df["forward_rvol30"].quantile(low_q)
    high_rv = df["forward_rvol30"].quantile(high_q)
    df = df[df["pnl_norm"].between(low_pnl, high_pnl) & df["forward_rvol30"].between(low_rv, high_rv)]

    # First subplot: PnL vs Forward RV
    plot_df_rv = df[["forward_rvol30", "pnl_norm"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x_rv = plot_df_rv["forward_rvol30"].to_numpy()
    y_rv = plot_df_rv["pnl_norm"].to_numpy()

    # fit a line of best fit for RV
    m_rv = b_rv = np.nan
    if len(x_rv) >= 2:
        m_rv, b_rv = np.polyfit(x_rv, y_rv, 1)
        x_line_rv = np.linspace(x_rv.min(), x_rv.max(), 200)
        y_line_rv = m_rv * x_line_rv + b_rv

    # Second subplot: PnL vs VRP
    plot_df_vrp = df[["vrp", "pnl_norm"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x_vrp = plot_df_vrp["vrp"].to_numpy()
    y_vrp = plot_df_vrp["pnl_norm"].to_numpy()

    # fit a line of best fit for VRP
    m_vrp = b_vrp = np.nan
    if len(x_vrp) >= 2:
        m_vrp, b_vrp = np.polyfit(x_vrp, y_vrp, 1)
        x_line_vrp = np.linspace(x_vrp.min(), x_vrp.max(), 200)
        y_line_vrp = m_vrp * x_line_vrp + b_vrp

    owns_figure = (ax_rv is None) or (ax_vrp is None)
    if owns_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        ax1, ax2 = ax_rv, ax_vrp

    # Left subplot: PnL vs Forward RV
    ax1.scatter(x_rv, y_rv, s=2, marker=".", alpha=0.3, linewidths=0, label="Trades")
    if len(x_rv) >= 2:
        ax1.plot(x_line_rv, y_line_rv, color="red", linewidth=1, label=f"Best fit: y = {m_rv:.4f}x + {b_rv:.4f}")
    ax1.set_xlabel("Forward-looking Realised Volatility")
    ax1.set_ylabel("Normalised PnL")
    ax1.set_title(f"PnL vs Forward RV - {name}")
    ax1.grid(True)
    ax1.legend()

    # Right subplot: PnL vs VRP
    ax2.scatter(x_vrp, y_vrp, s=2, marker=".", alpha=0.3, linewidths=0, label="Trades")
    if len(x_vrp) >= 2:
        ax2.plot(x_line_vrp, y_line_vrp, color="red", linewidth=1, label=f"Best fit: y = {m_vrp:.4f}x + {b_vrp:.4f}")
    ax2.set_xlabel("Volatility Risk Premium (VRP)")
    ax2.set_ylabel("Normalised PnL")
    ax2.set_title(f"PnL vs VRP - {name}")
    ax2.grid(True)
    ax2.legend()

    if owns_figure:
        plt.tight_layout()
        plt.show()


def plot_feature_scatter_static(
        df: pd.DataFrame,
        signal: str,
        alpha: float = 0.1,
        title = None,
        file_name: str = None):
    """Non-interactive scatter with best-fit line and equation in legend."""

    scatter_df = (
        df[[signal, "pnl_norm"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    x = scatter_df[signal].to_numpy()
    y = scatter_df["pnl_norm"].to_numpy()

    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else (1 - ss_res / ss_tot)

    order = np.argsort(x)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=0.5, alpha=alpha)
    plt.plot(x[order], y_hat[order], color="crimson", linewidth=0.5, label="Best-fit line")

    eq_label = f"y = {m:.4f}x + {b:.4f}\nR^2 = {r2:.4f}"
    plt.plot([], [], " ", label=eq_label)

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Scatter: {signal} vs pnl_norm")
    plt.xlabel(signal)
    plt.ylabel("Normalised PnL")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", framealpha=0.8, edgecolor="gray")

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300)
    plt.show()


def plot_feature_scatter_interactive(df: pd.DataFrame, signal: str, max_points: int = 20000):
    """Interactive scatter using WebGL (Scattergl) with optional downsampling."""

    scatter_df = (
        df[[signal, "pnl_norm", "ticker", "open_date"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    scatter_df["date_str"] = scatter_df["open_date"].dt.strftime("%Y-%m-%d")

    n_total = len(scatter_df)
    if n_total > max_points:
        scatter_df = scatter_df.sample(max_points, random_state=42)
        title_suffix = f" (sampled {max_points:,} of {n_total:,})"
    else:
        title_suffix = ""

    fig_scatter = go.Figure(
        go.Scattergl(
            x=scatter_df[signal].astype("float32"),
            y=scatter_df["pnl_norm"].astype("float32"),
            mode="markers",
            marker=dict(size=4, opacity=0.6),
            customdata=np.stack(
                [scatter_df["ticker"], scatter_df["date_str"]], axis=1
            ),
            hovertemplate=(
                f"{signal}: %{{x:.4f}}<br>"
                "pnl_norm: %{y:.4f}<br>"
                "Ticker: %{customdata[0]}<br>"
                "Date: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig_scatter.update_layout(
        title=f"Interactive Scatter: {signal} vs pnl_norm{title_suffix}",
        xaxis_title=signal,
        yaxis_title="Normalised PnL",
    )

    fig_scatter.show()


def plot_signal_hist(
    df: pd.DataFrame,
    signal: str,
    name: str = None,
    boundary: tuple[float, float] = (0.001, 0.999),
    log_volume: bool = False,
    file_path: str = None,
):
    """Three-panel signal histogram: boxplot, average PnL, and trade count by signal bin."""

    name = signal if name is None else name

    df = df[[signal, "pnl_norm"]].dropna().copy()

    # Shared robust range for signal-range plots
    low_q, high_q = boundary
    signal_min = df[signal].quantile(low_q)
    signal_max = df[signal].quantile(high_q)

    # Clip signal to robust range and bin
    n_bins = 50
    bin_edges = np.linspace(signal_min, signal_max, n_bins + 1)
    df["signal_bin"] = pd.cut(
        df[signal].clip(lower=signal_min, upper=signal_max),
        bins=bin_edges,
        include_lowest=True
    )
    binned_avg = (
        df.groupby("signal_bin", observed=True)["pnl_norm"]
        .mean()
        .reset_index(name="avg_pnl_norm")
    )
    binned_avg["bin_mid"] = binned_avg["signal_bin"].apply(lambda x: x.mid)

    # Per-bin samples for boxplot
    binned_samples = (
        df.groupby("signal_bin", observed=True)["pnl_norm"]
        .apply(list)
        .reset_index(name="pnl_samples")
    )
    binned_samples["bin_mid"] = binned_samples["signal_bin"].apply(lambda x: x.mid)

    #====================================================================
    # Three-panel figure: boxplot, average PnL bar, trade-count histogram
    #====================================================================

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
        constrained_layout=True,
    )
    fig.suptitle(f"Signal Histogram - {name}", fontsize=14, fontweight="bold")

    bin_width = (signal_max - signal_min) / n_bins

    # Plot 1: PnL distribution by signal bin (boxplot)
    ax1.boxplot(
        binned_samples["pnl_samples"].tolist(),
        positions=binned_samples["bin_mid"].to_numpy(),
        widths=bin_width * 0.8,
        showmeans=True,
        showfliers=False,
        manage_ticks=False,
        patch_artist=True,
        medianprops={"color": "tab:blue"},
        boxprops={"facecolor": "tab:blue", "alpha": 0.35, "edgecolor": "tab:blue"},
        whiskerprops={"color": "tab:blue", "alpha": 0.8},
        capprops={"color": "tab:blue", "alpha": 0.8},
    )
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Normalised PnL")
    ax1.grid(alpha=0.3)

    # Plot 2: average PnL by signal bin
    ax2.bar(
        binned_avg["bin_mid"],
        binned_avg["avg_pnl_norm"],
        width=bin_width * 0.9,
        edgecolor="white"
    )

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Average normalised PnL")
    ax2.grid(alpha=0.3)

    # Plot 3: trade count histogram by signal range
    plot_signal = df[signal].clip(lower=signal_min, upper=signal_max)
    ax3.hist(plot_signal, bins=n_bins, range=(signal_min, signal_max), edgecolor="white", rwidth=0.9)
    ax3.set_xlabel(name)
    ax3.set_ylabel("Trade count")
    if log_volume:
        ax3.set_yscale("log")
    ax3.grid(alpha=0.3)

    if file_path is not None:
        fig.savefig(file_path)
    plt.show()

# ==================================
# ML Evaluation
# ==================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import pandas as pd



def evaluate_logistic_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    quantile_threshold=None,
    absolute_threshold=None,
    ax_scores=None,
    ax_roc=None,
):
    """Evaluate a classifier: print classification reports, plot score distribution and ROC."""

    # y_pred = model.predict(X_train)
    y_score_train = model.predict_proba(X_train)[:, 1] # this should be better than a 0.5 threshold
    y_score_val = model.predict_proba(X_val)[:, 1]

    if quantile_threshold is not None:
        cutoff = np.quantile(y_score_val, quantile_threshold)
    elif absolute_threshold is not None:
        cutoff = absolute_threshold
    else:
        raise ValueError("Must specify either quantile_threshold or absolute_threshold")
    
    y_pred_train = (y_score_train >= cutoff).astype(int)
    y_pred_val = (y_score_val >= cutoff).astype(int)
    print(f"Absolute threshold: {cutoff:.4f}")
    print(f"Quantile threshold: {1 - y_pred_val.mean():.4f}")
    print(f"Class balance: {sum(y_train)} pass out of {len(y_train)} total")
    print(f"Ratio: {sum(y_train) / len(y_train):.4f}")
    print()

    for name, y_true, y_pred, y_score in [
        ("Train", y_train, y_pred_train, y_score_train),
        ("Val", y_val, y_pred_val, y_score_val),
    ]:
        print(f"--- {name} ---")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")
        print()    

    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_score_val)
    auc_train = roc_auc_score(y_train, y_score_train)
    auc_val = roc_auc_score(y_val, y_score_val)

    owns_figure = (ax_scores is None) or (ax_roc is None)
    if owns_figure:
        fig, (ax_scores, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))

    ax_scores.hist(y_score_train, bins=50, alpha=0.5, label="Train", rwidth=0.9)
    ax_scores.hist(y_score_val, bins=50, alpha=0.5, label="Validation", rwidth=0.9)
    ax_scores.set_xlabel("Predicted Probability")
    ax_scores.set_ylabel("Frequency")
    ax_scores.set_title("Distribution of Predicted Probabilities")
    ax_scores.legend()
    ax_scores.grid(alpha=0.3)

    ax_roc.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.4f})")
    ax_roc.plot(fpr_val, tpr_val, label=f"Validation ROC (AUC = {auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    if owns_figure:
        plt.tight_layout()
        plt.show()

    return y_pred_train, y_pred_val


# ==================================
# Portfolio Analytics
# ==================================

"""
Backtest Analytics for Trade-Level Data
========================================
Handles trades where:
  - You only know PnL at close (not during the trade)
  - Multiple trades can be open simultaneously
  - Trades have different durations (~30 days avg)

Key design choice: PnL is booked on close_date (realised PnL curve).
For capital-at-risk metrics, entry_cashflow is used as the margin/collateral.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


def build_portfolio_curve(
    df: pd.DataFrame,
    starting_capital: float = 100.0,
    fraction: float = 0.001, # percentage of portfolio per trade
) -> pd.DataFrame:
    """
    Builds a portfolio equity curve using constant fractional position sizing.

    Core logic:
      - Each trade uses `margin_fraction` of portfolio value AT open_date
    - Dollar PnL of trade = margin_fraction * portfolio_value_at_open * pnl_norm
      - Portfolio is updated at close_date when PnL is realised
      - Multiple trades closing the same day are all applied to that day's
        starting portfolio value (simultaneous, not sequential within a day)

    Why this is the honest approach given no intraday path:
      - We know portfolio value at open (from prior closes)
    - We don't know it during the trade - so we fix margin at open
      - This is equivalent to "committed capital" sizing, standard in
        options backtesting (e.g. tastytrade, OptionAlpha methodology)

    Returns DataFrame indexed by business date with columns:
        portfolio_value   - equity curve starting at starting_capital
        daily_return      - portfolio % return each day
        n_trades_closed   - how many trades closed that day
        open_interest     - how many trades are live that day
    """
    if df.empty:
        return pd.DataFrame(
            columns=['portfolio_value', 'daily_return', 'open_interest', 'n_trades_closed']
        )

    df = df.copy()
    df['capital_reserved'] = 0.0 # initialise a margin_reserved column
    df['open_date'] = pd.to_datetime(df['open_date']).dt.normalize() # normalise to midnight
    df['close_date'] = pd.to_datetime(df['close_date']).dt.normalize()

    # Group close events by date for efficient lookup
    opens_by_date = df.groupby('open_date').groups   # dict: date -> list of trade indices opening today
    closes_by_date = df.groupby('close_date').groups # dict: date -> list of trade indices closing today
    all_dates = pd.date_range(df['open_date'].min(), df['close_date'].max(), freq='B')


    # portfolio return
    pv_dict = {} # date -> portfolio value eod
    current_portfolio = starting_capital
    for date in all_dates:

        pnl_today = 0.0
        pv_dict[date] = current_portfolio # start with yesterday's portfolio value

        if date in opens_by_date:
            opening_indices = opens_by_date[date]
            for idx in opening_indices:
                # reserve margin for this trade
                df.loc[idx, 'capital_reserved'] = fraction * current_portfolio

        if date in closes_by_date:
            closing_indices = closes_by_date[date]
            for idx in closing_indices:
                row = df.loc[idx]
                dollar_pnl = row['capital_reserved'] * row['pnl_norm'] # realised PnL in dollars
                pnl_today += dollar_pnl

        current_portfolio += pnl_today
        pv_dict[date] = current_portfolio

    result = pd.DataFrame({'portfolio_value': pv_dict})


    # simple daily return
    simple_pv_dict = {}
    current_portfolio = starting_capital
    for date in all_dates:

        pnl_today = 0.0
        simple_pv_dict[date] = current_portfolio # start with yesterday's portfolio value

        if date in closes_by_date:
            closing_indices = closes_by_date[date]
            for idx in closing_indices:
                row = df.loc[idx]
                dollar_pnl = fraction * current_portfolio * row['pnl_norm'] # realised PnL in dollars
                pnl_today += dollar_pnl

        current_portfolio += pnl_today
        simple_pv_dict[date] = current_portfolio

    result['simple_portfolio_value'] = pd.Series(simple_pv_dict)


    # compute dataframe columns
    result['daily_return'] = result['portfolio_value'].pct_change()

    result['open_interest'] = 0
    result['n_trades_closed'] = 0
    for date in all_dates:
        open_mask  = (date >= df['open_date']) & (date <= df['close_date'])
        result.loc[date, 'open_interest']   = open_mask.sum() # how many trades are live today
        result.loc[date, 'trades_closed_today'] = (df['close_date'] == date).sum() # how many trades closed today
    result['trades_closed_total'] = result['trades_closed_today'].cumsum()

    return result


def _compute_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with:
        rolling_max   - running high-water mark
        drawdown      - current drawdown in dollars
        drawdown_pct  - drawdown as % of high-water mark
    """
    hwm = equity_curve.cummax()
    dd = equity_curve - hwm
    dd_pct = dd / hwm.replace(0, np.nan) * 100

    return pd.DataFrame({
        'rolling_max': hwm,
        'drawdown': dd,
        'drawdown_pct': dd_pct
    })


def max_drawdown_stats(equity_curve: pd.Series) -> dict:
    """Returns max drawdown details including duration."""
    dd_df = _compute_drawdown(equity_curve)
    max_dd = dd_df['drawdown'].min()
    max_dd_pct = dd_df['drawdown_pct'].min()
    trough_date = dd_df['drawdown'].idxmin()

    # Find the peak before the trough
    peak_date = equity_curve[:trough_date].idxmax()

    # Find recovery date (first date after trough where we exceed the peak)
    recovery_series = equity_curve[trough_date:][equity_curve[trough_date:] >= equity_curve[peak_date]]
    recovery_date = recovery_series.index[0] if len(recovery_series) > 0 else None

    dd_duration = (trough_date - peak_date).days
    recovery_duration = (recovery_date - trough_date).days if recovery_date else None

    return {
        'max_drawdown_$':    round(max_dd, 2),
        'max_drawdown_%':    round(max_dd_pct, 2),
        'peak_date':         peak_date.date(),
        'trough_date':       trough_date.date(),
        'recovery_date':     recovery_date.date() if recovery_date else 'Not recovered',
        'drawdown_duration_days':  dd_duration,
        'recovery_duration_days':  recovery_duration,
    }


def risk_metrics(daily: pd.DataFrame,
                 annual_rf: float = 0.04,
                 trading_days: int = 252) -> dict:
    """Compute annualised return, volatility, Sharpe ratio, and win rate from daily equity curve."""

    pv = daily['portfolio_value']
    returns = daily['daily_return'].dropna()
    daily_rf = annual_rf / trading_days

    ann_return = (1 + returns).prod() ** (trading_days / len(returns)) - 1
    ann_vol    = returns.std() * np.sqrt(trading_days)
    sharpe     = (ann_return - annual_rf) / ann_vol if ann_vol != 0 else np.nan

    winning       = returns[returns > 0]
    losing        = returns[returns < 0]
    profit_factor = (winning.sum() / abs(losing.sum())) if losing.sum() != 0 else np.nan

    return {
        'ann_return_%':       round(ann_return * 100, 2),
        'ann_volatility_%':   round(ann_vol * 100, 2),
        'sharpe_ratio':       round(sharpe, 3),
        'win_rate_%':         round(len(winning) / len(returns) * 100, 2),
        'avg_daily_return_%': round(returns.mean() * 100, 4),
        'best_day_%':         round(returns.max() * 100, 2),
        'worst_day_%':        round(returns.min() * 100, 2),
    }


def monte_carlo(daily: pd.DataFrame,
                n_simulations: int = 1000,
                n_days: int = 252,
                seed: int = 42,
                percentiles: list = [5, 25, 50, 75, 95]) -> dict:
    """Bootstrap Monte Carlo simulation: resample daily returns to project forward equity paths."""

    rng     = np.random.default_rng(seed)
    returns = daily['daily_return'].dropna().values

    start_equity = daily['portfolio_value'].iloc[-1]
    start_equity = 100

    # Resample daily returns with replacement
    sampled = rng.choice(returns, size=(n_simulations, n_days), replace=True)

    # Compound returns: V(t) = V(0) * prod(1 + r_i)
    paths = start_equity * np.cumprod(1 + sampled, axis=1)

    final_values = paths[:, -1]
    pct_paths    = {p: np.percentile(paths, p, axis=0) for p in percentiles}

    # Max drawdown per path: (peak - trough) / peak
    def path_max_dd(p):
        hwm = np.maximum.accumulate(p)
        return np.min((p - hwm) / hwm) * 100  # negative number

    expected_max_dd = float(np.mean([path_max_dd(p) for p in paths]))

    sim_stats = {
        'start_equity':             round(start_equity, 2),
        'median_final_equity':      round(float(np.median(final_values)), 2),
        'mean_final_equity':        round(float(np.mean(final_values)), 2),
        'p5_final_equity':          round(float(np.percentile(final_values, 5)), 2),
        'p95_final_equity':         round(float(np.percentile(final_values, 95)), 2),
        'prob_profit_%':            round(float(np.mean(final_values > start_equity)) * 100, 1),
        'prob_drawdown_20pct_%':    round(float(np.mean(
            np.min(paths, axis=1) < start_equity * 0.8)) * 100, 1),
        'expected_max_dd_%':        round(expected_max_dd, 2),
    }

    return {
        'paths':            paths,
        'percentile_paths': pct_paths,
        'final_values':     final_values,
        'stats':            sim_stats,
        'n_days':           n_days,
        'start_equity':     start_equity,
    }


def full_backtest_report(
    df: pd.DataFrame,
    fraction: float,
    starting_capital: float = 100.0,
    annual_rf: float = 0.05,
    mc_sims: int = 1000,
    mc_days: int = 252,
    file_path_pnl: str = None,
    file_path_mc: str = None,
    title: str = "Portfolio Return",
    print_results: bool = True,
    plot_portfolio: bool = True,
    plot_monte_carlo: bool = True,
) -> dict:
    """End-to-end backtest: build equity curve, compute risk metrics, run Monte Carlo, and plot."""

    daily = build_portfolio_curve(
        df,
        starting_capital=starting_capital,
        fraction=fraction,
    )
    dd_stats = max_drawdown_stats(daily['portfolio_value'])
    rm = risk_metrics(daily, annual_rf=annual_rf)
    mc = monte_carlo(daily, n_simulations=mc_sims, n_days=mc_days)

    if print_results:
        print("\n" + "=" * 50)
        print("  BACKTEST SUMMARY")
        print("=" * 50)
        print("\nRISK METRICS")
        for k, v in rm.items():
            print(f"   {k:<28} {v}")
        print("\nMAX DRAWDOWN")
        for k, v in dd_stats.items():
            print(f"   {k:<28} {v}")
        print(f"\nMONTE CARLO (next {mc_days} days * {mc_sims} sims)")
        for k, v in mc['stats'].items():
            print(f"   {k:<28} {v}")
        print("=" * 50)

    if plot_portfolio:
        _plot_pnl(daily, rm, dd_stats, file_path=file_path_pnl, title=title)
    if plot_monte_carlo:
        _plot_monte_carlo(mc, daily.index[-1], file_path=file_path_mc)

    return {
        'daily':          daily,
        'drawdown_stats': dd_stats,
        'risk_metrics':   rm,
        'monte_carlo':    mc,
    }


def _plot_pnl(
    daily: pd.DataFrame,
    rm: dict,
    dd_stats: dict,
    file_path: str = None,
    title: str = "Portfolio Return"
):
    """Four-panel equity chart: portfolio value, drawdown, open interest, and trades closed."""
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 1])

    # Panel 1: Portfolio Value
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(daily.index, daily['simple_portfolio_value'].values, linewidth=1.5, color='orange', label='Simple Daily Sizing')
    ax1.plot(daily.index, daily['portfolio_value'].values, linewidth=1.5, label='Reserved Capital Sizing')
    ax1.set_title('Portfolio Value')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(daily.index[0], daily.index[-1])
    ax1.grid(True, alpha=0.3)
    stats_text = (f"Ann. Return: {rm['ann_return_%']:.2f}%\n"
                  f"Ann. Vol: {rm['ann_volatility_%']:.2f}%\n"
                  f"Sharpe: {rm['sharpe_ratio']:.3f}")
    ax1.text(0.98, 0.02, stats_text,
             transform=ax1.transAxes,
             verticalalignment='bottom',
             horizontalalignment='right')

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    # changed: compute_drawdown on portfolio_value
    dd_df = _compute_drawdown(daily['portfolio_value'])
    ax2.fill_between(dd_df.index, dd_df['drawdown_pct'].values, alpha=0.4)
    ax2.plot(dd_df.index, dd_df['drawdown_pct'].values, linewidth=0.8)
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.98, 0.02, f"Max DD: {dd_stats['max_drawdown_%']:.2f}%",
             transform=ax2.transAxes,
             verticalalignment='bottom',
             horizontalalignment='right')

    # Panel 3: Number of open trades
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(daily.index, daily['open_interest'].values, linewidth=0.8)
    ax3.set_ylabel('Open Interest')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Daily and cumulative trades closed
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    trades_closed_today = daily['trades_closed_today']
    trades_closed_cum = trades_closed_today.cumsum()
    ax4.bar(daily.index, trades_closed_today.values, width=1.0, alpha=0.35, label='Closed Today')
    ax4.set_ylabel('Closed Today')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    ax4_right = ax4.twinx()
    ax4_right.plot(daily.index, trades_closed_cum.values, color='orange', label='Cumulative Closed')
    ax4_right.set_ylabel('Cumulative Closed')

    lines_left, labels_left = ax4.get_legend_handles_labels()
    lines_right, labels_right = ax4_right.get_legend_handles_labels()
    ax4.legend(lines_left + lines_right, labels_left + labels_right, loc='upper left', fontsize=8)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if file_path is not None:
        fig.savefig(file_path, dpi=300)
    plt.show()


def _plot_monte_carlo(mc: dict, start_date: pd.Timestamp, file_path: str = None):
    """Plot Monte Carlo fan chart (percentile bands) with terminal distribution sidebar."""

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.05)

    # Monte Carlo Paths
    ax4 = fig.add_subplot(gs[0, 0])
    future_dates = pd.date_range(start_date, periods=mc['n_days'], freq='B')
    pp = mc['percentile_paths']
    ax4.fill_between(future_dates, pp[5],  pp[95], alpha=0.2, label='5-95%')
    ax4.fill_between(future_dates, pp[25], pp[75], alpha=0.3, label='25-75%')
    ax4.plot(future_dates, pp[50], linewidth=1.5, label='Median')
    ax4.axhline(mc['start_equity'], linewidth=1, linestyle='--', alpha=0.8)
    ax4.legend(fontsize=8)
    ax4.set_title(f'Monte Carlo ({mc["n_days"]} day forward simulation)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.set_xlim(future_dates[0], future_dates[-1])
    ax4.margins(x=0)
    ax4.grid(True, alpha=0.3)

    # Terminal Value Distribution (rotated, right side)
    ax5 = fig.add_subplot(gs[0, 1], sharey=ax4)
    final = mc['final_values']
    ax5.hist(final, bins=60, alpha=0.3, orientation='horizontal', rwidth=0.9)
    ax5.axhline(mc['start_equity'], linewidth=1, linestyle='--', label=f'Start: ${mc["start_equity"]:,.1f}')
    ax5.axhline(float(np.median(final)), linewidth=1.5, label=f'Median: ${np.median(final):,.1f}')
    ax5.legend(fontsize=7)
    ax5.set_title('Terminal Distribution')
    ax5.set_xlabel('Frequency')
    ax5.tick_params(axis='y', labelleft=False)
    ax5.grid(True, alpha=0.3)

    fig.suptitle('Monte Carlo Analysis', fontsize=14, fontweight='bold')

    if file_path is not None:
        fig.savefig(file_path, dpi=300)
    plt.show()


# example usage for backtest
if __name__ == '__main__':

    df = pd.read_excel(BASE_DIR / 'filtered_trades' / 'tree_prediction_top_5_percent.xlsx')

    results = full_backtest_report(df, 0.01, mc_sims=10000)

    # Access individual pieces:
    daily        = results['daily']           # daily PnL dataframe
    risk         = results['risk_metrics']    # Sharpe, Sortino, Calmar, etc.
    dd           = results['drawdown_stats']  # max drawdown details
    mc_stats     = results['monte_carlo']['stats']  # MC summary

    
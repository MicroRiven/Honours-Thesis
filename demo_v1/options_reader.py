import pandas as pd
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def read_options_data(
    dt: pd.Timestamp,
    *,
    tickers: list[str] | None = None, # if None, read all tickers
    max_dte: int | None = None,  # if not None, filter out options with dte > max_dte after reading
    chunksize: int = 1_000_000,  # tune as needed, our daily option data should be right under 1_000_000 rows
    cols = None
) -> pd.DataFrame:
    
    # get path
    y = dt.year
    ymd = dt.strftime("%Y%m%d")
    option_today_path = BASE_DIR / "data" / str(y) / f"ORATS_SMV_Strikes_{ymd}.zip"
    if not option_today_path.exists():
        logger.warning(f"Option data file not found for date {dt.date()}, skipping.")
        return pd.DataFrame()

    # Ensure required columns are included
    required_cols = ["ticker", "stkPx", "trade_date", "expirDate", "yte", "strike","cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol"]
    if cols is None:
        cols = required_cols.copy()
    else:
        for col in required_cols:
            if col not in cols:
                cols.append(col)
    
    tickers_set = set(tickers) if tickers is not None else None
    
    # dtypes reduce memory a lot
    # NOTE: if column in dtypes but not cols, nothing breaks,
    #   but if column in cols but not dtypes, it will be read as object, which is memory inefficient
    dtypes = {
        "ticker": "string",
        "stkPx": "float32",
        "yte": "float32",
        "strike": "float32",
        "cVolu": "Int32",
        "pVolu": "Int32",
        "cValue": "float32",
        "pValue": "float32",
        "smoothSmvVol": "float32",
    }

    filtered_chunks = []

    for chunk in pd.read_csv(
        option_today_path, # e.g. "ORATS_SMV_Strikes_20241023.zip"
        compression="zip",
        usecols=cols,
        dtype=dtypes,
        chunksize=chunksize,
        low_memory=False,
    ):        
        if tickers_set is not None:
            # keep only requested tickers ASAP
            chunk = chunk[chunk["ticker"].isin(tickers_set)]
            if chunk.empty:
                continue
        filtered_chunks.append(chunk)

    if not filtered_chunks:
        logger.warning(f"No data found for date {dt.date()} after filtering by tickers.")
        return pd.DataFrame()

    opt = pd.concat(filtered_chunks, ignore_index=True)

    opt = opt.rename(columns={"smoothSmvVol": "vol",})
    opt["trade_date"] = pd.to_datetime(opt["trade_date"])
    opt["expirDate"]  = pd.to_datetime(opt["expirDate"])
    opt["dte"] = (opt["expirDate"] - opt["trade_date"]).dt.days.astype("float32")

    if max_dte is not None:
        opt = opt[opt["dte"] <= max_dte]

    return opt


if __name__ == "__main__":
    
    date_str = "2024-10-23"
    dt = pd.to_datetime(date_str)
    cols = ["ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "yte", "strike",
         "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol", "iRate", "divRate",]
    df = read_options_data(dt, max_dte=100, cols = cols)
    print(df.head())
    print(df.dtypes)
    print(df.shape)

    # check if a ticker exists
    ticker = "VXX"
    if ticker in df["ticker"].values:
        print(f"Ticker {ticker} found in data.")
        df_filtered = df[df["ticker"] == ticker]
        print(df_filtered.head())
        print(df_filtered.dtypes)
        df_filtered.to_excel(f"{ticker}_options_{date_str}.xlsx", index=False)
    else:
        print(f"Ticker {ticker} not found in data.")
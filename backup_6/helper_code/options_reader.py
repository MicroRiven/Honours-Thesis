import pandas as pd
from pathlib import Path


def read_options_data(
    option_today_path: str,
    tickers: list[str] | None, # if None, read all tickers
    *,
    chunksize: int = 1_000_000,  # tune as needed, our daily option data should be right under 1_000_000 rows
    cols = [
        "ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "strike",
         "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol",
    ]
) -> pd.DataFrame:
    
    # Ensure required columns are included
    required_cols = ["ticker", "trade_date", "expirDate", "smoothSmvVol"]
    for col in required_cols:
        if col not in cols:
            cols.append(col)
    
    tickers_set = set(tickers) if tickers is not None else None
    
    # dtypes reduce memory a lot
    # NOTE: if column in dtypes but not cols, nothing breaks,
    #   but if column in cols but not dtypes, it will be read as object, which is memory inefficient
    dtypes = {
        "ticker": "string",
        "cOpra": "string",
        "pOpra": "string",
        "stkPx": "float32",
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
        return pd.DataFrame()

    opt = pd.concat(filtered_chunks, ignore_index=True)

    opt = opt.rename(columns={"smoothSmvVol": "vol",})
    opt["trade_date"] = pd.to_datetime(opt["trade_date"])
    opt["expirDate"]  = pd.to_datetime(opt["expirDate"])
    opt["dte"] = (opt["expirDate"] - opt["trade_date"]).dt.days.astype("float32")


    return opt


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    date_str = "20230323"
    option_today_path = BASE_DIR / "data" / "2023" / f"ORATS_SMV_Strikes_{date_str}.zip"
    

    cols = ["ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "yte", "strike",
         "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol", "iRate", "divRate",]
    df = read_options_data(option_today_path, None, cols = cols)
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
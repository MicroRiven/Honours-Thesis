import pandas as pd




def read_options_data(
    option_today_path: str,
    tickers: list[str] | None, # if None, read all tickers
    chunksize: int = 1_000_000,  # tune as needed, our daily option data should be right under 1_000_000 rows
) -> pd.DataFrame:
    
    tickers_set = set(tickers) if tickers is not None else None

    usecols = [
        "ticker", "cOpra", "pOpra", "stkPx", "trade_date", "expirDate", "strike",
         "cVolu", "pVolu", "cValue", "pValue", "smoothSmvVol",
    ]
    
    # dtypes reduce memory a lot
    dtypes = {
        "ticker": "string",
        "strike": "float32",
        "stkPx": "float32",
        "cValue": "float32",
        "pValue": "float32",
        "smoothSmvVol": "float32",
    }

    filtered_chunks = []

    for chunk in pd.read_csv(
        option_today_path, # e.g. "ORATS_SMV_Strikes_20241023.zip"
        compression="zip",
        usecols=usecols,
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

    keep = ["ticker","trade_date","expirDate","dte","strike","stkPx","cValue","pValue","vol"]
    opt = opt[keep].copy()

    return opt


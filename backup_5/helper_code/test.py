from options_reader import read_options_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
date_str = "20230109"
option_today_path = BASE_DIR / "data" / "2023" / f"ORATS_SMV_Strikes_{date_str}.zip"
df = read_options_data(option_today_path, None)
print(df.head())

# check if a ticker exists
ticker = "F"
if ticker in df["ticker"].values:
    print(f"Ticker {ticker} found in data.")
    df_filtered = df[df["ticker"] == ticker]
    print(df_filtered)
    df_filtered.to_excel(f"{ticker}_options_{date_str}.xlsx", index=False)
else:
    print(f"Ticker {ticker} not found in data.")
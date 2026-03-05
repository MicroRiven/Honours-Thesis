import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_underlying import fetch_underlying

# Fetch all available data for ARM
ticker = "ARM"
start_date = "2019-01-01"  # Adjust if you want earlier data
end_date = "2025-12-31"

df = fetch_underlying(ticker, start_date, end_date)

if not df.empty:
    print(f"Fetched {len(df)} rows for {ticker}")
    print(f"Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
else:
    print(f"No data found for {ticker}")

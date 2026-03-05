import os
from pathlib import Path
import pandas as pd


def parquet_to_csv(parquet_path: str, csv_path: str | None = None) -> str:
    """
    Convert a Parquet file to CSV.

    Parameters
    ----------
    parquet_path : str
        Path to the input .parquet file.
    csv_path : str | None
        Path to the output .csv file. If None, uses the same name with .csv extension.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    parquet_path = Path(parquet_path)

    if csv_path is None:
        csv_path = parquet_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)

    # Read parquet
    df = pd.read_parquet(parquet_path)

    # Write csv (no index column)
    df.to_csv(csv_path, index=False)

    print(f"Converted {parquet_path} -> {csv_path}")
    return str(csv_path)


if __name__ == "__main__":

    path = "C:\\Users\\Wang0\\Documents\\UNSW\\Honours Thesis\\code\\data"
    pq_path = os.path.join(path, "all_underlyings.parquet")
    csv_path = pq_path.replace(".parquet", ".csv")
    parquet_to_csv(pq_path, csv_path)
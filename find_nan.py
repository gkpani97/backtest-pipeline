import pandas as pd

df = pd.read_parquet("data/AAPL.parquet")

bad = df[df[["Open", "High", "Low", "Close"]].isna().any(axis=1)]
print(bad)
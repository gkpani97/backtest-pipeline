import os
import yfinance as yf

TICKER = "AAPL"
DATA_FOLDER = "data"

df = yf.Ticker(TICKER).history(period="5y", interval="1d")

print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\n",df.head())

if not os.path.exists(os.path.join(os.getcwd(), DATA_FOLDER)):
    os.mkdir(DATA_FOLDER)

df.to_parquet(f"data/{TICKER}.parquet")
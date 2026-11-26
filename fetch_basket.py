import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",   # tech / growth
    "JPM", "BAC",                                                      # banks
    "XOM", "CVX",                                                      # energy
    "JNJ", "PFE",                                                      # healthcare
    "KO", "PEP", "WMT",                                                # staples
    "DIS", "INTC", "CAT",                                              # mixed, includes some laggards
]

Path("data/basket").mkdir(parents=True, exist_ok=True)

for t in TICKERS:
    try:
        df = yf.Ticker(t).history(period="5y", interval="1d")
        df = df.dropna(subset=["Close"])          # drops incomplete rows like the AAPL one
        out = df[["Close", "Volume"]].copy()
        out.index = out.index.tz_localize(None).normalize()
        out.index.name = "Date"
        out.to_parquet(f"data/basket/{t}.parquet")
        flag = "" if len(out) >= 1200 else "   <-- CHECK: fewer rows than expected"
        print(f"{t:6s} rows={len(out):5d}  {out.index.min().date()} -> {out.index.max().date()}{flag}")
    except Exception as e:
        print(f"{t:6s} FAILED: {e}")
    time.sleep(1)   # be polite to Yahoo's rate limits
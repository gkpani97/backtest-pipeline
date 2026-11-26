from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST = 0.001
WARMUP = 200                      # common start, so every setting is compared on the same days
SPLIT = "2024-07-01"
STRATS = {"20/50": (20, 50), "30/50": (30, 50), "10/200": (10, 200)}


def stats(r: pd.Series):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {
        "total": eq.iloc[-1] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


def strat_returns(close: pd.Series, fast: int, slow: int):
    ret = close.pct_change()
    signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
    position = signal.shift(1).fillna(0)            # trade the day after the signal
    trades = position.diff().abs().fillna(0)
    return position * ret - trades * COST


rows = []
for path in sorted(Path("data/basket").glob("*.parquet")):
    ticker = path.stem
    close = pd.read_parquet(path)["Close"]
    start = close.index[WARMUP]

    returns = {"Buy&hold": close.pct_change()}
    for name, (f, s) in STRATS.items():
        returns[name] = strat_returns(close, f, s)

    windows = {
        "Full": (start, None),
        "A: start to Jun 2024": (start, "2024-06-30"),
        "B: Jul 2024 on": (SPLIT, None),
    }
    for wname, (a, b) in windows.items():
        for sname, r in returns.items():
            rows.append({"ticker": ticker, "window": wname, "strategy": sname, **stats(r.loc[a:b])})

res = pd.DataFrame(rows)


def summarize(window):
    w = res[res.window == window]
    bh = w[w.strategy == "Buy&hold"].set_index("ticker")
    out = {"Buy&hold": {
        "median total": bh.total.median(),
        "median Sharpe": bh.sharpe.median(),
        "median max DD": bh.maxdd.median(),
    }}
    for s in STRATS:
        x = w[w.strategy == s].set_index("ticker")
        out[s] = {
            "median total": x.total.median(),
            "median Sharpe": x.sharpe.median(),
            "median max DD": x.maxdd.median(),
            "beat BH on return": f"{int((x.total > bh.total).sum())}/{len(x)}",
            "beat BH on Sharpe": f"{int((x.sharpe > bh.sharpe).sum())}/{len(x)}",
            "smaller drawdown": f"{int((x.maxdd > bh.maxdd).sum())}/{len(x)}",
        }
    return pd.DataFrame(out).T


for window in res.window.unique():
    print(f"\n=== {window} ===")
    print(summarize(window).round(3).to_string())

# Per-ticker detail for the textbook 20/50 setting over the full window
full = res[res.window == "Full"]
detail = full[full.strategy == "20/50"].set_index("ticker")[["total", "sharpe", "maxdd"]].join(
    full[full.strategy == "Buy&hold"].set_index("ticker")[["total", "sharpe", "maxdd"]],
    lsuffix="_strat", rsuffix="_bh",
)
print("\n=== 20/50 per ticker, full window (0.10 means 10%) ===")
print(detail.round(2).to_string())
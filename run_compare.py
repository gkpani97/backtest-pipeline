from pathlib import Path

import pandas as pd

from backtest import WARMUP, check_no_lookahead, compute_metrics, run_backtest
from strategies import BuyAndHold, RsiMeanReversion, SmaCrossover, Strategy


class Cheater(Strategy):
    """Deliberately peeks at tomorrow's price. Exists only to prove the test catches cheats."""
    name = "Cheater (peeks)"

    def signal(self, close):
        return (close.shift(-1) > close).astype(int)


STRATEGIES = [
    BuyAndHold(),
    SmaCrossover(20, 50),
    SmaCrossover(10, 200),
    RsiMeanReversion(period=14, entry_below=30, exit_above=55),
]

closes = {p.stem: pd.read_parquet(p)["Close"] for p in sorted(Path("data/basket").glob("*.parquet"))}
aapl = closes["AAPL"]

# --- 1. Causality tests ---
print("=== Look-ahead check (AAPL) ===")
for s in STRATEGIES + [Cheater()]:
    ok, when = check_no_lookahead(s, aapl)
    print(f"{s.name:28s} {'OK' if ok else 'FAILED, first mismatch on ' + str(when.date())}")

# --- 2. Run everything ---
rows = []
for ticker, close in closes.items():
    for s in STRATEGIES:
        rows.append({"ticker": ticker, "strategy": s.name, **compute_metrics(run_backtest(close, s))})
df = pd.DataFrame(rows)

# Traceability: which dates, and what did AAPL buy-and-hold do?
res = run_backtest(aapl, BuyAndHold())
print(f"\nWindow: {res.returns.index[0].date()} -> {res.returns.index[-1].date()} "
      f"(warm-up {WARMUP} rows, {len(closes)} tickers)")
print("AAPL buy-and-hold total:", round(compute_metrics(res)["total"], 3))

# --- 3. Summary across tickers ---
total = df.pivot(index="ticker", columns="strategy", values="total")
sharpe = df.pivot(index="ticker", columns="strategy", values="sharpe")
maxdd = df.pivot(index="ticker", columns="strategy", values="maxdd")
bh = BuyAndHold().name

summary = pd.DataFrame({
    "median total": total.median(),
    "median Sharpe": sharpe.median(),
    "median max DD": maxdd.median(),
    "median exposure": df.groupby("strategy").exposure.median(),
    "median trade legs": df.groupby("strategy").legs.median(),
    "beat BH on return": [(total[c] > total[bh]).sum() for c in total.columns],
    "beat BH on Sharpe": [(sharpe[c] > sharpe[bh]).sum() for c in total.columns],
    "shallower DD than BH": [(maxdd[c] > maxdd[bh]).sum() for c in total.columns],
})
print("\n=== Summary across tickers (0.28 means 28%) ===")
print(summary.round(3).to_string())
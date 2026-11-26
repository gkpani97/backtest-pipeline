from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies import Strategy

TRADING_DAYS = 252
COST = 0.001      # 0.1% per buy or sell
DELAY = 1         # act on the day after the signal
WARMUP = 200      # every strategy is evaluated from the same row onward


@dataclass
class BacktestResult:
    returns: pd.Series    # daily strategy returns after costs, evaluation window only
    position: pd.Series   # 1 = held, 0 = cash, same window
    trade_legs: int       # number of buys + sells in the window


def run_backtest(close, strategy: Strategy, cost=COST, delay=DELAY, warmup=WARMUP):
    ret = close.pct_change().fillna(0.0)
    signal = strategy.signal(close).fillna(0).astype(int)
    position = signal.shift(delay).fillna(0)
    legs = position.diff().abs().fillna(0)
    strat_ret = position * ret - legs * cost
    return BacktestResult(
        returns=strat_ret.iloc[warmup:],
        position=position.iloc[warmup:],
        trade_legs=int(legs.iloc[warmup:].sum()),
    )


def compute_metrics(res: BacktestResult) -> dict:
    r = res.returns
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    return {
        "total": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
        "maxdd": (equity / equity.cummax() - 1).min(),
        "exposure": res.position.mean(),
        "legs": res.trade_legs,
    }


def check_no_lookahead(strategy: Strategy, close: pd.Series, n_checks=25):
    """Causality test: the signal for day t must be identical whether or not
    the data after day t exists. Returns (passed, first_failing_date)."""
    full = strategy.signal(close).fillna(0).astype(int)
    for t in np.linspace(250, len(close) - 1, n_checks).astype(int):
        partial = strategy.signal(close.iloc[: t + 1]).fillna(0).astype(int)
        if partial.iloc[-1] != full.iloc[t]:
            return False, close.index[t]
    return True, None
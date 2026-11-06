"""Vectorized simulator for running a fixed trading schedule over many price paths.

The RL env (env.py) uses the same step math, so the two should agree. That's
checked in the tests.
"""
import numpy as np

from .params import MarketParams


def impact_step(S, n, p: MarketParams, xi):
    """One period of Almgren-Chriss dynamics.

    S  : price at start of the period (array or float)
    n  : shares sold this period
    xi : standard normal shock(s)

    Returns (execution price for this trade, next period's price).
    """
    v = n / p.tau  # trading rate
    # temporary impact only hits the price we get on this trade
    exec_price = S - (p.epsilon * np.sign(n) + p.eta * v)
    # permanent impact sticks around for the rest of the episode
    S_next = S + p.sigma * np.sqrt(p.tau) * xi - p.tau * p.gamma * v
    return exec_price, S_next


def run_schedule(trades, p: MarketParams, n_paths=10_000, seed=0):
    """Run a deterministic schedule of trades over n_paths random price paths.

    trades: array of length N with shares sold each period (should sum to X)
    Returns implementation shortfall per path, in dollars.
    """
    trades = np.asarray(trades, dtype=float)
    assert len(trades) == p.N, "need one trade size per period"
    rng = np.random.default_rng(seed)

    S = np.full(n_paths, p.S0)
    cash = np.zeros(n_paths)
    for k in range(p.N):
        xi = rng.standard_normal(n_paths)
        exec_price, S = impact_step(S, trades[k], p, xi)
        cash += trades[k] * exec_price

    # shortfall = what we'd have gotten at the arrival price minus what we got
    return p.X * p.S0 - cash

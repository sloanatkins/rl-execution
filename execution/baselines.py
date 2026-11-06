"""Baseline execution strategies: TWAP and the closed-form Almgren-Chriss schedule."""
import numpy as np

from .params import MarketParams


def twap_trades(p: MarketParams):
    # just split the order evenly across all periods
    return np.full(p.N, p.X / p.N)


def ac_kappa(p: MarketParams):
    """Urgency parameter kappa from the AC paper (discrete-time version)."""
    eta_tilde = p.eta - 0.5 * p.gamma * p.tau
    if p.lam == 0:
        return 0.0
    kappa_tilde_sq = p.lam * p.sigma ** 2 / eta_tilde
    # solve cosh(kappa * tau) = 1 + tau^2 * kappa_tilde^2 / 2
    return np.arccosh(0.5 * kappa_tilde_sq * p.tau ** 2 + 1) / p.tau


def ac_holdings(p: MarketParams):
    """Optimal remaining inventory x_0..x_N (x_0 = X, x_N = 0)."""
    t = np.arange(p.N + 1) * p.tau
    kappa = ac_kappa(p)
    if kappa == 0:
        # risk neutral case collapses to a straight line (= TWAP)
        return p.X * (1 - t / p.T)
    return p.X * np.sinh(kappa * (p.T - t)) / np.sinh(kappa * p.T)


def ac_trades(p: MarketParams):
    x = ac_holdings(p)
    return x[:-1] - x[1:]


def expected_cost(trades, p: MarketParams):
    """Closed-form expected shortfall for any deterministic sell schedule."""
    trades = np.asarray(trades, dtype=float)
    eta_tilde = p.eta - 0.5 * p.gamma * p.tau
    return (0.5 * p.gamma * p.X ** 2
            + p.epsilon * np.sum(np.abs(trades))
            + eta_tilde / p.tau * np.sum(trades ** 2))


def cost_variance(trades, p: MarketParams):
    """Closed-form variance of shortfall for a deterministic schedule."""
    trades = np.asarray(trades, dtype=float)
    # holdings after each trade, x_1..x_N (x_N should be 0)
    x = p.X - np.cumsum(trades)
    # price shock in period k hits the shares still held after trade k
    return p.sigma ** 2 * p.tau * np.sum(x ** 2)

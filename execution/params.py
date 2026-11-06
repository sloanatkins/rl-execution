from dataclasses import dataclass


@dataclass
class MarketParams:
    """Market + order parameters for the Almgren-Chriss style simulator.

    Defaults follow the numerical example in Almgren & Chriss (2000):
    sell 1M shares of a $50 stock over 5 days in 50 trading periods.
    """
    S0: float = 50.0          # starting (arrival) price
    X: float = 1_000_000      # shares to liquidate
    T: float = 5.0            # horizon in days
    N: int = 50               # number of trading periods
    sigma: float = 0.95       # price vol in $/share/sqrt(day)
    epsilon: float = 0.0625   # fixed temp cost per share (half spread + fees)
    eta: float = 2.5e-6       # temporary impact coefficient
    gamma: float = 2.5e-7     # permanent impact coefficient
    lam: float = 1e-6         # risk aversion

    @property
    def tau(self) -> float:
        # length of one trading period
        return self.T / self.N

"""Compare TWAP vs Almgren-Chriss on the default market and save a plot."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from execution import (MarketParams, run_schedule, twap_trades, ac_trades,
                       ac_holdings, expected_cost, cost_variance)


def main():
    p = MarketParams()
    strategies = {"TWAP": twap_trades(p), "Almgren-Chriss": ac_trades(p)}

    print(f"Selling {p.X:,.0f} shares at ${p.S0} over {p.T} days ({p.N} periods)\n")
    print(f"{'Strategy':<16}{'E[cost] theory':>16}{'E[cost] sim':>14}"
          f"{'Std theory':>13}{'Std sim':>11}{'E + lam*Var':>14}")
    for name, trades in strategies.items():
        sf = run_schedule(trades, p, n_paths=100_000, seed=42)
        E, V = expected_cost(trades, p), cost_variance(trades, p)
        print(f"{name:<16}{E:>16,.0f}{sf.mean():>14,.0f}"
              f"{np.sqrt(V):>13,.0f}{sf.std():>11,.0f}{E + p.lam * V:>14,.0f}")

    # plot remaining inventory over time
    os.makedirs("results", exist_ok=True)
    t = np.arange(p.N + 1) * p.tau
    plt.figure(figsize=(7, 4))
    plt.plot(t, p.X - np.concatenate([[0], np.cumsum(twap_trades(p))]), label="TWAP")
    plt.plot(t, ac_holdings(p), label=f"Almgren-Chriss (lambda={p.lam:g})")
    plt.xlabel("Days")
    plt.ylabel("Shares remaining")
    plt.title("Liquidation schedules")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_schedules.png", dpi=150)
    print("\nSaved results/baseline_schedules.png")


if __name__ == "__main__":
    main()

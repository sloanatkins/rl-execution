# RL Optimal Execution

Training a reinforcement learning agent to liquidate a large stock position while keeping market impact costs low, benchmarked against TWAP and the closed-form Almgren-Chriss schedule.

## Status

**Week 1 done:** simulator, baselines, tests.

- [x] Almgren-Chriss style simulator (temporary + permanent impact)
- [x] Gymnasium environment for RL agents
- [x] TWAP and Almgren-Chriss baselines with closed-form cost and variance
- [x] Tests checking the simulator against theory
- [ ] DQN agent (adapted from the BTE 422 notebook)
- [ ] PPO agent
- [ ] Volatility / liquidity regimes
- [ ] Short-horizon signal

## Layout

```
execution/
  params.py      # market + order parameters
  simulate.py    # vectorized price/impact dynamics, runs a schedule over many paths
  baselines.py   # TWAP, Almgren-Chriss trajectory, expected cost + variance formulas
  env.py         # Gymnasium env: action = fraction of remaining shares to sell
tests/
  test_simulator.py
scripts/
  run_baselines.py
```

## Model

Selling X shares over N periods of length tau. If we sell n_k shares in period k:

- **Execution price:** S_{k-1} - epsilon - eta * n_k / tau (temporary impact, only hits this trade)
- **Next price:** S_k = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k (permanent impact sticks)
- **Cost:** implementation shortfall = X * S0 - total cash received

The default parameters are from the example in Almgren & Chriss (2000).

## Run it

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_baselines.py
```

## First results

1M shares at $50 over 5 days, lambda = 1e-6, 100k simulated paths:

| Strategy | E[cost] theory | E[cost] sim | Std theory | Std sim | E + lambda * Var |
|---|---|---|---|---|---|
| TWAP | $685,000 | $680,068 | $1,208,032 | $1,214,068 | $2,144,342 |
| Almgren-Chriss | $961,940 | $959,714 | $828,686 | $833,014 | $1,648,660 |

Almgren-Chriss pays about $277k more in expected impact by selling faster, but it cuts the risk by about a third, which gives it a lower mean-variance objective. The simulated numbers line up with the formulas, so the simulator is behaving.

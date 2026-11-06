import numpy as np
import pytest

from execution import (MarketParams, ExecutionEnv, run_schedule, twap_trades,
                       ac_trades, ac_holdings, expected_cost, cost_variance)


@pytest.fixture
def p():
    return MarketParams()


def test_schedules_sell_everything(p):
    assert np.isclose(twap_trades(p).sum(), p.X)
    assert np.isclose(ac_trades(p).sum(), p.X)
    x = ac_holdings(p)
    assert np.isclose(x[0], p.X) and np.isclose(x[-1], 0)


def test_ac_sells_front_loaded(p):
    # risk averse trader should sell faster early on
    trades = ac_trades(p)
    assert np.all(np.diff(trades) <= 1e-9)


def test_zero_risk_aversion_is_twap(p):
    p.lam = 0.0
    assert np.allclose(ac_trades(p), twap_trades(p))


def test_no_vol_matches_formula_exactly(p):
    # with sigma = 0 the shortfall is deterministic, so it should equal the formula
    p.sigma = 0.0
    for trades in (twap_trades(p), ac_trades(p)):
        sf = run_schedule(trades, p, n_paths=5)
        assert np.allclose(sf, expected_cost(trades, p))


@pytest.mark.parametrize("schedule", [twap_trades, ac_trades])
def test_monte_carlo_matches_mean_and_variance(p, schedule):
    trades = schedule(p)
    sf = run_schedule(trades, p, n_paths=200_000, seed=1)
    se = sf.std() / np.sqrt(len(sf))
    # mean within 4 standard errors, variance within 2%
    assert abs(sf.mean() - expected_cost(trades, p)) < 4 * se
    assert np.isclose(sf.var(), cost_variance(trades, p), rtol=0.02)


def test_ac_has_best_mean_variance_objective(p):
    # AC should minimize E + lambda * Var among deterministic schedules
    def obj(t):
        return expected_cost(t, p) + p.lam * cost_variance(t, p)
    best = obj(ac_trades(p))
    assert best <= obj(twap_trades(p))
    rng = np.random.default_rng(0)
    for _ in range(50):
        # perturb the AC schedule a bit and make sure it gets worse
        t = ac_trades(p) * (1 + 0.1 * rng.standard_normal(p.N))
        t = np.clip(t, 0, None)
        t *= p.X / t.sum()
        assert best <= obj(t) + 1e-6


def test_env_liquidates_and_tracks_cash(p):
    env = ExecutionEnv(p)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, r, done, _, info = env.step([0.1])
    assert np.isclose(env.x, 0)
    assert obs[1] == 0
    assert np.isclose(info["shortfall"], p.X * p.S0 - env.cash)


def test_env_agrees_with_vectorized_sim(p):
    # replaying TWAP through the env with sigma = 0 should give the formula cost
    p.sigma = 0.0
    env = ExecutionEnv(p)
    env.reset(seed=0)
    done = False
    while not done:
        # fraction of remaining that matches an even split
        frac = 1.0 / (p.N - env.k)
        _, _, done, _, info = env.step([frac])
    assert np.isclose(info["shortfall"], expected_cost(twap_trades(p), p))


def test_rewards_sum_to_shortfall_without_risk_penalty(p):
    env = ExecutionEnv(p, reward_scale=1.0, risk_penalty=False)
    env.reset(seed=3)
    total, done = 0.0, False
    while not done:
        _, r, done, _, info = env.step([0.05])
        total += r
    assert np.isclose(-total, info["shortfall"])

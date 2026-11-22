import numpy as np
import pytest

from ppo import PPO, PPOConfig, gae
from ppo.agent import RunningMeanStd


def gae_direct(r, v, done, last_v, gamma, lam):
    """Âₜ = Σₗ (γλ)ˡ δₜ₊ₗ, summed until the episode ends: an O(T²) reference."""
    T = len(r)
    nxt = np.append(v[1:], last_v)
    delta = r + gamma * nxt * (1 - done) - v
    adv = np.zeros(T)
    for t in range(T):
        coef = 1.0
        for k in range(t, T):
            adv[t] += coef * delta[k]
            if done[k]:
                break
            coef *= gamma * lam
    return adv


def test_gae_matches_direct_sum():
    rng = np.random.default_rng(0)
    T = 40
    r, v = rng.standard_normal(T), rng.standard_normal(T)
    done = (rng.random(T) < 0.15).astype(float)
    adv, ret = gae(r[:, None], v[:, None], done[:, None], np.array([0.7]), 0.97, 0.9)
    assert np.allclose(adv[:, 0], gae_direct(r, v, done, 0.7, 0.97, 0.9))
    assert np.allclose(ret, adv + v[:, None])


def test_gae_limits():
    """λ = 0 gives one-step TD errors; λ = 1 gives Monte Carlo return minus the baseline."""
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[0.5], [0.1], [0.2]])
    done = np.array([[0.0], [0.0], [1.0]])
    td, _ = gae(r, v, done, np.array([9.0]), 0.9, 0.0)
    assert np.allclose(td[:, 0], [1 + 0.9 * 0.1 - 0.5, 2 + 0.9 * 0.2 - 0.1, 3 - 0.2])
    mc, _ = gae(r, v, done, np.array([9.0]), 0.9, 1.0)
    returns = [1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3]
    assert np.allclose(mc[:, 0], np.array(returns) - v[:, 0])


def test_running_mean_std_matches_batch_statistics():
    rng = np.random.default_rng(1)
    batches = [rng.normal(3, 2, size=(n, 2)) for n in (5, 50, 7, 300)]
    rms = RunningMeanStd(2)
    for b in batches:
        rms.update(b)
    allx = np.concatenate(batches)
    assert np.allclose(rms.mean, allx.mean(axis=0), atol=1e-4)
    assert np.allclose(rms.var, allx.var(axis=0), rtol=1e-3)


def test_cartpole_learns_to_balance():
    """Random actions last about 20 steps; after 100k steps the greedy policy should last
    hundreds (5 of 6 seeds reach at least 430 of the 500-step maximum)."""
    agent = PPO(PPOConfig(env="cartpole", seed=0))
    assert agent.evaluate(episodes=10) < 100
    agent.train(100_000, log=None)
    assert agent.evaluate(episodes=10) >= 300


@pytest.mark.parametrize("seed", [1])
def test_pendulum_improves(seed):
    cfg = PPOConfig(env="pendulum", n_envs=8, n_steps=256, epochs=10, minibatch=256, gamma=0.9, lr=1e-3,
                    ent_coef=0.0, normalize_obs=True, scale_rewards=True, seed=seed)
    agent = PPO(cfg)
    before = agent.evaluate(episodes=10)
    agent.train(80_000, log=None)
    after = agent.evaluate(episodes=10)
    assert before < -900 and after > -500

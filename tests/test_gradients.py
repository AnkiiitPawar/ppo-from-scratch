import numpy as np
import pytest

from ppo import PPO, PPOConfig, clipped_surrogate
from ppo.distributions import Categorical, DiagGaussian
from ppo.nn import MLP


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def close(a, b, tol=1e-6):
    return np.linalg.norm(a - b) <= tol * max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def test_mlp_gradients():
    rng = np.random.default_rng(0)
    net = MLP([3, 5, 4, 2], rng, out_gain=1.0)
    x = rng.standard_normal((6, 3))
    r = rng.standard_normal((6, 2))
    net.zero_grad()
    net.forward(x)
    dx = net.backward(r)
    loss = lambda: float((net.forward(x) * r).sum())  # noqa: E731
    for p, g in zip(net.params(), net.grads()):
        assert close(g, num_grad(loss, p))
    assert close(dx, num_grad(loss, x))


def test_categorical_gradients():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((5, 4))
    a = rng.integers(0, 4, 5)
    w = rng.standard_normal(5)
    d = Categorical(z)
    assert close(w[:, None] * d.grad_log_prob(a), num_grad(lambda: float((Categorical(z).log_prob(a) * w).sum()), z))
    assert close(w[:, None] * d.grad_entropy(), num_grad(lambda: float((Categorical(z).entropy() * w).sum()), z))
    assert np.allclose(d.entropy(), -(d.p * np.log(d.p)).sum(axis=1))


def test_gaussian_log_prob_formula_and_gradients():
    rng = np.random.default_rng(2)
    mu = rng.standard_normal((5, 2))
    log_std = rng.standard_normal(2) * 0.3
    a = rng.standard_normal((5, 2))
    d = DiagGaussian(mu, log_std)
    std = np.exp(log_std)
    direct = np.log(np.exp(-0.5 * ((a - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))).sum(axis=1)
    assert np.allclose(d.log_prob(a), direct)
    g_mu, g_ls = d.grad_log_prob(a)
    assert close(g_mu, num_grad(lambda: float(DiagGaussian(mu, log_std).log_prob(a).sum()), mu))
    assert close(g_ls.sum(axis=0), num_grad(lambda: float(DiagGaussian(mu, log_std).log_prob(a).sum()), log_std))
    assert close(np.ones(2) * 5, num_grad(lambda: float(DiagGaussian(mu, log_std).entropy().sum()), log_std))


def test_clipped_surrogate_gradient_away_from_the_kinks():
    rng = np.random.default_rng(3)
    logp_old = rng.standard_normal(200) * 0.1
    logp = logp_old + rng.uniform(-0.5, 0.5, 200)
    ratio = np.exp(logp - logp_old)
    keep = (np.abs(ratio - 0.8) > 0.02) & (np.abs(ratio - 1.2) > 0.02)  # finite differences straddle no kink
    logp, logp_old = logp[keep], logp_old[keep]
    adv = rng.standard_normal(len(logp))
    _, d, stats = clipped_surrogate(logp, logp_old, adv, 0.2)
    assert close(d, num_grad(lambda: clipped_surrogate(logp, logp_old, adv, 0.2)[0], logp))
    assert 0 < stats["clip_frac"] < 1


def test_clipping_removes_the_incentive_to_move_further():
    """Positive advantage and a ratio already above 1 + ε: no gradient."""
    _, d, _ = clipped_surrogate(np.array([np.log(1.5)]), np.array([0.0]), np.array([1.0]), 0.2)
    assert d[0] == 0.0
    _, d, _ = clipped_surrogate(np.array([np.log(1.5)]), np.array([0.0]), np.array([-1.0]), 0.2)
    assert d[0] != 0.0  # negative advantage: still pushed back down


@pytest.mark.parametrize("env", ["cartpole", "pendulum"])
def test_full_policy_loss_gradient(env):
    """Finite differences of the whole PPO policy loss (surrogate and entropy bonus)."""
    agent = PPO(PPOConfig(env=env, hidden=(8, 8), ent_coef=0.05, seed=4))
    rng = np.random.default_rng(5)
    obs = rng.standard_normal((32, agent.env.obs_dim))
    d = agent.dist(agent.pi.forward(obs))
    act = d.sample(rng)
    logp_old = d.log_prob(act) + rng.normal(0, 0.05, 32)  # ratios near 1, a few clipped
    adv = rng.standard_normal(32)
    agent.policy_loss_and_grad(obs, act, logp_old, adv)
    analytic = [g.copy() for g in agent.pi_grads]
    loss = lambda: agent.policy_loss_and_grad(obs, act, logp_old, adv)[0]  # noqa: E731
    for p, g in zip(agent.pi_params, analytic):
        assert close(g, num_grad(loss, p), tol=1e-5)

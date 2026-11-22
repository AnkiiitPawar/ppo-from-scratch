import numpy as np
import pytest

from ppo.envs import CartPole, Pendulum


def test_cartpole_closed_form_matches_lagrangian_mass_matrix():
    """Solve M(q) q̈ = f(q, q̇, F) directly and compare with the closed-form accelerations."""
    rng = np.random.default_rng(0)
    states = rng.uniform([-2, -3, -1.2, -4], [2, 3, 1.2, 4], size=(200, 4))
    forces = rng.uniform(-10, 10, size=200)
    xdd, thdd = CartPole.accelerations(states, forces)
    M, m, l, g = CartPole.m_cart + CartPole.m_pole, CartPole.m_pole, CartPole.half_len, CartPole.g
    for (x, xd, th, thd), F, a, b in zip(states, forces, xdd, thdd):
        mass = np.array([[M, m * l * np.cos(th)], [m * l * np.cos(th), 4.0 / 3.0 * m * l**2]])
        rhs = np.array([F + m * l * thd**2 * np.sin(th), m * g * l * np.sin(th)])
        assert np.allclose(np.linalg.solve(mass, rhs), [a, b], rtol=1e-12, atol=1e-12)


def test_cartpole_mirror_symmetry():
    """Reflecting the state and the force reflects the accelerations."""
    s = np.array([[0.3, -0.5, 0.1, 0.7]])
    a1 = CartPole.accelerations(s, np.array([4.0]))
    a2 = CartPole.accelerations(-s, np.array([-4.0]))
    assert np.allclose(a1, [-a2[0], -a2[1]])


def test_cartpole_equilibrium_and_instability():
    xdd, thdd = CartPole.accelerations(np.zeros((1, 4)), np.zeros(1))
    assert xdd[0] == 0 and thdd[0] == 0
    _, thdd = CartPole.accelerations(np.array([[0, 0, 0.01, 0]]), np.zeros(1))
    assert thdd[0] > 0  # upright is unstable: a small tilt grows


def test_cartpole_terminates_and_truncates():
    env = CartPole(1, seed=0)
    env.state[:] = [0, 0, 0.25, 0]  # beyond 12 degrees after one step
    _, _, term, trunc, final = env.step(np.array([1]))
    assert term[0] and not trunc[0] and abs(final[0, 2]) > CartPole.theta_limit
    assert np.all(np.abs(env.state) <= 0.05)  # reset automatically

    env = CartPole(1, seed=1)
    trunc_seen = False
    for t in range(500):  # balance by pushing toward the side the pole leans
        _, _, term, trunc, _ = env.step(np.array([int(env.state[0, 2] + 0.5 * env.state[0, 3] > 0)]))
        assert not term[0]
        trunc_seen = trunc[0]
    assert trunc_seen


def pendulum_energy(th, thd):
    m, l, g = Pendulum.m, Pendulum.l, Pendulum.g
    return 0.5 * (m * l**2 / 3) * thd**2 + m * g * (l / 2) * np.cos(th)


def test_pendulum_equation_is_the_torque_balance():
    th, u = 0.7, 1.3
    m, l, g = Pendulum.m, Pendulum.l, Pendulum.g
    inertia = m * l**2 / 3
    assert Pendulum.angular_acceleration(th, u) == pytest.approx((m * g * l / 2 * np.sin(th) + u) / inertia)


def test_pendulum_free_swing_has_no_energy_drift():
    """Semi-implicit Euler is symplectic: with no torque the energy error oscillates (here by
    up to ~9 % at dt = 0.05 s) but does not grow; explicit Euler's error runs away."""
    env = Pendulum(1, seed=0)
    env.th[:] = np.pi - 1.0  # 1 rad from hanging straight down, at rest
    env.thd[:] = 0.0
    e0 = pendulum_energy(env.th[0], env.thd[0])
    err = []
    for _ in range(199):  # about six swings, all inside one episode
        env.step(np.zeros(1))
        err.append(abs(pendulum_energy(env.th[0], env.thd[0]) - e0) / abs(e0))
    first, last = max(err[:100]), max(err[100:])
    assert last < 0.12 and last < first * 1.1

    th, thd = np.pi - 1.0, 0.0  # explicit Euler on the same problem, for contrast
    explicit = []
    for _ in range(199):
        th, thd = th + thd * Pendulum.dt, thd + Pendulum.angular_acceleration(th, 0.0) * Pendulum.dt
        explicit.append(abs(pendulum_energy(th, thd) - e0) / abs(e0))
    assert max(explicit) > 1.0  # explicit Euler gains energy every swing: over 100 % here


def test_pendulum_reward_is_zero_only_when_balanced():
    env = Pendulum(2, seed=0)
    env.th[:] = [0.0, np.pi]
    env.thd[:] = 0.0
    _, r, _, _, _ = env.step(np.zeros(2))
    assert r[0] == 0.0 and r[1] == pytest.approx(-np.pi**2)


def test_pendulum_truncates_at_200_steps():
    env = Pendulum(3, seed=0)
    for t in range(200):
        _, _, term, trunc, _ = env.step(np.zeros(3))
        assert not term.any()
    assert trunc.all()

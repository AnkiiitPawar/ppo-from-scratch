"""Two control problems written from their equations of motion, vectorised over N copies.

Both follow the classic benchmark definitions (Barto, Sutton & Anderson 1983 for the
cart-pole; the torque-limited pendulum swing-up), with the dynamics derived below rather
than imported from a library.

``step(actions)`` returns (obs, reward, terminated, truncated, final_obs). Environments
that finish are reset automatically; ``final_obs`` holds the observation they ended on,
which the learner needs to bootstrap through a time limit.
"""

from __future__ import annotations

import numpy as np


class CartPole:
    """A cart on a frictionless track with a uniform rod hinged on top.

    Coordinates: cart position x, rod angle θ from upright (positive toward +x). The rod
    has mass m and half-length l (moment of inertia m l²/3 about its centre); the cart has
    mass M_c, and M = M_c + m. Lagrange's equations give

        M ẍ + m l cosθ θ̈ = F + m l θ̇² sinθ
        m l cosθ ẍ + (4/3) m l² θ̈ = m g l sinθ

    Solving the 2×2 system:
        a = (F + m l θ̇² sinθ) / M
        θ̈ = (g sinθ - a cosθ) / (l (4/3 - m cos²θ / M)),   ẍ = a - m l θ̈ cosθ / M

    Actions: 0 pushes left, 1 pushes right, with |F| = 10 N. Reward +1 per step. The
    episode terminates when |θ| > 12° or |x| > 2.4 m, and is truncated at 500 steps.
    Integration: semi-implicit Euler with dt = 0.02 s.
    """

    obs_dim, n_actions = 4, 2
    g, m_cart, m_pole, half_len, force, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    theta_limit, x_limit, max_steps = 12 * np.pi / 180, 2.4, 500

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((n, 4))
        self.t = np.zeros(n, dtype=int)
        self.reset_where(np.ones(n, dtype=bool))

    def reset_where(self, which: np.ndarray) -> None:
        k = int(which.sum())
        self.state[which] = self.rng.uniform(-0.05, 0.05, size=(k, 4))
        self.t[which] = 0

    def reset(self) -> np.ndarray:
        self.reset_where(np.ones(self.n, dtype=bool))
        return self.state.copy()

    @classmethod
    def accelerations(cls, state: np.ndarray, force: np.ndarray):
        """(ẍ, θ̈) for each row of state = (x, ẋ, θ, θ̇) under horizontal force F."""
        _, _, th, thd = state.T
        M, m, l = cls.m_cart + cls.m_pole, cls.m_pole, cls.half_len
        s, c = np.sin(th), np.cos(th)
        a = (force + m * l * thd**2 * s) / M
        thdd = (cls.g * s - a * c) / (l * (4.0 / 3.0 - m * c**2 / M))
        xdd = a - m * l * thdd * c / M
        return xdd, thdd

    def step(self, actions: np.ndarray):
        force = np.where(np.asarray(actions) == 1, self.force, -self.force)
        xdd, thdd = self.accelerations(self.state, force)
        st = self.state
        st[:, 1] += self.dt * xdd
        st[:, 3] += self.dt * thdd
        st[:, 0] += self.dt * st[:, 1]
        st[:, 2] += self.dt * st[:, 3]
        self.t += 1
        terminated = (np.abs(st[:, 0]) > self.x_limit) | (np.abs(st[:, 2]) > self.theta_limit)
        truncated = (self.t >= self.max_steps) & ~terminated
        reward = np.ones(self.n)
        final_obs = st.copy()
        done = terminated | truncated
        if done.any():
            self.reset_where(done)
        return st.copy(), reward, terminated, truncated, final_obs


class Pendulum:
    """Torque-limited swing-up of a rigid rod (mass m, length l) about a fixed pivot.

    With θ = 0 upright, the rod's moment of inertia about the pivot is m l²/3 and gravity
    acts at its centre (l/2), so

        (m l²/3) θ̈ = m g (l/2) sinθ + u   ⇒   θ̈ = (3g / 2l) sinθ + (3 / m l²) u

    Torque |u| <= 2, speed clipped to |θ̇| <= 8, dt = 0.05 s, 200 steps per episode (a
    time limit, never a failure). Observation (cosθ, sinθ, θ̇). Reward
    -(θ² + 0.1 θ̇² + 0.001 u²) with θ wrapped to [-π, π): 0 is perfect balance upright.
    """

    obs_dim, act_dim = 3, 1
    g, m, l, dt, max_speed, max_torque, max_steps = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0, 200

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.th = np.zeros(n)
        self.thd = np.zeros(n)
        self.t = np.zeros(n, dtype=int)
        self.reset_where(np.ones(n, dtype=bool))

    def reset_where(self, which: np.ndarray) -> None:
        k = int(which.sum())
        self.th[which] = self.rng.uniform(-np.pi, np.pi, size=k)
        self.thd[which] = self.rng.uniform(-1.0, 1.0, size=k)
        self.t[which] = 0

    def reset(self) -> np.ndarray:
        self.reset_where(np.ones(self.n, dtype=bool))
        return self.obs()

    def obs(self) -> np.ndarray:
        return np.stack([np.cos(self.th), np.sin(self.th), self.thd], axis=1)

    @classmethod
    def angular_acceleration(cls, th, u):
        return 3 * cls.g / (2 * cls.l) * np.sin(th) + 3.0 / (cls.m * cls.l**2) * u

    def step(self, actions: np.ndarray):
        u = np.clip(np.asarray(actions, dtype=float).reshape(self.n), -self.max_torque, self.max_torque)
        wrapped = (self.th + np.pi) % (2 * np.pi) - np.pi
        reward = -(wrapped**2 + 0.1 * self.thd**2 + 0.001 * u**2)
        self.thd = np.clip(self.thd + self.angular_acceleration(self.th, u) * self.dt, -self.max_speed, self.max_speed)
        self.th = self.th + self.thd * self.dt
        self.t += 1
        terminated = np.zeros(self.n, dtype=bool)
        truncated = self.t >= self.max_steps
        final_obs = self.obs()
        if truncated.any():
            self.reset_where(truncated)
        return self.obs(), reward, terminated, truncated, final_obs


ENVS = {"cartpole": CartPole, "pendulum": Pendulum}

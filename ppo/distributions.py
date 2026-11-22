"""Action distributions with closed-form log-probabilities, entropies and their gradients."""

from __future__ import annotations

import numpy as np

LOG_2PI = float(np.log(2 * np.pi))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Categorical:
    """π(a|s) = softmax(logits)[a].

    log π(a) = zₐ - logsumexp(z)        ∂ log π(a) / ∂z = onehot(a) - p
    H = -Σ pⱼ log pⱼ                     ∂H / ∂zⱼ = -pⱼ (log pⱼ + H)
    """

    def __init__(self, logits: np.ndarray) -> None:
        self.logp_all = log_softmax(logits)
        self.p = np.exp(self.logp_all)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        u = rng.random((len(self.p), 1))
        return (u > np.cumsum(self.p, axis=1)).sum(axis=1).clip(max=self.p.shape[1] - 1)

    def mode(self) -> np.ndarray:
        return self.p.argmax(axis=1)

    def log_prob(self, a: np.ndarray) -> np.ndarray:
        return np.take_along_axis(self.logp_all, a[:, None].astype(int), axis=1)[:, 0]

    def entropy(self) -> np.ndarray:
        return -(self.p * self.logp_all).sum(axis=1)

    def grad_log_prob(self, a: np.ndarray) -> np.ndarray:
        g = -self.p.copy()
        g[np.arange(len(a)), a.astype(int)] += 1.0
        return g

    def grad_entropy(self) -> np.ndarray:
        return -self.p * (self.logp_all + self.entropy()[:, None])


class DiagGaussian:
    """π(a|s) = N(μ(s), diag σ²), with log σ a free (state-independent) parameter.

    log π(a) = Σₖ [-½ ((aₖ - μₖ)/σₖ)² - log σₖ - ½ log 2π]
        ∂/∂μₖ = (aₖ - μₖ) / σₖ²        ∂/∂ log σₖ = ((aₖ - μₖ)/σₖ)² - 1
    H = Σₖ (log σₖ + ½ log 2πe)          ∂H/∂μ = 0, ∂H/∂ log σₖ = 1
    """

    def __init__(self, mean: np.ndarray, log_std: np.ndarray) -> None:
        self.mean = mean
        self.log_std = np.broadcast_to(log_std, mean.shape)
        self.std = np.exp(self.log_std)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.mean + self.std * rng.standard_normal(self.mean.shape)

    def mode(self) -> np.ndarray:
        return self.mean

    def log_prob(self, a: np.ndarray) -> np.ndarray:
        z = (a - self.mean) / self.std
        return (-0.5 * z * z - self.log_std - 0.5 * LOG_2PI).sum(axis=1)

    def entropy(self) -> np.ndarray:
        return (self.log_std + 0.5 * (LOG_2PI + 1.0)).sum(axis=1)

    def grad_log_prob(self, a: np.ndarray):
        """Returns (∂/∂μ, ∂/∂ log σ), each shaped like the mean."""
        z = (a - self.mean) / self.std
        return z / self.std, z * z - 1.0

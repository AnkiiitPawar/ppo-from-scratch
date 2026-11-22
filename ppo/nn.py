"""A tanh multilayer perceptron with a hand-written backward pass, and the Adam optimiser."""

from __future__ import annotations

import numpy as np


def orthogonal(n_in: int, n_out: int, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Orthogonal initialisation (Saxe et al., 2014), standard for policy-gradient networks."""
    a = rng.standard_normal((max(n_in, n_out), min(n_in, n_out)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    w = q if n_in >= n_out else q.T
    return gain * w[:n_in, :n_out]


class MLP:
    """h₀ = x,  hᵢ₊₁ = tanh(hᵢ Wᵢ + bᵢ) for hidden layers, output = h Wₙ + bₙ (linear).

    Backward: for the output layer dz = dout; for a hidden layer dz = dh ⊙ (1 - h²);
    then dW = hᵀ dz, db = Σ dz and the gradient passed down is dz Wᵀ.
    """

    def __init__(self, sizes, rng: np.random.Generator, out_gain: float = 1.0) -> None:
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            self.W.append(orthogonal(a, b, gain, rng))
            self.b.append(np.zeros(b))
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(b) for b in self.b]

    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    def grads(self):
        return [g for pair in zip(self.gW, self.gb) for g in pair]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.hs = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            self.hs.append(h)
        return h

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Accumulates parameter gradients and returns d(loss)/d(input)."""
        d = dout
        n = len(self.W)
        for i in reversed(range(n)):
            if i < n - 1:
                d = d * (1.0 - self.hs[i + 1] ** 2)
            self.gW[i] += self.hs[i].T @ d
            self.gb[i] += d.sum(axis=0)
            d = d @ self.W[i].T
        return d

    def zero_grad(self) -> None:
        for g in self.grads():
            g.fill(0.0)


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction, updating parameters in place."""

    def __init__(self, params, grads, lr: float, betas=(0.9, 0.999), eps: float = 1e-5) -> None:
        self.params, self.grads = list(params), list(grads)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_grad_norm(grads, max_norm: float) -> float:
    total = float(np.sqrt(sum((g * g).sum() for g in grads)))
    if total > max_norm:
        for g in grads:
            g *= max_norm / (total + 1e-12)
    return total

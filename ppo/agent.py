"""Proximal Policy Optimisation (Schulman et al., 2017) with every gradient written out.

One iteration:
  1. run the current policy in N parallel environments for T steps;
  2. compute advantages with GAE(λ) (Schulman et al., 2016);
  3. for K epochs over shuffled minibatches, descend the clipped surrogate
         L = -E[min(r Â, clip(r, 1-ε, 1+ε) Â)] - c_H E[H(π)],   r = π(a|s) / π_old(a|s)
     and fit the value network to the GAE returns by least squares.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .distributions import Categorical, DiagGaussian
from .envs import ENVS
from .nn import MLP, Adam, clip_grad_norm


@dataclass
class PPOConfig:
    env: str = "cartpole"
    n_envs: int = 8
    n_steps: int = 128          # per environment, per iteration
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    lr: float = 2.5e-4
    lr_anneal: bool = True
    max_grad_norm: float = 0.5
    hidden: tuple = (64, 64)
    init_log_std: float = 0.0
    normalize_obs: bool = False
    scale_rewards: bool = False
    seed: int = 0


class RunningMeanStd:
    """Streaming mean and variance, merged batch by batch (Chan et al.'s parallel update)."""

    def __init__(self, shape=()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        b_mean, b_var, b_n = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalised advantage estimation over a (T, N) rollout.

    done[t] marks that the episode ended *after* step t, so nothing is bootstrapped across it:
        δₜ = rₜ + γ V(sₜ₊₁)(1 - doneₜ) - V(sₜ),    Âₜ = δₜ + γλ (1 - doneₜ) Âₜ₊₁
    Returns (advantages, value targets = advantages + values).
    """
    T = len(rewards)
    adv = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * live - values[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv, adv + values


def clipped_surrogate(logp_new, logp_old, adv, clip: float):
    """Loss -mean(min(r A, clip(r) A)) and its gradient with respect to logp_new.

    Where the clipped term is the smaller one the objective is flat in r, so those samples
    contribute no gradient; elsewhere ∂/∂ log π = -A r / n (because ∂r/∂ log π = r).
    """
    ratio = np.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
    loss = -float(np.minimum(unclipped, clipped).mean())
    active = unclipped <= clipped
    dlogp = np.where(active, -adv * ratio, 0.0) / len(adv)
    stats = {"clip_frac": float((~active & (adv != 0)).mean()),
             "approx_kl": float(((ratio - 1) - np.log(ratio)).mean())}
    return loss, dlogp, stats


@dataclass
class Log:
    steps: list = field(default_factory=list)
    returns: list = field(default_factory=list)     # (env step, episode return) for finished episodes
    updates: list = field(default_factory=list)     # per-iteration diagnostics
    seconds: float = 0.0


class PPO:
    def __init__(self, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = ENVS[cfg.env](cfg.n_envs, seed=cfg.seed)
        self.discrete = hasattr(self.env, "n_actions")
        obs_dim = self.env.obs_dim
        out_dim = self.env.n_actions if self.discrete else self.env.act_dim
        self.pi = MLP([obs_dim, *cfg.hidden, out_dim], self.rng, out_gain=0.01)
        self.vf = MLP([obs_dim, *cfg.hidden, 1], self.rng, out_gain=1.0)
        pi_params, pi_grads = self.pi.params(), self.pi.grads()
        if not self.discrete:
            self.log_std = np.full(out_dim, cfg.init_log_std)
            self.g_log_std = np.zeros(out_dim)
            pi_params, pi_grads = pi_params + [self.log_std], pi_grads + [self.g_log_std]
        self.pi_params, self.pi_grads = pi_params, pi_grads
        self.pi_opt = Adam(pi_params, pi_grads, cfg.lr)
        self.vf_opt = Adam(self.vf.params(), self.vf.grads(), cfg.lr)
        self.obs_rms = RunningMeanStd(obs_dim)
        self.ret_rms = RunningMeanStd(())
        self.ret_acc = np.zeros(cfg.n_envs)

    # ---- helpers -------------------------------------------------------------------
    def norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_obs:
            return obs
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

    def dist(self, out: np.ndarray):
        return Categorical(out) if self.discrete else DiagGaussian(out, self.log_std)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        d = self.dist(self.pi.forward(self.norm_obs(obs)))
        return d.mode() if deterministic else d.sample(self.rng)

    def evaluate(self, episodes: int = 10, seed: int = 10_000) -> float:
        """Mean return of the deterministic policy over fresh episodes."""
        env = ENVS[self.cfg.env](episodes, seed=seed)
        obs, total = env.reset(), np.zeros(episodes)
        finished = np.zeros(episodes, dtype=bool)
        while not finished.all():
            obs, r, term, trunc, _ = env.step(self.act(obs, deterministic=True))
            total += r * ~finished
            finished |= term | trunc
        return float(total.mean())

    # ---- training ------------------------------------------------------------------
    def train(self, total_steps: int, log_every: int = 10, log=print) -> Log:
        cfg, env = self.cfg, self.env
        N, T = cfg.n_envs, cfg.n_steps
        iterations = total_steps // (N * T)
        obs = env.reset()
        ep_ret, ep_steps = np.zeros(N), 0
        out = Log()
        t0 = time.perf_counter()
        for it in range(iterations):
            lr = cfg.lr * (1 - it / iterations) if cfg.lr_anneal else cfg.lr
            buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
            for _ in range(T):
                if cfg.normalize_obs:
                    self.obs_rms.update(obs)
                x = self.norm_obs(obs)
                d = self.dist(self.pi.forward(x))
                a = d.sample(self.rng)
                v = self.vf.forward(x)[:, 0]
                nxt, r, term, trunc, final = env.step(a)
                ep_ret += r
                ep_steps += N
                train_r = r.copy()
                if cfg.scale_rewards:
                    self.ret_acc = self.ret_acc * cfg.gamma + r
                    self.ret_rms.update(self.ret_acc)
                    train_r = r / np.sqrt(self.ret_rms.var + 1e-8)
                    self.ret_acc[term | trunc] = 0.0
                if trunc.any():  # a time limit is not a real ending: bootstrap from V(final state)
                    v_final = self.vf.forward(self.norm_obs(final[trunc]))[:, 0]
                    train_r[trunc] += cfg.gamma * v_final
                for i in np.flatnonzero(term | trunc):
                    out.steps.append(ep_steps)
                    out.returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0
                buf_obs.append(x); buf_act.append(a); buf_logp.append(d.log_prob(a))
                buf_rew.append(train_r); buf_done.append((term | trunc).astype(float)); buf_val.append(v)
                obs = nxt
            last_v = self.vf.forward(self.norm_obs(obs))[:, 0]
            adv, ret = gae(np.array(buf_rew), np.array(buf_val), np.array(buf_done), last_v, cfg.gamma, cfg.lam)
            stats = self.update(np.concatenate(buf_obs), np.concatenate(buf_act), np.concatenate(buf_logp),
                                adv.reshape(-1), ret.reshape(-1), lr)
            out.updates.append({"step": ep_steps, **stats})
            if log and (it % log_every == 0 or it == iterations - 1):
                recent = out.returns[-20:]
                mean_ret = np.mean(recent) if recent else float("nan")
                log(f"iter {it:4d}  steps {ep_steps:7d}  return {mean_ret:8.1f}  kl {stats['approx_kl']:.4f}"
                    f"  clip {stats['clip_frac']:.2f}  ({time.perf_counter() - t0:.0f} s)")
        out.seconds = time.perf_counter() - t0
        return out

    def update(self, obs, act, logp_old, adv, ret, lr):
        cfg = self.cfg
        n = len(obs)
        stats = []
        for _ in range(cfg.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, cfg.minibatch):
                mb = order[start:start + cfg.minibatch]
                a_mb = adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                stats.append(self.policy_step(obs[mb], act[mb], logp_old[mb], a_mb, lr))
                self.value_step(obs[mb], ret[mb], lr)
        return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}

    def policy_loss_and_grad(self, obs, act, logp_old, adv):
        """Total policy loss (surrogate - c_H * entropy); fills the policy gradients."""
        cfg = self.cfg
        self.pi.zero_grad()
        if not self.discrete:
            self.g_log_std.fill(0.0)
        d = self.dist(self.pi.forward(obs))
        surrogate, dlogp, stats = clipped_surrogate(d.log_prob(act), logp_old, adv, cfg.clip)
        entropy = d.entropy()
        n = len(obs)
        dent = -cfg.ent_coef / n  # the loss subtracts c_H * mean(entropy)
        if self.discrete:
            dout = dlogp[:, None] * d.grad_log_prob(act) + dent * d.grad_entropy()
        else:
            g_mu, g_ls = d.grad_log_prob(act)
            dout = dlogp[:, None] * g_mu
            self.g_log_std += (dlogp[:, None] * g_ls).sum(axis=0) + dent * n  # ∂H/∂ log σ = 1 per sample
        self.pi.backward(dout)
        stats["entropy"] = float(entropy.mean())
        stats["policy_loss"] = surrogate
        return surrogate - cfg.ent_coef * float(entropy.mean()), stats

    def policy_step(self, obs, act, logp_old, adv, lr) -> dict:
        _, stats = self.policy_loss_and_grad(obs, act, logp_old, adv)
        clip_grad_norm(self.pi_grads, self.cfg.max_grad_norm)
        self.pi_opt.step(lr)
        return stats

    def value_step(self, obs, ret, lr) -> None:
        self.vf.zero_grad()
        v = self.vf.forward(obs)[:, 0]
        self.vf.backward(((v - ret) / len(ret))[:, None])  # gradient of ½ mean (V - R)²
        clip_grad_norm(self.vf.grads(), self.cfg.max_grad_norm)
        self.vf_opt.step(lr)

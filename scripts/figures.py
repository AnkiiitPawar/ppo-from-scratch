"""README figures: learning curves over seeds, and rollouts of the seed-0 policies.

usage: python scripts/figures.py     (after scripts/run_experiments.py; needs matplotlib)
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo import PPO  # noqa: E402
from ppo.envs import CartPole, Pendulum  # noqa: E402
from scripts.run_experiments import SETTINGS  # noqa: E402

DOCS = Path("docs")


def binned(steps, returns, edges):
    steps, returns = np.asarray(steps), np.asarray(returns)
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        sel = (steps > edges[i]) & (steps <= edges[i + 1])
        if sel.any():
            out[i] = returns[sel].mean()
    return out


def learning_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, (env, title, ref) in zip(axes, [("cartpole", "Cart-pole (max 500)", 500),
                                            ("pendulum", "Pendulum swing-up", None)]):
        res = results[env]
        edges = np.linspace(0, res["steps"], 41)
        curves = np.array([binned(r["episode_end_step"], r["episode_return"], edges) for r in res["runs"]])
        mid = 0.5 * (edges[1:] + edges[:-1]) / 1000
        for c in curves:
            ax.plot(mid, c, color="#9fb7d0", lw=0.8)
        ax.plot(mid, np.nanmean(curves, axis=0), color="#1f5f99", lw=2, label=f"mean of {len(curves)} seeds")
        if ref:
            ax.axhline(ref, color="#c0504d", lw=0.8, ls="--")
        ax.set(xlabel="environment steps (thousands)", ylabel="episode return (training policy)", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "learning_curves.png", dpi=130)


def load_agent(env):
    cfg, _ = SETTINGS[env]
    agent = PPO(cfg)
    data = np.load(f"runs/{env}_seed0.npz")
    for i, p in enumerate(agent.pi_params):
        p[...] = data[f"p{i}"]
    agent.obs_rms.mean, agent.obs_rms.var = data["obs_mean"], data["obs_var"]
    return agent


def pendulum_swingup():
    agent = load_agent("pendulum")
    env = Pendulum(1, seed=0)
    env.th[:], env.thd[:] = np.pi, 0.0  # hanging straight down, at rest: the hardest start
    th, u = [env.th[0]], []
    for _ in range(150):
        a = np.clip(agent.act(env.obs(), deterministic=True), -2, 2)
        env.step(a)
        u.append(float(a[0, 0]))
        th.append(env.th[0])
    th = (np.array(th) + np.pi) % (2 * np.pi) - np.pi
    t = np.arange(len(th)) * Pendulum.dt

    fig = plt.figure(figsize=(10, 3.8))
    ax0 = fig.add_axes([0.03, 0.1, 0.3, 0.8])
    for k in range(0, 90, 3):  # strobe of the first 4.5 seconds
        x, y = np.sin(th[k]), np.cos(th[k])
        ax0.plot([0, x], [0, y], color=plt.cm.viridis(k / 90), lw=2, alpha=0.35 + 0.65 * k / 90)
    ax0.plot(0, 0, "ko", ms=4)
    ax0.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", xticks=[], yticks=[],
            title="rod every 0.15 s, 0 to 4.5 s (dark to light)")
    ax1 = fig.add_axes([0.42, 0.56, 0.55, 0.36])
    ax1.plot(t, np.cos(th), color="#1f5f99")
    ax1.axhline(1, color="#999", lw=0.6)
    ax1.set(ylabel="tip height, cos θ", xticklabels=[], yticks=[-1, 0, 1], ylim=(-1.1, 1.1))
    ax1.text(t[-1], 0.85, "upright", ha="right", fontsize=8, color="#666")
    ax2 = fig.add_axes([0.42, 0.13, 0.55, 0.36])
    ax2.step(t[:-1], u, where="post", color="#c0504d")
    ax2.set(xlabel="time (s)", ylabel="torque (N·m)", ylim=(-2.3, 2.3))
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(DOCS / "pendulum_swingup.png", dpi=130)


def cartpole_rollout():
    agent = load_agent("cartpole")
    env = CartPole(1, seed=0)
    env.state[:] = [0.0, 0.0, 0.1, 0.0]  # start tilted by 0.1 rad, near the 0.21 rad failure limit
    xs, ths = [0.0], [0.1]
    for _ in range(499):
        _, _, term, _, _ = env.step(agent.act(env.state, deterministic=True))
        if term[0]:
            break
        xs.append(env.state[0, 0])
        ths.append(env.state[0, 2])
    t = np.arange(len(xs)) * CartPole.dt
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
    a.plot(t, xs, color="#1f5f99")
    a.axhspan(-2.4, 2.4, color="#e8f0f8", zorder=0)
    a.set(ylabel="cart x (m)", ylim=(-2.6, 2.6))
    b.plot(t, np.degrees(ths), color="#c0504d")
    b.axhspan(-12, 12, color="#f8ece8", zorder=0)
    b.set(ylabel="pole angle (°)", xlabel="time (s)", ylim=(-13, 13))
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    a.set_title(f"Greedy policy from a 0.1 rad tilt: balanced for {len(xs)} steps (shaded: allowed range)",
                fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "cartpole_rollout.png", dpi=130)


if __name__ == "__main__":
    results = json.loads((DOCS / "results.json").read_text())
    learning_curves(results)
    pendulum_swingup()
    cartpole_rollout()
    print("figures written to docs/")

"""Train PPO on both environments over several seeds; save learning curves and seed-0 policies.

usage: python scripts/run_experiments.py [--seeds 5]
Writes docs/results.json and runs/<env>_seed0.npz.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo import PPO, PPOConfig  # noqa: E402

SETTINGS = {
    "cartpole": (PPOConfig(env="cartpole"), 150_000),
    "pendulum": (PPOConfig(env="pendulum", n_envs=8, n_steps=256, epochs=10, minibatch=256, gamma=0.9,
                           lr=1e-3, ent_coef=0.0, normalize_obs=True, scale_rewards=True), 300_000),
}


def save_policy(agent: PPO, path: Path) -> None:
    arrays = {f"p{i}": p for i, p in enumerate(agent.pi_params)}
    np.savez(path, obs_mean=agent.obs_rms.mean, obs_var=agent.obs_rms.var, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    results = {}
    Path("runs").mkdir(exist_ok=True)
    for env, (base, steps) in SETTINGS.items():
        results[env] = {"config": {k: v for k, v in asdict(base).items() if k != "seed"}, "steps": steps, "runs": []}
        for seed in range(args.seeds):
            cfg = PPOConfig(**{**asdict(base), "seed": seed})
            agent = PPO(cfg)
            t0 = time.perf_counter()
            log = agent.train(steps, log=None)
            final = agent.evaluate(episodes=20)
            print(f"{env} seed {seed}: final greedy return {final:.1f} ({time.perf_counter() - t0:.0f} s)")
            results[env]["runs"].append({"seed": seed, "episode_end_step": log.steps, "episode_return": log.returns,
                                          "final_eval": final, "seconds": log.seconds})
            if seed == 0:
                save_policy(agent, Path(f"runs/{env}_seed0.npz"))
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()

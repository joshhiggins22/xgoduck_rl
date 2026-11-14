"""SB3 PPO trainer for the XgoDuck sit->stand CPU-MuJoCo env (Mac).

Proven guards from the M1 gait skill: ent_coef 0.005, target_kl 0.03, gae_lambda
0.99, linear LR decay, versioned checkpoints, progress.jsonl + eval callback.
Parameterized via env vars so curriculum stages are one-line relaunches.

  HEEL=1            fit the +15mm heel clip
  N_ENVS=12         SubprocVecEnv workers
  TOTAL_STEPS=20000000
  N_STEPS=2048      rollout length per worker (>= episode cap for survival)
  LR=3e-4           (decays to LR/10)
  ENT=0.005  KL=0.03  LAMBDA=0.99  GAMMA=0.99
  RUN_NAME=mac-sitstand
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from duck_sitstand_env import DuckSitStandEnv

HERE = Path(__file__).resolve().parent


def _env_fn():
    heel = os.environ.get("HEEL", "0") == "1"
    return DuckSitStandEnv(heel=heel)


class ProgressEvalCallback(BaseCallback):
    """Periodic fresh-env eval + progress.jsonl + versioned checkpoints."""

    def __init__(self, eval_every=200_000, n_eval=8, run_dir=None):
        super().__init__()
        self.eval_every = eval_every
        self.n_eval = n_eval
        self.run_dir = Path(run_dir)
        self._next = eval_every
        self._t0 = time.time()

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            self._eval()
        return True

    def _eval(self):
        env = _env_fn()  # fresh instance, never reach into training_env
        lens, rews, stands, fell_count = [], [], 0, 0
        for ep in range(self.n_eval):
            obs, _ = env.reset(seed=1000 + ep)
            done, ep_r, steps = False, 0.0, 0
            last = {}
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, last = env.step(a)
                ep_r += r
                steps += 1
                done = term or trunc
            lens.append(steps)
            rews.append(ep_r)
            fell_count += int(last.get("fell", False))
            stands += int(last.get("z", 0) > 0.125 and not last.get("fell", False))
        rec = {
            "timesteps": self.num_timesteps,
            "wall_s": round(time.time() - self._t0, 1),
            "eval_ep_len_mean": float(np.mean(lens)),
            "eval_rew_mean": float(np.mean(rews)),
            "eval_stand_frac": stands / self.n_eval,
            "eval_fell_frac": fell_count / self.n_eval,
        }
        with open(self.run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.model.save(str(self.run_dir / f"ckpt_{self.num_timesteps}"))
        print("[eval]", json.dumps(rec), flush=True)


def main():
    n_envs = int(os.environ.get("N_ENVS", "12"))
    total = int(os.environ.get("TOTAL_STEPS", "20_000_000"))
    n_steps = int(os.environ.get("N_STEPS", "2048"))
    lr0 = float(os.environ.get("LR", "3e-4"))
    run_name = os.environ.get("RUN_NAME", "mac-sitstand")
    run_dir = HERE / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([_env_fn for _ in range(n_envs)]))
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=lambda frac: lr0 * (0.1 + 0.9 * frac),  # linear decay to lr0/10
        n_steps=n_steps,
        batch_size=int(os.environ.get("BATCH", "4096")),
        gamma=float(os.environ.get("GAMMA", "0.99")),
        gae_lambda=float(os.environ.get("LAMBDA", "0.99")),
        ent_coef=float(os.environ.get("ENT", "0.005")),
        target_kl=float(os.environ.get("KL", "0.03")),
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log=None,
    )
    cb = ProgressEvalCallback(
        eval_every=int(os.environ.get("EVAL_EVERY", "200000")),
        n_eval=int(os.environ.get("N_EVAL", "8")),
        run_dir=run_dir,
    )
    print(f"[train] run={run_name} heel={os.environ.get('HEEL','0')} envs={n_envs} "
          f"steps={total} n_steps={n_steps} lr={lr0}", flush=True)
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    model.save(str(run_dir / "final_model"))
    print("[train] done ->", run_dir / "final_model.zip", flush=True)


if __name__ == "__main__":
    main()
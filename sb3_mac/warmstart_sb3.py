"""Warm-start SB3 PPO from the deployed official XgoDuck ONNX actor, then fine-tune
the sit->stand rise (with the heel clip).

The official policy already holds stand in this env; warm-starting carries that
balance prior so RL only has to discover the rise. Guards against the fresh-critic
collapse: the actor is FROZEN for the first WARMUP_STEPS timesteps (critic learns a
stable value function first), then unfrozen at a low fine-tune LR.

ONNX actor = obs_normalizer -> ELU MLP 61->512->256->128->14. We bake the normalizer
into a gym wrapper and map the MLP onto SB3's policy_net + action_net.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from duck_sitstand_env import DuckSitStandEnv

HERE = Path(__file__).resolve().parent
ONNX_PATH = os.environ.get("ONNX", "/Users/joshhiggins/duck-probe/official_sitstand.onnx")
HEEL = os.environ.get("HEEL", "1") == "1"


def load_onnx_actor(path):
    # The verified warm-start checkpoint carries the exact obs normalizer (_mean/_std)
    # and the MLP weights; the raw ONNX stores std as a Div constant, so prefer the .pt.
    pt = os.environ.get("WARMSTART_PT", "/Users/joshhiggins/duck-probe/official_warmstart.pt")
    sd = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = np.asarray(sd["obs_normalizer._mean"]).reshape(-1).astype(np.float64)
    if "obs_normalizer._std" in sd:
        std = np.asarray(sd["obs_normalizer._std"]).reshape(-1).astype(np.float64)
        var = std**2
    else:
        var = np.asarray(sd["obs_normalizer._var"]).reshape(-1).astype(np.float64)
        std = np.sqrt(var + 1e-8)
    w = {k: v.numpy() for k, v in sd.items() if k.startswith("mlp.")}
    return mean, std, var, w


class ObsNormWrapper(gym.Wrapper):
    def __init__(self, env, mean, std, var):
        super().__init__(env)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.zero_var = var < 1e-8

    def _norm(self, obs):
        n = (obs - self.mean) / self.std
        n[self.zero_var] = 0.0
        return np.clip(n, -5.0, 5.0).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._norm(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._norm(obs), r, term, trunc, info


def make_env(mean, std, var):
    return ObsNormWrapper(DuckSitStandEnv(heel=HEEL), mean, std, var)


def _actor_params(policy):
    for name, param in policy.named_parameters():
        if "value_net" not in name:
            yield param


class ActorFreezeCallback(BaseCallback):
    def __init__(self, warmup_steps):
        super().__init__()
        self.warmup_steps = warmup_steps
        self._unfrozen = False

    def _on_step(self):
        if not self._unfrozen and self.num_timesteps >= self.warmup_steps:
            for param in _actor_params(self.model.policy):
                param.requires_grad_(True)
            self._unfrozen = True
            print(f"[warmstart] actor UNFROZEN at {self.num_timesteps} steps", flush=True)
        return True


class WarmstartEvalCallback(BaseCallback):
    """Eval on the SAME normalized env the policy was trained on."""
    def __init__(self, mean, std, var, eval_every, n_eval, run_dir):
        super().__init__()
        self.mean, self.std, self.var = mean, std, var
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
        env = make_env(self.mean, self.std, self.var)
        lens, rews, stands, fells = [], [], 0, 0
        for ep in range(self.n_eval):
            obs, _ = env.reset(seed=1000 + ep)
            done, ep_r, steps, last = False, 0.0, 0, {}
            while not done:
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, last = env.step(a)
                ep_r += r; steps += 1; done = term or trunc
            lens.append(steps); rews.append(ep_r)
            fells += int(last.get("fell", False))
            stands += int(last.get("z", 0) > 0.125 and not last.get("fell", False))
        rec = {
            "timesteps": self.num_timesteps, "wall_s": round(time.time() - self._t0, 1),
            "eval_ep_len_mean": float(np.mean(lens)), "eval_rew_mean": float(np.mean(rews)),
            "eval_stand_frac": stands / self.n_eval, "eval_fell_frac": fells / self.n_eval,
        }
        with open(self.run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.model.save(str(self.run_dir / f"ckpt_{self.num_timesteps}"))
        print("[eval]", json.dumps(rec), flush=True)


def main():
    mean, std, var, w = load_onnx_actor(ONNX_PATH)
    n_envs = int(os.environ.get("N_ENVS", "12"))
    total = int(os.environ.get("TOTAL_STEPS", "20_000_000"))
    warmup = int(os.environ.get("WARMUP_STEPS", "1_000_000"))
    run_name = os.environ.get("RUN_NAME", "mac-heel-warmstart")
    run_dir = HERE / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([lambda: make_env(mean, std, var) for _ in range(n_envs)]))
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=float(os.environ.get("LR", "3e-5")),
        n_steps=int(os.environ.get("N_STEPS", "2048")),
        batch_size=int(os.environ.get("BATCH", "4096")),
        gamma=0.99, gae_lambda=0.99,
        ent_coef=float(os.environ.get("ENT", "0.003")),
        target_kl=float(os.environ.get("KL", "0.02")),
        clip_range=0.1,
        policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=torch.nn.ELU),
        verbose=1,
    )
    sd = model.policy.state_dict()
    mapping = {
        "mlp.0.weight": "mlp_extractor.policy_net.0.weight",
        "mlp.0.bias": "mlp_extractor.policy_net.0.bias",
        "mlp.2.weight": "mlp_extractor.policy_net.2.weight",
        "mlp.2.bias": "mlp_extractor.policy_net.2.bias",
        "mlp.4.weight": "mlp_extractor.policy_net.4.weight",
        "mlp.4.bias": "mlp_extractor.policy_net.4.bias",
        "mlp.6.weight": "action_net.weight",
        "mlp.6.bias": "action_net.bias",
    }
    loaded = 0
    for src, dst in mapping.items():
        if src in w and dst in sd:
            sd[dst] = torch.tensor(w[src], dtype=sd[dst].dtype)
            loaded += 1
    model.policy.load_state_dict(sd)
    print(f"[warmstart] loaded {loaded}/8 actor tensors from ONNX", flush=True)
    with torch.no_grad():
        model.policy.log_std.fill_(float(os.environ.get("LOG_STD", "-1.5")))
    for param in _actor_params(model.policy):
        param.requires_grad_(False)
    print(f"[warmstart] actor FROZEN for first {warmup} steps; heel={HEEL}", flush=True)

    cb = [
        ActorFreezeCallback(warmup),
        WarmstartEvalCallback(mean, std, var,
                              eval_every=int(os.environ.get("EVAL_EVERY", "500000")),
                              n_eval=8, run_dir=run_dir),
    ]
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    model.save(str(run_dir / "final_model"))
    print("[warmstart] done ->", run_dir / "final_model.zip", flush=True)


if __name__ == "__main__":
    main()
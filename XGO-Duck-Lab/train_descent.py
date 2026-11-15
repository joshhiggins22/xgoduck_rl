"""Fit the joint-bounded sit maneuver with two-worker CEM in native MuJoCo."""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from controlled_cycle import descent_action
from duck_env import ROOT, DuckSitStandEnv
from learner import onnx_session, predict_onnx
from rise import load_policy

INITIAL = np.array([0.04, 0.85, -0.4, 0.0, -0.03, 1.40, -0.39, 0.0, 0.0, 0.0])
LOW = np.array([-1.15, 0.1, -1.92, -2.66, -1.15, 0.85, -1.92, -2.66, -1.8, -1.8])
HIGH = np.array([1.92, 1.55, 1.16, 0.67, 1.92, 1.55, 1.16, 0.67, 1.8, 1.8])


def initialize_worker():
    global factory, rise, environments
    factory = onnx_session(ROOT / "runtime/xgoduck_sitstand.onnx")
    rise = load_policy(ROOT / "runs/heel-rise-bounded/rise.onnx")
    environments = {}


def rollout(case):
    parameters, surface, seed, rise_seconds = case
    key = (surface, rise_seconds)
    if key not in environments:
        environments[key] = DuckSitStandEnv("v2-heel15", surface, "stand", 1.8 + rise_seconds)
    env = environments[key]
    obs, _ = env.reset(seed=seed)
    seated_steps = 0
    descent_steps = 90
    for tick in range(env.max_steps):
        if tick < descent_steps:
            cmd = env.feedback.obs(env._last_action, cmd=[1.0, 0.0, 0.0])
            action = descent_action(predict_onnx(factory, cmd), cmd, tick, parameters)
        else:
            if tick == descent_steps:
                rise.reset(obs)
            action = rise(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if tick < descent_steps:
            q = env.data.qpos[env.qpos_adr]
            knee = min(abs(q[3]), abs(q[12]))
            tilt, z = info["tilt_deg"], info["z"]
            seat_frame = knee > 0.8 and z < 0.115 and tilt < 25.0
            seated_steps = seated_steps + 1 if seat_frame else 0
            seated = seated_steps >= 12
            pose_error = np.mean((q[[2, 3, 4, 5]] - [-0.4079, 1.35, 0.0, 0.349]) ** 2)
            pose_reward = (
                np.exp(-((min(0.0, knee - 1.25) / 0.35) ** 2))
                + 1.3 * np.exp(-(((z - 0.100) / 0.026) ** 2))
                + 1.6 * np.exp(-((tilt / 23.0) ** 2))
                + 2.0 * np.exp(-pose_error / 0.23)
            )
            seat_state = {"z": z, "tilt": tilt, "knee": knee, "seated": seated}
            if tick == descent_steps - 1 and not seated:
                break
        if terminated or truncated:
            break
    streak = info["upright_streak_s"]
    success = seated and not terminated and truncated and streak >= 1.0
    score = min((tick + 1) / descent_steps, 1.0) + pose_reward + 10 * seated + 35 * success
    if tick >= descent_steps:
        score += 8 * streak / rise_seconds
    return {
        "score": float(score),
        "seat": bool(seated),
        "rise": bool(success),
        "fell": bool(terminated),
        "seat_state": seat_state,
        "tick": tick,
        "end_streak": float(streak),
    }


def fit(out, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(740)
    mean = INITIAL.copy()
    sigma = np.array([0.37, 0.34, 0.46, 0.4, 0.37, 0.22, 0.46, 0.5, 0.85, 0.85])
    finalists = []
    with multiprocessing.get_context("spawn").Pool(workers, initializer=initialize_worker) as pool:
        for generation in range(7):
            candidates = np.clip(rng.normal(mean, sigma, (36, len(mean))), LOW, HIGH)
            candidates[0] = mean
            surface = "hard" if generation % 2 == 0 else "carpet"
            cases = [(p, surface, 3100 + generation, 2.6) for p in candidates]
            outcomes = pool.map(rollout, cases)
            scores = np.array([r["score"] for r in outcomes])
            elite = scores.argsort()[-5:]
            mean = candidates[elite].mean(0)
            sigma = np.maximum(0.06, candidates[elite].std(0) * 1.22)
            finalists.extend((scores[i], candidates[i].tolist()) for i in elite[-2:])
            print(
                f"Generation {generation + 1}: {sum(r['rise'] for r in outcomes)}/36 cycles",
                flush=True,
            )

        validation = []
        for _, candidate in sorted(finalists, reverse=True)[:6]:
            cases = [
                (candidate, surface, seed, 6.0)
                for surface in ("hard", "carpet")
                for seed in (3300, 3301)
            ]
            outcomes = pool.map(rollout, cases)
            rank = (sum(r["rise"] for r in outcomes), min(r["score"] for r in outcomes))
            validation.append((rank, candidate, outcomes))
        _, selected, validation_trials = max(validation, key=lambda row: row[0])
        cases = [
            (selected, surface, seed, 6.0)
            for surface in ("hard", "carpet")
            for seed in range(3500, 3512)
        ]
        test = [
            dict(surface=case[1], seed=case[2], **r) for case, r in zip(cases, pool.map(rollout, cases))
        ]
    report = {"params": selected, "validation": validation_trials}
    (out / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "held-out.json").write_text(json.dumps(dict(**report, test=test), indent=2) + "\n")
    print(f"Held out: {sum(r['rise'] for r in test)}/{len(test)} cycles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    fit(args.out, args.workers)

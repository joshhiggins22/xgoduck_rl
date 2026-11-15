"""Seated-start transfer check: factory Sit, then factory or PPO stand on the same sensor/action history."""

import argparse
import json
from pathlib import Path
import numpy as np
from duck_env import DuckSitStandEnv, DT, DECIMATION
from learner import SOURCE, onnx_session, predict_onnx
from lab import policy


def trial(env, stock, candidate, seed, dwell):
    obs, _ = env.reset(seed=seed, options={"start": "stand"})
    seated_count = 0
    for _ in range(round(dwell / (DT * DECIMATION))):
        obs = obs.copy()
        obs[48] = 1.0
        obs, _, term, _, info = env.step(predict_onnx(stock, obs))
        both = np.abs(env.data.qpos[env.qpos_adr[[3, 12]]])
        seated_count = (
            seated_count + 1 if info["z"] < 0.115 and info["tilt_deg"] < 25 and min(both) > 0.8 else 0
        )
        if term:
            return dict(seed=seed, seated=False, success=False, fell=True)
    seated = seated_count >= 12
    if hasattr(candidate, "reset"):
        candidate.reset(obs)
    streak = 0
    while True:
        obs, _, term, trunc, info = env.step(candidate(obs))
        streak = streak + 1 if info["upright"] else 0
        if term or trunc:
            break
    return dict(
        seed=seed,
        seated=seated,
        success=bool(seated and not term and streak >= 50),
        fell=bool(term),
        final_hold_s=round(streak * DT * DECIMATION, 3),
        end_tilt_deg=info["tilt_deg"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--foot", default="v2-stock")
    p.add_argument("--surface", default="carpet")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=3100)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    env = DuckSitStandEnv(args.foot, args.surface, "stand", episode_s=9.0)
    stock = onnx_session(SOURCE)
    blocks = {}
    for label, path in [("factory", SOURCE), ("candidate", args.candidate)]:
        pred = policy(path)
        rows = [
            trial(env, stock, pred, seed, 3.0) for seed in range(args.seed, args.seed + args.episodes)
        ]
        counts = {key: sum(row[key] for row in rows) for key in ["seated", "success", "fell"]}
        blocks[label] = dict(**counts, trials=rows)
        print(label, counts, flush=True)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(
        json.dumps(
            dict(
                foot=args.foot,
                surface=args.surface,
                duration_s=9,
                policy=str(args.candidate),
                command_sequence="3 s Sit using factory; 6 s Stand; sensor and action history retained",
                comparison=blocks,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

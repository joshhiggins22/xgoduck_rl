"""Evaluate the experimental learned descent and rise as one continuous command cycle."""

import argparse
import json
from pathlib import Path

import numpy as np

from duck_env import ROOT, DuckSitStandEnv
from learner import onnx_session, predict_onnx
from rise import load_policy


def descent_action(factory_action, obs, tick, parameters):
    """Symmetric, state-dependent descent at 50 Hz from upright to a sustained seat."""
    p = np.asarray(parameters)
    first = min(tick / 30.0, 1.0)
    second = float(np.clip((tick - 30) / 30.0, 0.0, 1.0))
    pose = p[:4] * first * (1 - second) + p[4:8] * second
    pose = pose.copy()
    pose[0] += p[8] * obs[3]
    pose[2] += p[9] * obs[3]
    pose = np.clip(pose, [-1.167, -1.55, -1.93, -2.67], [1.93, 1.55, 1.167, 0.67])
    action = factory_action.copy()
    action[[2, 3, 4, 5]] = pose
    action[[11, 12, 13]] = -pose[:3]
    action[[1, 10, 6]] = [0.087 * first, -0.087 * first, 0.0]
    return action


def evaluate(parameters, surface, seeds, sit_duration=1.8, rise_duration=6.0, observer=None):
    if sit_duration < 1.2 or rise_duration < 1.0:
        raise ValueError("Descent takes 1.2 s, and the final upright check requires 1 s.")
    steps = round(sit_duration / 0.02)
    env = DuckSitStandEnv("v2-heel15", surface, "stand", sit_duration + rise_duration)
    factory = onnx_session(ROOT / "runtime/xgoduck_sitstand.onnx")
    rise = load_policy(ROOT / "runs/heel-rise-bounded/rise.onnx")
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        consecutive = 0
        seated = False
        at_switch = None
        for tick in range(env.max_steps):
            if observer is not None:
                observer(env, tick)
            if tick < steps:
                cmd = env.feedback.obs(env._last_action, cmd=[1.0, 0.0, 0.0])
                action = descent_action(predict_onnx(factory, cmd), cmd, tick, parameters)
            else:
                if tick == steps:
                    if not seated:
                        break
                    rise.reset(obs)
                action = rise(obs)
            obs, _, term, trunc, info = env.step(action)
            if tick < steps:
                knees = min(abs(env.data.qpos[env.qpos_adr[[3, 12]]]))
                ready = knees > 0.8 and info["z"] < 0.115 and info["tilt_deg"] < 25.0
                consecutive = consecutive + 1 if ready else 0
                if tick == steps - 1:
                    seated = consecutive >= 12
                    at_switch = {
                        "height_m": info["z"],
                        "tilt_deg": info["tilt_deg"],
                        "both_knee_min_rad": knees,
                    }
            if term or trunc:
                break
        success = seated and trunc and not term and info["upright_streak_s"] >= 1.0
        rows.append(
            {
                "seed": seed,
                "success": bool(success),
                "seated": seated,
                "fell": info["fell"],
                "at_stand_command": at_switch,
                "final_upright_s": info["upright_streak_s"],
            }
        )
    return {
        "foot": "v2-heel15",
        "surface": surface,
        "sit_s": sit_duration,
        "rise_s": rise_duration,
        "successes": sum(r["success"] for r in rows),
        "episodes": len(rows),
        "trials": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parameters",
        type=Path,
        default=ROOT / "runs/heel-cycle-descent/parameters.json",
    )
    p.add_argument("--surface", choices=["hard", "carpet"], default="carpet")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=3500)
    p.add_argument("--sit-seconds", type=float, default=1.8)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = evaluate(
        json.loads(args.parameters.read_text())["params"],
        args.surface,
        range(args.seed, args.seed + args.episodes),
        args.sit_seconds,
    )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['successes']}/{result['episodes']} complete cycles")

"""Fit a short seated-rise maneuver by cross-entropy search in CPU MuJoCo."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from duck_env import DuckSitStandEnv, ROOT
from rise import END_TICK, export, table


def initialize():
    import onnxruntime as ort

    global teacher, envs
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    teacher = ort.InferenceSession(
        str(ROOT / "runtime/xgoduck_sitstand.onnx"), opt, providers=["CPUExecutionProvider"]
    )
    envs = {}


def evaluate_case(case):
    gains, start, surface, seed, seconds = case
    key = (start, surface, seconds)
    if key not in envs:
        envs[key] = DuckSitStandEnv("v2-heel15", surface, start, seconds)
    env = envs[key]
    obs, _ = env.reset(seed=seed)
    seated = (obs[9] - obs[18]) * 0.5 > 0.65
    poses, weights = table(gains)
    total = peak = 0.0
    for step in range(env.max_steps):
        base = teacher.run(None, {"obs": obs.reshape(1, 61)})[0][0]
        tick = min(step, END_TICK) if seated else END_TICK
        action = base + weights[tick] * (poses[tick] - base)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        quality = np.exp(
            -(((info["z"] - 0.139) / 0.027) ** 2)
            - (info["tilt_deg"] / 28.0) ** 2
            - (info["knee_rad"] / 0.75) ** 2
        )
        if step >= 12:
            peak = max(peak, quality)
        if term or trunc:
            break
    success = not term and info["upright_streak_s"] >= 1.0
    final = np.exp(-(((info["z"] - 0.139) / 0.02) ** 2) - (info["tilt_deg"] / 20.0) ** 2)
    score = (
        30 * success
        + 15 * peak
        + 6 * info["upright_streak_s"] / seconds
        + 3 * total / env.max_steps
        + 2 * final
    )
    return float(score), bool(success)


def select(pool, choices, seeds):
    trials = [
        (g, start, surf, seed, 6.0)
        for g in choices
        for start in ("sit", "stand")
        for surf in ("hard", "carpet")
        for seed in seeds
    ]
    results = pool.map(evaluate_case, trials)
    size = 4 * len(seeds)

    def rank(idx):
        subset = results[idx * size : (idx + 1) * size]
        return (
            sum(row[1] for row in subset),
            min(row[0] for row in subset),
            -np.linalg.norm(choices[idx]),
        )

    winner = max(range(len(choices)), key=rank)
    return choices[winner], rank(winner)[0], size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="New output directory.")
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--generations", type=int, default=9)
    p.add_argument("--population", type=int, default=52)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--joint-bounds",
        action="store_true",
        help="Limit scripted joint requests to just inside simulated joint stops during search.",
    )
    p.add_argument(
        "--from-json", type=Path, help="Initialize the search around a previously fitted set of knots."
    )
    args = p.parse_args()
    if args.generations < 1 or args.population < 8 or args.workers < 1:
        p.error("Require >=1 generation, >=8 candidates and >=1 worker.")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    mean = np.array([-0.1, 0.95, -0.15, 0.25, -0.25, 0.45, 0.05, 0.45, 0, 0.05, 0.1, 0.1])
    std = np.tile([0.6, 0.55, 0.6, 0.70], 3)
    lo, hi = np.full(12, -2.0), np.full(12, 2.0)
    if args.joint_bounds:
        lo = np.tile([-1.167, -1.55, -1.93, -2.67], 3)
        hi = np.tile([1.93, 1.55, 1.167, 0.67], 3)
    if args.from_json:
        mean = np.clip(np.array(json.loads(args.from_json.read_text())["knots"]).reshape(12), lo, hi)
        std *= 0.60
    history = []
    with mp.get_context("spawn").Pool(args.workers, initializer=initialize) as pool:
        for k in range(args.generations):
            pop = np.clip(rng.normal(mean, std, (args.population, 12)), lo, hi)
            pop[0] = mean
            trials = [(g, "sit", "carpet" if k % 2 else "hard", 1800 + k, 2.2) for g in pop]
            outcomes = pool.map(evaluate_case, trials)
            scores = np.array([1.05 * r - 0.2 * np.linalg.norm(g) for (r, _), g in zip(outcomes, pop)])
            elite = scores.argsort()[-max(3, round(args.population * 0.115)) :]
            mean = pop[elite].mean(0)
            std = np.maximum(0.08, pop[elite].std(0) * 1.2)
            history.extend([(float(scores[e]), pop[e].tolist()) for e in elite[-3:]])
            print(
                f"Generation {k + 1}: {sum(x[1] for x in outcomes)}/{len(outcomes)} rose; best {scores[elite[-1]]:.2f}",
                flush=True,
            )
        choices = [v for _, v in sorted(history, reverse=True)[:8]]
        knots, successes, trials = select(pool, choices, [2000, 2001])
    error = export(ROOT / "runtime/xgoduck_sitstand.onnx", knots, args.out / "rise.onnx")
    output = dict(
        knots=knots,
        seed=args.seed,
        validation_successes=successes,
        validation_trials=trials,
        bounded_search=args.joint_bounds,
        max_export_error=error,
    )
    (args.out / "rise.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()

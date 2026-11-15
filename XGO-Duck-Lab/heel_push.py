"""Paired backward-push benchmark, starting upright on the same synthetic carpet."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from duck_env import DuckSitStandEnv, ROOT
from stock_policy import StockPolicy
from v2_feet import VARIANTS

IMPULSES = (0.0, 0.10, 0.22, 0.35)  # N s / kg = m/s; not imposed velocities
SETTLE_S, PUSH_S, AFTER_S = 2.0, 0.10, 3.0
READY_S, SUCCESS_S = 0.4, 1.0


def episode(env, actor, seed, specific_impulse):
    _, initial = env.reset(seed=seed)
    data = env.data
    actor.reset(data)
    dt = env.dt * env.decimation
    settle_steps, push_steps = round(SETTLE_S / dt), round(PUSH_S / dt)
    mass = float(env.model.body_subtreemass[env.trunk])
    # xfrc_applied is a Cartesian wrench in world coordinates, at the body COM.
    wrench = data.xfrc_applied[env.trunk]
    wrench[:] = 0
    force = specific_impulse * mass / PUSH_S
    eligible = False
    peak_tilt = 0.0
    post_steps = 0
    pre_tilt = None
    impulse_applied = 0.0
    for step in range(env.max_steps):
        if step == settle_steps:
            pre_tilt = info["tilt_deg"]
            if info["upright_streak_s"] + 1e-9 < READY_S:
                break
            eligible = True
            heading = data.xmat[env.trunk].reshape(3, 3)[:2, 0].copy()
            wrench[:2] = -force * heading / np.linalg.norm(heading)
        if step == settle_steps + push_steps:
            wrench[:] = 0
        target = actor.targets()
        _, _, fell, timeout, info = env.advance(target, actor.raw_action, sample_callback=actor.sample)
        if step >= settle_steps:
            post_steps += 1
            peak_tilt = max(peak_tilt, info["tilt_deg"])
            if step < settle_steps + push_steps:
                impulse_applied += force * dt
        if fell or timeout:
            break
    wrench[:] = 0
    if not eligible:
        outcome = "fell_before_push" if info["fell"] else "unsettled_before_push"
    elif info["fell"]:
        outcome = "fell_after_push"
    elif info["upright_streak_s"] + 1e-9 >= SUCCESS_S:
        outcome = "recovered"
    else:
        outcome = "not_upright_at_end"
    return dict(
        seed=seed,
        input=initial,
        eligible=eligible,
        outcome=outcome,
        final_upright=bool(info["upright"]),
        final_upright_s=round(info["upright_streak_s"], 3),
        elapsed_s=round(float(data.time), 3),
        issue_force_N=round(force, 5),
        delivered_impulse_Ns=round(impulse_applied, 5),
        pre_tilt_deg=pre_tilt,
        post_peak_tilt_deg=round(peak_tilt, 3) if post_steps else None,
    )


def benchmark(
    policy,
    seeds,
    mass_delta=True,
    impulses=IMPULSES,
    candidates=("v2-heel15",),
    policy_file=None,
    scale=1.0,
):
    seeds = list(seeds)
    path = Path(policy_file) if policy_file else ROOT / "runtime" / f"xgoduck_{policy}.onnx"
    rows = []
    for foot in ("v2-stock", *candidates):
        env = DuckSitStandEnv(
            foot=foot,
            surface="carpet",
            start="stand",
            episode_s=SETTLE_S + PUSH_S + AFTER_S,
            mass_delta=mass_delta,
        )
        actor = StockPolicy(env.model, ROOT / "runtime", path, scale=scale)
        try:
            for impulse in impulses:
                cases = [episode(env, actor, seed, impulse) for seed in seeds]
                counts = {}
                for case in cases:
                    outcome = case["outcome"]
                    counts[outcome] = counts.get(outcome, 0) + 1
                rows.append(dict(foot=foot, impulse_per_mass_m_s=impulse, counts=counts, trials=cases))
                print(foot, impulse, counts, flush=True)
        finally:
            env.close()
    paired = []
    for candidate in candidates:
        for impulse in impulses:
            a, b = [
                r["trials"]
                for r in rows
                if r["impulse_per_mass_m_s"] == impulse and r["foot"] in ("v2-stock", candidate)
            ]
            common = [(x, y) for x, y in zip(a, b) if x["eligible"] and y["eligible"]]
            assert all(x["seed"] == y["seed"] and x["input"] == y["input"] for x, y in zip(a, b))
            paired.append(
                dict(
                    candidate=candidate,
                    impulse_per_mass_m_s=impulse,
                    both_eligible=len(common),
                    both_recovered=sum(x["outcome"] == y["outcome"] == "recovered" for x, y in common),
                    stock_only_recovered=sum(
                        x["outcome"] == "recovered" and y["outcome"] != "recovered" for x, y in common
                    ),
                    heel_only_recovered=sum(
                        y["outcome"] == "recovered" and x["outcome"] != "recovered" for x, y in common
                    ),
                    neither_recovered=sum(
                        x["outcome"] != "recovered" and y["outcome"] != "recovered" for x, y in common
                    ),
                )
            )
    return dict(
        candidates=list(candidates),
        policy=policy,
        policy_file=path.name,
        scale=scale,
        policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seeds=seeds,
        surface="carpet",
        foot_added_mass_estimate=mass_delta,
        settle_s=SETTLE_S,
        pulse_s=PUSH_S,
        after_s=AFTER_S,
        protocol="Zero command. At t=2 s screen for >=0.4 s continuously upright. Apply a constant force backward along initial push-time trunk heading at trunk COM for 0.10 s. Recover when no fall and >=1 s continuously upright at t=5.10 s.",
        thresholds="upright: z>=0.124 m, tilt<15 deg, both knees <0.40 rad magnitude; fall: z<0.052 m or tilt>65 deg",
        groups=rows,
        paired=paired,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("walk", "sitstand"), default="walk")
    parser.add_argument(
        "--candidate",
        nargs="+",
        choices=[v for v in VARIANTS if v.startswith("v2-heel")],
        default=["v2-heel15"],
    )
    parser.add_argument("--stock-only", action="store_true", help="Test the original V2 feet alone.")
    parser.add_argument(
        "--policy-file", type=Path, help="Use this 61-input, 14-output ONNX for the complete trial."
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Runtime action multiplier.")
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--impulses",
        nargs="+",
        type=float,
        default=IMPULSES,
        help="Pulse impulse divided by total robot mass in m/s. Include zero as a control.",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("At least one trial is required.")
    if len(set(args.candidate)) != len(args.candidate):
        parser.error("Use distinct candidate feet.")
    if any(not np.isfinite(v) or v < 0 for v in args.impulses) or len(set(args.impulses)) != len(
        args.impulses
    ):
        parser.error("Use distinct nonnegative finite impulse levels.")
    result = benchmark(
        args.policy,
        range(args.seed, args.seed + args.trials),
        not args.geometry_only,
        args.impulses,
        () if args.stock_only else args.candidate,
        args.policy_file,
        args.scale,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

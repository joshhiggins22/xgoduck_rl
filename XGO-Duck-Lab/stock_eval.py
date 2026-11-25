"""Headless, paired-seed stock ONNX evaluation on the training environment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from duck_env import DuckSitStandEnv, FEET, ROOT
from stock_policy import StockPolicy


def evaluate(
    *,
    foot,
    surface,
    start,
    seeds,
    duration_s,
    model_name="sitstand",
    handoff_s=None,
    scale=1.0,
    sit_dwell=0.0,
    pose_handoff=False,
    sitstand_onnx=None,
    walk_onnx=None,
    rise_push=0.0,
):
    """Run a fixed command from reset; optionally switch SitStand -> Walk on a timer."""
    if duration_s <= 0 or (
        handoff_s is not None and (model_name != "sitstand" or not 0 <= handoff_s < duration_s)
    ):
        raise ValueError("A handoff applies to sitstand and must occur within the episode.")
    if sit_dwell < 0 or (sit_dwell > 0 and (start != "stand" or model_name != "sitstand")):
        raise ValueError("A sit dwell starts from stand with the SitStand policy.")
    if pose_handoff and (handoff_s is not None or model_name != "sitstand"):
        raise ValueError("Pose-based handoff applies to sitstand and cannot be combined with a timer.")
    if not np.isfinite(rise_push) or rise_push < 0:
        raise ValueError("Rise push must be a nonnegative finite impulse per unit mass.")
    if rise_push and duration_s < 0.30:
        raise ValueError("The rise push runs from 0.20 to 0.30 seconds after Stand.")
    env = DuckSitStandEnv(foot=foot, surface=surface, start=start, episode_s=duration_s + sit_dwell)
    runtime = ROOT / "runtime"

    def model_path(name):
        if name == "sitstand" and sitstand_onnx is not None:
            return Path(sitstand_onnx)
        if name == "walk" and walk_onnx is not None:
            return Path(walk_onnx)
        return runtime / f"xgoduck_{name}.onnx"

    def load(name, gain):
        return StockPolicy(env.model, runtime, model_path(name), scale=gain)

    first = load(model_name, scale)
    walk = load("walk", 1.0) if handoff_s is not None or pose_handoff else None
    rows = []
    try:
        for seed in seeds:
            _, reset = env.reset(seed=int(seed))
            actor = first
            actor.reset(env.data)
            if walk is not None:
                walk.policy.reset_action_filter()
                # The IMU remains live across the controller's change of policy.
                walk.feedback = first.feedback
            switched = timed_out = False
            ready_ticks = 0
            switched_at = None
            info = {}
            max_streak = 0.0
            max_raw = 0.0
            first_stable = None
            seated_streak = 0
            seat_confirmed = None if sit_dwell == 0 else False
            rise_streak = 0.0
            dt = env.dt * env.decimation
            wrench = env.data.xfrc_applied[env.trunk]
            wrench[:] = 0
            pushed_ticks = 0
            while True:
                if rise_push:
                    stand_tick = round((env.data.time - sit_dwell) / dt)
                    if stand_tick == round(0.20 / dt):
                        heading = env.data.xmat[env.trunk].reshape(3, 3)[:2, 0]
                        force = rise_push * env.model.body_subtreemass[env.trunk] / 0.10
                        wrench[:2] = -force * heading / np.linalg.norm(heading)
                    if stand_tick == round(0.30 / dt):
                        wrench[:] = 0
                    if np.any(wrench[:2]):
                        pushed_ticks += 1
                sitting = env.data.time + 1e-9 < sit_dwell
                if walk is not None and not switched and not sitting:
                    elapsed = float(env.data.time) - sit_dwell
                    if pose_handoff:
                        if elapsed + 1e-9 >= 3.0:
                            timed_out = True
                            break
                        st = actor.feedback.state
                        grav = np.asarray(st.grav)
                        tilt = np.degrees(np.arccos(np.clip(-grav[2] / np.linalg.norm(grav), -1, 1)))
                        ids = actor.feedback.core.MOTOR_IDS
                        knees = [abs(st.angles[ids.index(i)]) for i in (13, 23)]
                        ready_ticks = ready_ticks + 1 if tilt <= 15 and max(knees) < 25 else 0
                        ready = ready_ticks >= 3
                    else:
                        ready = elapsed + 1e-9 >= handoff_s
                    if ready:
                        actor = walk
                        switched = True
                        switched_at = round(elapsed, 3)
                target = actor.targets(command=[1.0, 0.0, 0.0] if sitting else None)
                raw = actor.raw_action
                max_raw = max(max_raw, float(np.abs(raw).max()))
                _, _, terminated, truncated, info = env.advance(
                    target, raw, sample_callback=actor.sample
                )
                if sitting:
                    knees = np.abs(env.data.qpos[env.qpos_adr[[3, 12]]])
                    seated = info["z"] < 0.115 and info["tilt_deg"] < 25 and min(knees) > 0.8
                    seated_streak = seated_streak + 1 if seated else 0
                    seat_confirmed = bool(seated_streak * dt >= 0.24)
                else:
                    rise_streak = round(rise_streak + dt, 5) if info["upright"] else 0.0
                    if first_stable is None and info["upright"]:
                        first_stable = round(float(env.data.time) - sit_dwell, 4)
                    max_streak = max(max_streak, rise_streak)
                if terminated or truncated:
                    break
            rows.append(
                dict(
                    seed=int(seed),
                    start=reset["start"],
                    surface=reset["surface"],
                    seconds=round(float(env.data.time), 4),
                    fell=info["fell"],
                    push_applied_s=round(pushed_ticks * dt, 3),
                    first_upright_s=first_stable,
                    longest_upright_s=round(max_streak, 3),
                    final_upright_s=round(rise_streak, 3),
                    seated_at_command=seat_confirmed,
                    end_height_m=round(info["z"], 5),
                    end_tilt_deg=round(info["tilt_deg"], 2),
                    max_abs_raw_action=round(max_raw, 5),
                    switched_to_walk=switched,
                    switch_after_stand_s=switched_at,
                    handoff_timed_out=timed_out,
                    succeeded=bool(
                        not timed_out
                        and not info["fell"]
                        and rise_streak >= 1
                        and seat_confirmed is not False
                    ),
                )
            )
    finally:
        env.close()
    names = {model_name, "walk"} if walk is not None else {model_name}
    hashes = {n: hashlib.sha256(model_path(n).read_bytes()).hexdigest() for n in names}
    return dict(
        foot=foot,
        surface=surface,
        start=start,
        rise_push_Ns_per_kg=rise_push,
        duration_s=duration_s,
        policy=model_name,
        sitstand_override=str(sitstand_onnx) if sitstand_onnx else None,
        scale=scale,
        handoff_s=handoff_s,
        pose_handoff=pose_handoff,
        sit_dwell_s=sit_dwell,
        sha256=hashes,
        seat_definition="when dwelling, both knees >0.8 rad, z<0.115 m and tilt<25 deg over final 0.24 s of sit command",
        success_definition="upright for final >=1 s and no fall before timeout",
        successes=sum(r["succeeded"] for r in rows),
        trials=len(rows),
        episodes=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foot", choices=FEET, default="upstream")
    p.add_argument("--surface", choices=["hard", "carpet", "mixed", "hardwood", "residential-carpet"], default="hard")
    p.add_argument("--start", choices=["stand", "sit", "mixed"], default="sit")
    p.add_argument("--policy", choices=["sitstand", "walk"], default="sitstand")
    p.add_argument(
        "--sitstand-onnx",
        type=Path,
        help="Optional single-input replacement actor, used for both Sit and Stand.",
    )
    p.add_argument(
        "--sit-dwell",
        type=float,
        default=0.0,
        help="Command Sit for this many seconds before Stand; use --start stand.",
    )
    p.add_argument("--handoff-seconds", type=float, help="Optional timed SitStand to Walk handoff.")
    p.add_argument(
        "--pose-handoff",
        action="store_true",
        help="Switch after three 50 Hz checks with tilt ≤15° and both knees <25°; stop on a 3 s timeout.",
    )
    p.add_argument(
        "--walk-onnx", type=Path, help="Optional Walk actor used after handoff or during a Walk trial."
    )
    p.add_argument(
        "--rise-push",
        type=float,
        default=0.0,
        help="Backward torso impulse in N·s/kg over 0.10 s, starting 0.20 s after Stand.",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Initial policy action multiplier.")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    if args.trials < 1:
        p.error("At least one trial is required.")
    result = evaluate(
        foot=args.foot,
        surface=args.surface,
        start=args.start,
        duration_s=args.seconds,
        seeds=range(args.seed, args.seed + args.trials),
        model_name=args.policy,
        handoff_s=args.handoff_seconds,
        scale=args.scale,
        sit_dwell=args.sit_dwell,
        pose_handoff=args.pose_handoff,
        sitstand_onnx=args.sitstand_onnx,
        walk_onnx=args.walk_onnx,
        rise_push=args.rise_push,
    )
    payload = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

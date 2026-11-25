"""Command Sit, Stand and optional Walk on one continuous simulated heel-equipped duck."""

import argparse
import json
from pathlib import Path
import numpy as np
from duck_env import DuckSitStandEnv, ROOT
from rise import load_policy
from learner import onnx_session, predict_onnx


def evaluate(policy, surface, seeds, handoff):
    env = DuckSitStandEnv("v2-heel15", surface, "stand", 9.0)
    rise = load_policy(policy)
    factory = onnx_session(ROOT / "runtime/xgoduck_sitstand.onnx")
    walk = onnx_session(ROOT / "runtime/xgoduck_walk.onnx")
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        seated_ticks = ready = 0
        seated = switched = expired = False
        before_stand = None
        switched_at = None
        for tick in range(env.max_steps):
            if tick < 150:
                obs = env.feedback.obs(env._last_action, cmd=[1.0, 0.0, 0.0])
                action = predict_onnx(factory, obs)
            else:
                if tick == 150:
                    seated = seated_ticks >= 12
                    before_stand = dict(
                        z_m=round(info["z"], 4),
                        tilt_deg=round(info["tilt_deg"], 2),
                        knee_rad=np.round(env.data.qpos[env.qpos_adr[[3, 12]]], 3).tolist(),
                    )
                    obs = env.feedback.obs(env._last_action)
                    rise.reset(obs)
                elapsed = (tick - 150) * 0.02
                if handoff == "pose" and not switched:
                    st = env.feedback.state
                    gravity = np.asarray(st.grav)
                    tilt = np.rad2deg(np.arccos(np.clip(-gravity[2] / np.linalg.norm(gravity), -1, 1)))
                    knees = [abs(st.angles[env.feedback.core.MOTOR_IDS.index(i)]) for i in (13, 23)]
                    ready = ready + 1 if tilt <= 15.0 and max(knees) < 25.0 else 0
                    if elapsed >= 3.0:
                        expired = True
                        break
                if not switched and (
                    (handoff == "timer" and elapsed >= 1.0) or (handoff == "pose" and ready >= 3)
                ):
                    switched = True
                    switched_at = elapsed
                    env.actions.reset()
                    obs = env.feedback.obs(np.zeros(14, dtype=np.float32))
                action = predict_onnx(walk, obs) if switched else rise(obs)
            obs, _, term, trunc, info = env.step(action)
            if tick < 150:
                kk = np.abs(env.data.qpos[env.qpos_adr[[3, 12]]])
                seated_ticks = (
                    seated_ticks + 1
                    if info["z"] < 0.115 and info["tilt_deg"] < 25 and min(kk) > 0.8
                    else 0
                )
            if term or trunc:
                break
        rows.append(
            dict(
                seed=seed,
                confirmed_seated=seated,
                before_stand=before_stand,
                fell=info["fell"],
                timed_out=expired,
                handoff_s=switched_at,
                final_upright_s=min(info["upright_streak_s"], max(0.0, (tick + 1 - 150) * 0.02)),
                success=bool(
                    seated and not expired and not info["fell"] and info["upright_streak_s"] >= 1.0
                ),
            )
        )
    return dict(
        policy=str(policy.name),
        surface=surface,
        handoff=handoff,
        foot="v2-heel15",
        sit_s=3.0,
        rise_s=6.0,
        successes=sum(r["success"] for r in rows),
        episodes=len(rows),
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=ROOT / "runs/heel-rise-bounded/rise.onnx")
    p.add_argument("--surface", choices=["hard", "carpet", "hardwood", "residential-carpet"], default="carpet")
    p.add_argument("--handoff", choices=["none", "timer", "pose"], default="none")
    p.add_argument("--seed", type=int, default=2610)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = evaluate(
        args.policy, args.surface, range(args.seed, args.seed + args.episodes), args.handoff
    )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['successes']}/{result['episodes']} passed; {sum(r['confirmed_seated'] for r in result['rows'])} seated before the Stand command."
    )

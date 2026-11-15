"""Train and replay a bounded backward-balance correction to the stock V2 Walk actor."""

import argparse
import hashlib
import json
import multiprocessing
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from duck_env import ROOT, DuckSitStandEnv
from learner import onnx_session, predict_onnx

WALK = ROOT / "runtime/walk-4ce3315c.onnx"
STAND = ROOT / "runtime/xgoduck_sitstand.onnx"
DEFAULT_RUN = ROOT / "runs/stock-walk-recovery"
LEVELS = (0.0, 0.35, 0.55, 0.75, 0.90, 1.0)
LOW = np.array([-0.38, -0.38, -0.38, -0.38, 0.06, 0.045, 2.0])
HIGH = np.array([0.38, 0.38, 0.38, 0.38, 0.25, 0.12, 8.0])


def vector(parameters):
    hip, knee, ankle, neck = parameters[:4]
    return np.array(
        [
            0.0,
            0.0,
            hip,
            knee,
            ankle,
            neck,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -hip,
            -knee,
            -ankle,
        ],
        np.float32,
    )


def activation(obs, parameters):
    horizon, margin, gain = parameters[4:]
    if np.any(obs[48:51] != 0):
        return 0.0
    return np.clip((-obs[3] - horizon * obs[1] - margin) * gain, 0.0, 1.0)


def export(parameters, path, base=WALK):
    model = onnx.load(base)
    if (
        [i.name for i in model.graph.input] != ["obs"]
        or [o.name for o in model.graph.output] != ["actions"]
        or model.graph.node[-1].output[0] != "actions"
        or model.opset_import[0].version != 18
    ):
        raise ValueError(
            "Expected a stock XGO Walk output graph with opset 18 and obs/actions ports."
        )
    graph = model.graph
    graph.node[-1].output[0] = "factory_walk"

    def constant(name, value, dtype=np.float32):
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        )
        return name

    def node(op, args, dest, **attrs):
        graph.node.append(helper.make_node(op, args, [dest], **attrs))
        return dest

    rate = node(
        "Gather", ["obs", constant("rate_index", [1], np.int64)], "rate_y", axis=1
    )
    grav = node(
        "Gather", ["obs", constant("grav_index", [3], np.int64)], "gravity_x", axis=1
    )
    command = node(
        "Gather",
        ["obs", constant("command_indices", [48, 49, 50], np.int64)],
        "movement",
        axis=1,
    )
    horizon = node("Mul", [rate, constant("horizon", parameters[4])], "rate_term")
    projected = node("Add", [grav, horizon], "projected")
    negative = node("Neg", [projected], "backward")
    excess = node("Sub", [negative, constant("margin", parameters[5])], "excess")
    scaled = node("Mul", [excess, constant("gain", parameters[6])], "scaled")
    zero = constant("zero", 0.0)
    gate = node("Clip", [scaled, zero, constant("one", 1.0)], "balance_gate")
    maximum = node(
        "ReduceMax",
        [
            node("Abs", [command], "movement_abs"),
            constant("movement_axis", [1], np.int64),
        ],
        "max_command",
        keepdims=1,
    )
    stationary = node("Equal", [maximum, zero], "stationary")
    scale = node("Where", [stationary, gate, zero], "balance_scale")
    delta = node(
        "Mul", [scale, constant("max_joint_delta", vector(parameters))], "balance_delta"
    )
    node("Add", ["factory_walk", delta], "actions")
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def check_export(base, path, params):
    before, after = onnx_session(base), onnx_session(path)
    rng = np.random.default_rng(9102)
    exact_inactive = exact_command = 0
    max_error = max_offset = 0.0
    for i in range(450):
        obs = rng.normal(0.0, 0.4, 61).astype(np.float32)
        obs[48:51] = 0.0
        if i % 3 == 0:
            obs[48 + (i // 3) % 3] = np.float32(1e-7)
        if i % 3 == 1:
            obs[1] = 0.0
            obs[3] = 0.015
        a = predict_onnx(before, obs)
        b = predict_onnx(after, obs)
        gate = activation(obs, params)
        expected = a + gate * vector(params)
        max_error = max(max_error, float(np.max(np.abs(b - expected))))
        max_offset = max(max_offset, float(np.max(np.abs(b - a))))
        if i % 3 == 0:
            assert np.array_equal(a, b)
            exact_command += 1
        elif i % 3 == 1:
            assert np.array_equal(a, b)
            exact_inactive += 1
    assert max_error < 2.0e-6
    return {
        "observations": 450,
        "exact_nonzero_command": exact_command,
        "exact_benign": exact_inactive,
        "max_formula_abs_error": max_error,
        "max_observed_action_delta_rad": max_offset,
    }


def initialize(exported=None, base=WALK, full_cycle=False):
    global env, factory, stand, candidate, commanded_sit
    commanded_sit = full_cycle
    env = DuckSitStandEnv("v2-stock", "carpet", "stand", 11.1 if commanded_sit else 5.1)
    factory = onnx_session(base)
    stand = onnx_session(STAND)
    candidate = onnx_session(exported) if exported else None


def rollout(case):
    params, seed, impulse, *modes = case
    continuous = bool(modes and modes[0] == "continuous")
    obs, contact = env.reset(seed=seed)
    wrench = env.data.xfrc_applied[env.trunk]
    eligible = False
    upright_s = 0.0
    peak = 0.0
    max_gate = 0.0
    ready_ticks = seated_ticks = 0
    seated = None
    switched = None if commanded_sit else 0 if continuous else 50
    push_at = None if commanded_sit else 100
    for tick in range(env.max_steps):
        if commanded_sit and tick == 150:
            seated = seated_ticks >= 12
            if not seated:
                break
        if commanded_sit and tick >= 150 and switched is None:
            if tick >= 300:
                break
            state = env.feedback.state
            gravity = np.asarray(state.grav)
            tilt = np.rad2deg(
                np.arccos(np.clip(-gravity[2] / np.linalg.norm(gravity), -1.0, 1.0))
            )
            knees = [
                abs(state.angles[env.feedback.core.MOTOR_IDS.index(i)])
                for i in (13, 23)
            ]
            ready_ticks = ready_ticks + 1 if tilt <= 15.0 and max(knees) < 25.0 else 0
            if ready_ticks >= 3:
                switched = tick
                push_at = tick + 50
        if tick == switched:
            env.actions.reset()
            obs = env.feedback.obs(np.zeros(14, np.float32))
        if tick == push_at:
            if upright_s < 0.4:
                break
            eligible = True
            heading = env.data.xmat[env.trunk].reshape(3, 3)[:2, 0]
            wrench[:2] = (
                -impulse
                / 0.10
                * env.model.body_subtreemass[env.trunk]
                * heading
                / np.linalg.norm(heading)
            )
        if push_at is not None and tick == push_at + 5:
            wrench[:] = 0.0
        if switched is None or tick < switched:
            if commanded_sit and tick < 150:
                obs = env.feedback.obs(env._last_action, cmd=[1.0, 0.0, 0.0])
            action = predict_onnx(stand, obs)
        elif isinstance(params, str):
            action = predict_onnx(candidate, obs)
        else:
            action = predict_onnx(factory, obs)
            if params is not None:
                gate = activation(obs, params)
                action = action + gate * vector(params)
                max_gate = max(gate, max_gate)
        obs, _, done, _, info = env.step(action)
        upright_s = info["upright_streak_s"]
        if commanded_sit and tick < 150:
            knees = np.abs(env.data.qpos[env.qpos_adr[[3, 12]]])
            is_seated = (
                info["z"] < 0.115 and info["tilt_deg"] < 25.0 and min(knees) > 0.8
            )
            seated_ticks = seated_ticks + 1 if is_seated else 0
        if push_at is not None and tick >= push_at:
            peak = max(peak, info["tilt_deg"])
        if done or (push_at is not None and tick == push_at + 154):
            break
    wrench[:] = 0.0
    outcome = (
        "ineligible"
        if not eligible
        else "fell"
        if info["fell"]
        else "recovered"
        if info["upright_streak_s"] >= 1.0
        else "not_upright"
    )
    # Rank the discrete objective first. Break ties using attenuation and length of the upright hold.
    score = (
        10.0 * (outcome == "recovered")
        + 2.0 * (eligible and not info["fell"])
        + min(info["upright_streak_s"], 3.0) / 3.0
        - peak / 90.0
    )
    return {
        "seed": seed,
        "startup": "cycle"
        if commanded_sit
        else "continuous"
        if continuous
        else "handoff",
        "seated": seated,
        "switch_s": None if switched is None else switched * 0.02,
        "specific_impulse": impulse,
        "outcome": outcome,
        "fall": info["fell"],
        "peak_tilt_deg": peak,
        "elapsed_s": (tick + 1) * 0.02 if eligible else round(float(env.data.time), 5),
        "final_upright_s": info["upright_streak_s"],
        "max_gate": float(max_gate)
        if params is not None and not isinstance(params, str)
        else None,
        "contact": contact,
        "score": score,
    }


def fit(out, base):
    rng = np.random.default_rng(5571)
    mean = (HIGH + LOW) / 2
    sigma = (HIGH - LOW) * 0.39
    finalists = []
    with multiprocessing.get_context("spawn").Pool(
        2, initializer=initialize, initargs=(None, base)
    ) as pool:
        for generation, (impulse, seed) in enumerate(
            zip(
                [0.38, 0.45, 0.43, 0.5, 0.45, 0.5, 0.40],
                [6100, 6101, 6102, 6103, 6104, 6105, 6106],
            )
        ):
            pop = np.clip(rng.normal(mean, sigma, (26, 7)), LOW, HIGH)
            pop[0] = mean
            rows = pool.map(rollout, [(p, seed, impulse) for p in pop])
            elite = np.argsort([r["score"] for r in rows])[-6:]
            mean = pop[elite].mean(0)
            sigma = np.maximum((HIGH - LOW) * 0.035, pop[elite].std(0) * 1.15)
            finalists.append(pop[elite[-1]].tolist())
            print(
                "Generation",
                generation + 1,
                Counter(r["outcome"] for r in rows),
                flush=True,
            )
        validation = []
        for p in finalists:
            cases = [
                (p, seed, level)
                for seed in (6200, 6201)
                for level in (0.0, 0.22, 0.35, 0.45, 0.55)
            ]
            rows = pool.map(rollout, cases)
            validation.append(
                (
                    sum(r["outcome"] == "recovered" for r in rows),
                    sum(r["score"] for r in rows),
                    p,
                    rows,
                )
            )
        wins, _, params, rows = max(validation, key=lambda x: x[:2])
    export(params, out / "walk.onnx", base)
    (out / "parameters.json").write_text(
        json.dumps(
            {
                "params": params,
                "export_check": check_export(base, out / "walk.onnx", params),
                "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "validation": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("Validation", wins, "of", len(rows), params, flush=True)


def fit_stress(out, base):
    rng = np.random.default_rng(5659)
    mean = (HIGH + LOW) / 2
    sigma = (HIGH - LOW) * 0.47
    finalists = []
    with multiprocessing.get_context("spawn").Pool(
        2, initializer=initialize, initargs=(None, base)
    ) as pool:
        for generation, (level_a, level_b) in enumerate(
            [
                (0.7, 0.75),
                (0.85, 0.9),
                (0.8, 0.8),
                (0.95, 0.85),
                (1.0, 0.95),
                (0.75, 0.85),
                (0.90, 1.0),
            ]
        ):
            pop = np.clip(rng.normal(mean, sigma, (30, 7)), LOW, HIGH)
            pop[0] = mean
            cases = [(p, 8100 + generation * 2, level_a, "continuous") for p in pop] + [
                (p, 8101 + generation * 2, level_b, "handoff") for p in pop
            ]
            rows = pool.map(rollout, cases)
            pairs = zip(rows[: len(pop)], rows[len(pop) :])
            scores = [
                a["score"]
                + b["score"]
                + 0.25 * (min(a["elapsed_s"], 4.0) + min(b["elapsed_s"], 4.0))
                for a, b in pairs
            ]
            elite = np.argsort(scores)[-6:]
            mean = pop[elite].mean(0)
            sigma = np.maximum((HIGH - LOW) * 0.055, pop[elite].std(0) * 1.12)
            finalists.append(pop[elite[-1]].tolist())
            print(
                "Stress generation",
                generation + 1,
                Counter(r["outcome"] for r in rows),
                flush=True,
            )
        validation = []
        # Identical disturbances across all finalists, including zero offsets.
        finalists.append([0.0, 0.0, 0.0, 0.0, 0.17, 0.07, 5.0])
        for p in finalists:
            cases = [
                (p, seed, level, mode)
                for mode in ("continuous", "handoff")
                for seed, level in [
                    (8250, 0.55),
                    (8251, 0.75),
                    (8252, 0.9),
                    (8253, 1.0),
                    (8254, 0.85),
                    (8255, 0.95),
                ]
            ]
            rows = pool.map(rollout, cases)
            passed = [
                sum(r["outcome"] == "recovered" for r in rows[k : k + 6])
                for k in (0, 6)
            ]
            key = (min(passed), sum(passed), sum(r["score"] for r in rows))
            validation.append((key, p, rows))
        rank, params, rows = max(validation, key=lambda v: v[0])
    export(params, out / "walk.onnx", base)
    (out / "parameters.json").write_text(
        json.dumps(
            {
                "params": params,
                "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "export_check": check_export(base, out / "walk.onnx", params),
                "validation": rows,
                "validation_rank": rank,
                "validation_identity_rank": validation[-1][0],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Stress validated", rank, "vs identity", validation[-1][0], params, flush=True
    )


def compare(path, seeds, out, base, levels=LEVELS, full_cycle=False, continuous=False):
    cases = [
        (policy, seed, level, "continuous" if continuous else "handoff")
        for policy in (None, "export")
        for level in levels
        for seed in seeds
    ]
    with multiprocessing.get_context("spawn").Pool(
        2, initializer=initialize, initargs=(path, base, full_cycle)
    ) as pool:
        rows = pool.map(rollout, cases)
    groups = []
    for level in levels:
        pair = []
        for policy, label in ((None, "factory"), ("export", "candidate")):
            subset = [
                r for c, r in zip(cases, rows) if c[0] == policy and c[2] == level
            ]
            pair.append(subset)
            print(level, label, Counter(r["outcome"] for r in subset), flush=True)
        counts = Counter()
        for a, b in zip(*pair):
            assert a["contact"] == b["contact"] and a["seed"] == b["seed"]
            counts[a["outcome"] + "->" + b["outcome"]] += 1
        groups.append(
            {
                "impulse": level,
                "paired_outcomes": counts,
                "factory": pair[0],
                "candidate": pair[1],
            }
        )
    report = {
        "full_cycle": full_cycle,
        "continuous_walk": continuous,
        "protocol": (
            "Three-second factory Sit, require a measured deep sit during the final 0.24 s. Stand until three 50 Hz samples show upright filtered gravity and both knees below 25 deg, aborting after 3 s. Reset Walk action filters; retain physics and IMU. One second after switching, check 0.4 s continuously upright then push for 0.10 s and assess recovery after 3 s, as in the upright-start benchmark."
            if full_cycle
            else "Continuous Walk from standing reset. Same stock V2, randomized carpet, zero command, t=2 s push, force and stability criteria as the handoff benchmark."
            if continuous
            else "Stock V2 on randomized carpet. Stand under factory SitStand for 1 s; switch to Walk with fresh Walk action filters and retained IMU/physics at 50 Hz. At t=2 s require 0.4 s continuously upright then apply backward force along trunk heading for 0.10 s at COM. Count recovery if never fallen and continuously upright for the final 1 s of 5.10 s. Zero velocity command."
        ),
        "factory_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "groups": groups,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "fit-stress", "evaluate"])
    parser.add_argument("--factory", type=Path, default=WALK)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--full-cycle", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--impulses", nargs="+", type=float, default=LEVELS)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/stock-walk-recovery-held-out.json"
    )
    args = parser.parse_args()
    if args.run is None:
        args.run = (
            ROOT / "runs/stock-walk-recovery-stress"
            if args.mode == "fit-stress"
            else DEFAULT_RUN
        )
    if args.continuous and args.full_cycle:
        parser.error(
            "--continuous and --full-cycle select different initial conditions"
        )
    if args.episodes < 1 or any(not np.isfinite(x) or x < 0 for x in args.impulses):
        parser.error("Use a positive episode count and finite, nonnegative impulses.")
    if args.mode == "fit":
        fit(args.run, args.factory)
    elif args.mode == "fit-stress":
        fit_stress(args.run, args.factory)
    else:
        compare(
            args.run / "walk.onnx",
            range(args.seed, args.seed + args.episodes),
            args.out,
            args.factory,
            args.impulses,
            args.full_cycle,
            args.continuous,
        )

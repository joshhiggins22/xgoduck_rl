"""Paired hardware-variation and abrupt-release trials for frozen heel profiles."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from statistics import fmean
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_walk import THREAD_VARS, verify_model_file, write_json


HEELS = {"v2-stock":"factory", "v2-heel5":"5mm", "v2-heel10":"10mm"}
FLOORS = ("hardwood","residential-carpet")


def mismatch(seed, trial, *, weak_leg=None, extra_delay_ms=None):
    import numpy as np
    rng = np.random.default_rng(seed + 2_100_000)
    # Prior fit scripts can still replay their original indexed perturbations.
    if weak_leg is None:
        weak_leg = (trial + trial//4) % 2
    if extra_delay_ms is None:
        extra_delay_ms = 5 if (trial//2 + trial//4)%2 == 0 else 0
    sides = rng.uniform(.98,1.04,2)
    sides[weak_leg] = rng.uniform(.87,.95)
    gains = np.ones(14)
    zero = np.zeros(14)
    for i, leg in enumerate((slice(0,5),slice(9,14))):
        gains[leg] = sides[i] + rng.uniform(-.02,.02,5)
        zero[leg] = rng.uniform(-.7,.7,5)
    return dict(weak_leg=("left","right")[weak_leg], extra_delay_ms=extra_delay_ms,
                torque_gain=gains.tolist(), encoder_zero_deg=zero.tolist(),
                imu_tilt_deg=rng.uniform(-1.5,1.5,2).tolist(),
                gyro_bias_dps=rng.uniform([-.3,-.3,-.6],[.3,.3,.6]).tolist())


@contextmanager
def apply_mismatch(plant, spec):
    import numpy as np
    actuator = plant.ctrl.model.actuator
    old_control, old_torque = actuator.compute_control, actuator.compute_torque
    old_feedback = plant.feedback._process
    old_filter = plant.feedback._filter.update
    gain = np.array(spec["torque_gain"])
    zero_deg = np.array(spec["encoder_zero_deg"])
    zero_rad = np.deg2rad(zero_deg)
    gyro_bias = np.array(spec["gyro_bias_dps"])
    r, p = np.deg2rad(spec["imu_tilt_deg"])
    cr,sr,cp,sp = math.cos(r),math.sin(r),math.cos(p),math.sin(p)
    rotation = np.array([[cp,sp*sr,sp*cr],[0.,cr,-sr],[-sp,cp*sr,cp*cr]])
    delayed = slice(0,5) if spec["weak_leg"] == "left" else slice(9,14)
    last_target = None

    def control(target, q, dq, dt):
        nonlocal last_target
        use = target
        if spec["extra_delay_ms"]:
            use = target.copy()
            if last_target is not None:
                use[delayed] = last_target
            last_target = target[delayed].copy()
        # The servo's internal position sensor and the reported encoder share
        # the same zero error. MuJoCo joint positions/limits remain physical.
        return old_control(use, q+zero_rad, dq, dt)

    def torque(*args, **kwargs):
        return gain * old_torque(*args, **kwargs)

    def feedback(data, acceleration):
        old_feedback(data, acceleration)
        plant.feedback.state.angles[plant.feedback.core.POLICY_INDEX] += zero_deg

    def imu(acc, gyro, *args, **kwargs):
        return old_filter(rotation @ acc, rotation @ gyro + gyro_bias, *args, **kwargs)

    with ExitStack() as scope:
        scope.enter_context(patch.object(actuator, "compute_control", control))
        scope.enter_context(patch.object(actuator, "compute_torque", torque))
        scope.enter_context(patch.object(plant.feedback, "_process", feedback))
        scope.enter_context(patch.object(plant.feedback._filter, "update", imu))
        yield


def factors(index):
    """Each 32 consecutive trials in one block cover the four-way factorial.

    Every release band visits all eight side/delay/calibration combinations.
    A second block reverses the forward command for every combination.
    """
    band = index % 4
    rep = index // 4 % 8
    a, b, c = rep & 1, (rep >> 1) & 1, (rep >> 2) & 1
    side = a ^ (band & 1)
    delay = a ^ b ^ (band >> 1)
    saved = a ^ c ^ (band & 1) ^ (band >> 1)
    fast = side ^ delay ^ saved ^ ((index // 32) & 1)
    return band, side, 5 * delay, bool(saved), .20 if fast else .15


def schedule(trials, seed, paths, reference, run_trim=False):
    from walk_bench import Case
    jobs = []
    ranges = ((.25,.85),(1.,2.65),(4.3,6.9),(7.05,10.15))
    for foot in HEELS:
        for f, floor in enumerate(FLOORS):
            for i in range(trials):
                s = seed + 101*i + f
                rng = random.Random(s)
                band, side, delay, saved, command = factors(i)
                low,high = ranges[band]
                duration = round(rng.uniform(low,high)*50)/50
                case = Case(s,foot=foot,surface=floor,vx=command,seconds=duration,
                            stop=3.5, calibrated=saved)
                roles = {**paths[foot],**reference}
                order = list(roles)
                rng.shuffle(order)
                spec = mismatch(s,i,weak_leg=side,extra_delay_ms=delay)
                jobs.append((case,spec,[(k,str(roles[k])) for k in order],run_trim,band))
    random.Random(seed+19).shuffle(jobs)
    return jobs


def trial(job):
    from run_audit import audit_episode
    from walk_bench import _actor, _environment
    case,spec,order,run_trim,band = job
    plant = _environment(case.foot,case.surface)
    results, cache = {}, {}
    for role, path in order:
        trimmed = run_trim and role == "run"
        key = (path, trimmed)
        if key not in cache:
            # Clear delay history on each fresh replay, including when the
            # reference follows the candidate instead of preceding it.
            with ExitStack() as context:
                if trimmed:
                    from heel_profiles.policy_trim_sim import run_policy_trim
                    actor = _actor(Path(path).resolve())
                    context.enter_context(run_policy_trim(plant,actor,enabled=True))
                context.enter_context(apply_mismatch(plant,spec))
                cache[key] = audit_episode(path,case,gait=True)
        results[role] = cache[key]
    return dict(case=case.__dict__,release_band=band,mismatch=spec,order=[role for role,_ in order],actors=results)


def summary(rows):
    output = {}
    for role in rows[0]["actors"]:
        trials = [r["actors"][role] for r in rows]
        peak = [r["stop_peak_m"] for r in trials if r["stop_peak_m"] is not None]
        errors = [r["case"]["seed"] for r in trials if r["fell"] or not r["stop_ok"]]
        margins = [r["min_leg_joint_margin_deg"] for r in trials if r["min_leg_joint_margin_deg"] is not None]
        torques = [r["peak_torque_ratio"] for r in trials if r["peak_torque_ratio"] is not None]
        late = [r["steady_speed_mps"] for r in trials if r["steady_speed_mps"] is not None]
        output[role] = dict(trials=len(trials),falls=sum(r["fell"] for r in trials),
                sustained_stops=sum(r["stop_ok"] for r in trials),fall_or_stop_failure_seeds=errors,
                mean_progress_per_requested_s=fmean(r["forward_m"]/r["case"]["seconds"] for r in trials),
                mature_walk_trials=len(late),mean_available_steady_mps=fmean(late) if late else None,
                worst_peak_stop_m=max(peak,default=None), mean_peak_stop_m=fmean(peak) if peak else None,
                mean_peak_cross_m=fmean(r["peak_cross_m"] for r in trials),
                min_joint_margin_deg=min(margins,default=None),max_torque_ratio=max(torques,default=None),
                running_pattern_trials=sum(r["running_pattern"] is True for r in trials))
    pairs = [("run","walk")]
    if "reference_walk" in output:
        pairs.append(("walk","reference_walk"))
    if "reference_run" in output:
        pairs.append(("run","reference_run"))
    for b,a in pairs:
        delta = [(r["actors"][b]["forward_m"]-r["actors"][a]["forward_m"])/r["case"]["seconds"] for r in rows]
        mature = [r["actors"][b]["steady_speed_mps"]-r["actors"][a]["steady_speed_mps"] for r in rows
                  if all(r["actors"][q]["steady_speed_mps"] is not None for q in (a,b))]
        output[f"{b}_minus_{a}"] = dict(mean_paired_progress_mps=fmean(delta),
                faster_trials=sum(x>0 for x in delta),paired_mature_trials=len(mature),
                mean_paired_steady_mps=fmean(mature) if mature else None,
                new_falls=sum(r["actors"][b]["fell"] and not r["actors"][a]["fell"] for r in rows),
                new_failed_stops=sum(r["actors"][a]["stop_ok"] and not r["actors"][b]["stop_ok"] for r in rows))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory",type=Path,help="Folder containing factory/5mm/10mm, each with walk.onnx and run.onnx")
    source.add_argument("--profiles",type=Path,help="JSON mapping v2-stock/v2-heel5/v2-heel10 to walk/run paths")
    ap.add_argument("--baseline-walk",type=Path)
    ap.add_argument("--baseline-run",type=Path)
    ap.add_argument("--run-policy-trim",action="store_true",help="Use the controller's policy-only Run gravity path on the candidate Run")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--trials",type=int,default=32,help="Paired releases per heel and floor. Use multiples of 32 for a full factorial and 64 to cross command speed too")
    ap.add_argument("--seed",type=int,default=13_429_880)
    args=ap.parse_args()
    if min(args.trials,args.workers)<1:
        ap.error("Need at least one trial and worker")
    for var in THREAD_VARS:
        os.environ[var]="1"
    if args.directory:
        paths = {f:{g:(args.directory/h/(g+".onnx")).resolve(strict=True) for g in ("walk","run")}
                 for f,h in HEELS.items()}
    else:
        data=json.loads(args.profiles.read_text())
        paths={f:{g:(args.profiles.parent/data[f][g]).resolve(strict=True) for g in ("walk","run")} for f in HEELS}
    reference={k:p.resolve(strict=True) for k,p in (("reference_walk",args.baseline_walk),
                                                    ("reference_run",args.baseline_run)) if p}
    all_paths={f"{f}/{g}":p for f,roles in paths.items() for g,p in roles.items()} | reference
    out=args.out.resolve()
    if out in all_paths.values():
        ap.error("Choose a report path separate from model files")
    hashes={k:verify_model_file(p) for k,p in all_paths.items()}
    jobs=schedule(args.trials,args.seed,paths,reference,args.run_policy_trim)
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        episodes=list(pool.map(trial,jobs))
    if hashes!={k:verify_model_file(p) for k,p in all_paths.items()}:
        raise RuntimeError("Input policy changed during paired trials")
    buckets=defaultdict(list)
    for row in sorted(episodes,key=lambda r:(r["case"]["foot"],r["case"]["surface"],r["case"]["seed"])):
        c=row["case"]
        buckets[f"{c['foot']}/{c['surface']}"].append(row)
    coverage = {}
    for group, rows in buckets.items():
        counts = Counter((r["release_band"],r["mismatch"]["weak_leg"],r["mismatch"]["extra_delay_ms"],
                          r["case"]["calibrated"],r["case"]["vx"]) for r in rows)
        coverage[group] = [dict(band=b,weak_leg=s,extra_delay_ms=d,calibrated=c,vx=v,trials=n)
                           for (b,s,d,c,v),n in sorted(counts.items())]
    report=dict(seed=args.seed, sha256=hashes, trials_per_heel_floor=args.trials,
                run_policy_trim=args.run_policy_trim, factorial_coverage=coverage,
                assumptions="Each episode holds one leg at 87–95% nominal torque, the opposite at 98–104%, "
                "plus independent joint variation of ±2 percentage points. In each 32-case block, each release band covers every combination "
                "of weak side, 0/5 ms extra target delay and saved/absent calibration. Each of the next 32 cases swaps the matching command speed. Encoder errors span ±0.7° per leg joint, with "
                "matching servo internal feedback and reported encoder readings. IMU roll/pitch errors span ±1.5°, "
                "gyro biases ±0.3/0.3/0.6°/s. These ranges are assumptions.",
                release="Random 50 Hz release ticks in four duration bands; 3.5 s of neutral command. "
                "Sustained stop uses the 100 Hz run_audit criterion. Each heel gets the same cases and "
                "mismatch as its comparators. Execution and actor order are seeded and shuffled.",
                groups={group:summary(rows) for group,rows in buckets.items()},episodes=episodes)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,report)
    for group,vals in report["groups"].items():
        print(group,vals["run_minus_walk"],"walk/run failed stops",
              len(vals["walk"]["fall_or_stop_failure_seeds"]),len(vals["run"]["fall_or_stop_failure_seeds"]))


if __name__=="__main__":
    main()

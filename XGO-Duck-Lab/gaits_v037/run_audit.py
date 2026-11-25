"""Compare a candidate with the installed Walk pilot on speed, flight and abrupt stops."""

import argparse
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from statistics import fmean

from train_walk import THREAD_VARS, verify_model_file, write_json


HZ = 100
QUIET_WINDOW = 20
QUIET_SECONDS = 2.
MAX_STOP_SPEED = .02
MAX_STOP_YAW_RATE = 8.
SOURCES = ("reference", "candidate")
Sample = namedtuple("Sample", "x y z yaw tilt support other flight torque_peak torque_sq joint_margin")
LEG_INDEX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


class Sampler:
    """Tap the plant immediately after its existing 100 Hz feedback sample."""

    def __init__(self, env, gait):
        import mujoco
        import numpy as np
        self.env, self.gait = env, gait
        self.mj, self.np = mujoco, np
        self.points = []
        self.leg_act = np.asarray(env.ctrl.act_indexes)[LEG_INDEX]
        self.leg_pos = env.qpos_adr[LEG_INDEX]
        self.joint_range = env.qrange[LEG_INDEX]
        self.torque_limit = np.max(np.abs(env.model.actuator_forcerange[self.leg_act]), axis=1)
        self.force = np.empty(6)
        self.weight = env.model.body_subtreemass[env.trunk] * np.linalg.norm(env.model.opt.gravity)
        self.callback = env.advance

    def capture(self, data):
        m, env, np = self.env.model, self.env, self.np
        x, y, z = data.xpos[env.trunk]
        rotation = data.xmat[env.trunk]
        yaw = math.atan2(rotation[3], rotation[0])
        tilt = math.degrees(math.acos(max(-1., min(1., rotation[8]))))
        support = other = flight = 0
        if self.gait:
            forces = [0., 0., 0.]
            for idx, c in enumerate(data.contact):
                g1, g2 = c.geom
                geom = g2 if g1 == env.floor else g1 if g2 == env.floor else -1
                if geom < 0 or c.efc_address < 0:
                    continue
                self.mj.mj_contactForce(m, data, idx, self.force)
                foot = 0 if geom == env.soles[0] else 1 if geom == env.soles[1] else 2
                forces[foot] += max(0., self.force[0])
            support = int(forces[0] > .02*self.weight) + 2*int(forces[1] > .02*self.weight)
            other = int(forces[2] > .02*self.weight)
            if not support and not other:
                flight = int(all(self.mj.mj_geomDistance(m, data, env.floor, sole, .02, None) > .001
                                 for sole in env.soles))
        ratios = np.abs(data.actuator_force[self.leg_act]) / self.torque_limit
        q = data.qpos[self.leg_pos]
        margin = min((q-self.joint_range[:, 0]).min(), (self.joint_range[:, 1]-q).min())
        self.points.append(Sample(x, y, z, yaw, tilt, support, other, flight,
                                  ratios.max(), np.mean(ratios*ratios), math.degrees(margin)))

    def advance(self, target, action, sample_callback=None):
        if not self.points:
            self.capture(self.env.data)
        def observe(data):
            if sample_callback is not None:
                sample_callback(data)
            self.capture(data)
        return self.callback(target, action, sample_callback=observe)


def _flight(steps, enabled):
    empty = dict(airborne_events=None, alternating_airborne_events=None, observed_flight_bouts=None,
                 airborne_sample_fraction=None, longest_flight_s=None, running_pattern=None, release_support=None)
    if not enabled or not steps:
        return empty
    # Ignore startup contacts through the first second of movement.
    starts, durations, directions = [], [], []
    i = HZ
    while i < len(steps):
        if not steps[i].flight:
            i += 1
            continue
        start = i
        while i < len(steps) and steps[i].flight:
            i += 1
        duration = (i - start - 1) / HZ
        durations.append(duration)
        if duration < .03:
            continue
        starts.append(start)
        before = next((p.support for p in reversed(steps[max(0,start-15):start]) if p.support), 0)
        after = next((p.support for p in steps[i:i+15] if p.support), 0)
        if (before, after) in ((1, 2), (2, 1)):
            directions.append((start, before, after))
    # Require flight late in the command and contacts consistent with each leg
    # taking a turn. An isolated launch, hop or momentary loss of contact fails.
    repeats = sum(i >= (HZ+len(steps)) / 2 for i, _, _ in directions)
    both_directions = {before for _, before, after in directions} == {1, 2}
    running = (len(directions) >= 4 and repeats >= 2 and both_directions
               and not any(p.other for p in steps[HZ:]))
    return dict(airborne_events=len(starts), alternating_airborne_events=len(directions),
                observed_flight_bouts=len(durations),
                airborne_sample_fraction=fmean(p.flight for p in steps[HZ:]) if len(steps)>HZ else 0.,
                longest_flight_s=max(durations, default=0.), running_pattern=running,
                release_support=("none", "left", "right", "both")[steps[-1].support])


def _stop_metrics(points, complete):
    import numpy as np
    result = dict(stop_peak_m=None, stop_net_m=None, settle_s=None, stop_ok=False,
                  standing_after_stop=False, stop_sampled_s=max(0, len(points)-1)/HZ)
    if len(points) < 2:
        return result
    xy = np.array([(p.x, p.y) for p in points])
    excursion = np.linalg.norm(xy-xy[0], axis=1)
    result.update(stop_peak_m=float(excursion.max()), stop_net_m=float(excursion[-1]))
    if len(points) < QUIET_SECONDS*HZ + 1 or not complete:
        return result
    standing = np.array([p.z >= .105 and p.tilt <= 15. for p in points])
    result["standing_after_stop"] = bool(standing[-(round(QUIET_SECONDS*HZ)+1):].all())
    world_yaw = np.unwrap([p.yaw for p in points])
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    rotation = np.r_[0., np.cumsum(np.abs(np.diff(np.rad2deg(world_yaw))))]
    w = QUIET_WINDOW
    speed = (path[w:]-path[:-w]) / (w/HZ)
    yaw_rate = (rotation[w:]-rotation[:-w]) / (w/HZ)
    # For the entire remaining quiet interval, each complete 0.2s window must
    # pass speed and yaw-rate limits, and every individual pose must be upright.
    bad = np.flatnonzero((speed > MAX_STOP_SPEED) | (yaw_rate > MAX_STOP_YAW_RATE))
    bad_pose = np.flatnonzero(~standing)
    start = max(bad[-1]+1 if bad.size else 0, bad_pose[-1]+1 if bad_pose.size else 0)
    if (len(points)-1-start)/HZ >= QUIET_SECONDS:
        result["settle_s"] = float(start/HZ)
        result["stop_ok"] = True
    return result


def audit_episode(model, case, distance_m=None, *, gait=True):
    """Evaluate one exported ONNX. Setting gait=False skips contact classification."""
    import numpy as np
    from walk_bench import _environment, _ticks, simulate
    env = _environment(case.foot, case.surface)
    tap = Sampler(env, gait)
    previous = env.__dict__.get("advance")
    env.advance = tap.advance
    try:
        bench = simulate(None, case, model, finish_forward_m=distance_m)
    finally:
        if previous is None:
            del env.advance
        else:
            env.advance = previous
    onset = 2 * _ticks(case.settle)
    release = onset + round(bench["movement_s"] * HZ)
    movement = tap.points[onset:release+1]
    stop = tap.points[release:] if bench["fell_phase"] not in ("settle","move") else []
    out = dict(case=bench["case"], distance_target_m=distance_m, fell=bench["fell"],
               fell_phase=bench["fell_phase"], commanded_s=bench["movement_s"],
               forward_m=bench["forward_m"], peak_cross_m=bench["max_cross_m"],
               max_tilt_deg=bench["max_tilt_deg"], rms_tilt_deg=bench["rms_tilt_deg"],
               rms_roll_deg=bench["rms_roll_deg"], peak_roll_deg=bench["peak_roll_deg"],
               surface_contact=bench["contact"],
               steady_speed_mps=None, late_speed_mps=None, min_late_1s_speed_mps=None,
               finish_s=None, reached=None if distance_m is None else False,
               peak_torque_ratio=None, rms_torque_ratio=None, torque_near_limit_fraction=None,
               min_leg_joint_margin_deg=None)
    out.update(_stop_metrics(stop, bench["completed"]))
    out.update(_flight(movement, gait))
    if len(movement) <= 1:
        return out
    origin = movement[0]
    x = np.array([p.x-origin.x for p in movement]); y = np.array([p.y-origin.y for p in movement])
    fwd = x*math.cos(origin.yaw) + y*math.sin(origin.yaw)
    cross = y*math.cos(origin.yaw) - x*math.sin(origin.yaw)
    out["peak_cross_m"] = float(np.max(np.abs(cross)))
    # Require at least one second in each half before calling speed sustained.
    if len(fwd)-1 >= 2*HZ:
        mid = (len(fwd)-1)//2
        out["steady_speed_mps"] = float((fwd[-1]-fwd[mid]) * HZ / (len(fwd)-1-mid))
        out["late_speed_mps"] = float(fwd[-1]-fwd[-1-HZ])
        out["min_late_1s_speed_mps"] = float(np.min(fwd[mid+HZ:]-fwd[mid:-HZ]))
    if distance_m is not None:
        # Use control ticks, at which the benchmark releases the command.
        reached = np.flatnonzero(fwd[::2] >= distance_m)
        if reached.size:
            i = 2*int(reached[0])
            frac = (distance_m-fwd[i-2]) / (fwd[i]-fwd[i-2])
            out["finish_s"] = float((i-2+2*frac)/HZ)
            out["reached"] = True
    out["peak_torque_ratio"] = float(max(p.torque_peak for p in movement))
    out["rms_torque_ratio"] = math.sqrt(fmean(p.torque_sq for p in movement))
    out["torque_near_limit_fraction"] = fmean(float(p.torque_peak >= .95) for p in movement)
    out["min_leg_joint_margin_deg"] = float(min(p.joint_margin for p in movement))
    return out


def _run_pair(job):
    ref, cand, case, distance, group = job
    return group, {label: audit_episode(path, case, distance)
                   for label, path in zip(SOURCES, (ref, cand))}


def summarize(rows):
    group = {}
    for label in SOURCES:
        trials = [row[label] for row in rows]
        keys = ("steady_speed_mps","late_speed_mps","min_late_1s_speed_mps","peak_cross_m","stop_peak_m",
                "settle_s","peak_torque_ratio","rms_torque_ratio","min_leg_joint_margin_deg","finish_s")
        means = {}
        for key in keys:
            values = [trial[key] for trial in trials if trial[key] is not None]
            means[key] = fmean(values) if values else None
        group[label] = dict(trials=len(trials), falls=sum(r["fell"] for r in trials),
                            running=sum(r["running_pattern"] is True for r in trials),
                            stopped=sum(r["stop_ok"] for r in trials),
                            reached=sum(r["reached"] is True for r in trials),
                            course_attempts=sum(r["reached"] is not None for r in trials),
                            mean_available=means, release_support=dict(Counter(
                                r["release_support"] for r in trials if r["release_support"] is not None)))
    both = [(r["reference"], r["candidate"]) for r in rows if all(r[k]["steady_speed_mps"] is not None for k in SOURCES)]
    group["paired"] = dict(comparable_speed=len(both), faster=sum(c["steady_speed_mps"] > a["steady_speed_mps"] for a,c in both),
                           faster_and_stopped=sum(c["steady_speed_mps"] > a["steady_speed_mps"] and c["stop_ok"] for a,c in both),
                           candidate_new_falls=sum(c["fell"] and not a["fell"] for a,c in
                                                   ((r["reference"], r["candidate"]) for r in rows)))
    return group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--trials", type=int, default=6, help="Pairs per short floor/speed group")
    p.add_argument("--seed", type=int, default=493100)
    p.add_argument("--long", action="store_true", help="Also compare 50 m courses at .20 m/s")
    p.add_argument("--long-trials", type=int, default=2, help="Pairs per endurance floor")
    args = p.parse_args()
    if min(args.workers,args.trials,args.long_trials) < 1:
        p.error("Workers and trial counts must be positive")
    for var in THREAD_VARS:
        os.environ[var] = "1"
    from walk_bench import Case
    ref, cand = (path.resolve(strict=True) for path in (args.reference,args.candidate))
    out = args.out.resolve()
    if out in (ref, cand):
        p.error("Choose an output separate from the model files")
    hashes = {key: verify_model_file(model) for key,model in zip(SOURCES,(ref,cand))}
    jobs, groups = [], {}
    for vx,distance,count in [(.20,None,args.trials),(.15,None,args.trials)] + ([(.20,50.,args.long_trials)] if args.long else []):
        for surface in ("hard","carpet"):
            group = f"{surface}-{vx:.2f}-{'50m' if distance else 'abrupt'}"
            slot = len(groups)
            groups[group] = []
            for j in range(count):
                seed = args.seed + 6*j + slot
                # The command release changes independently of encoder/stance state.
                seconds = 420. if distance else random.Random(seed).randint(260,445)/50.
                case = Case(seed,foot="v2-heel5",surface=surface,vx=vx,seconds=seconds,stop=3.5)
                jobs.append((ref,cand,case,distance,group))
    out.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for group,row in pool.map(_run_pair,jobs):
            groups[group].append(row)
    if hashes != {key: verify_model_file(model) for key,model in zip(SOURCES,(ref,cand))}:
        raise RuntimeError("A model changed while the audit was running")
    report = dict(
        sha256=hashes, seed=args.seed,
        speed_definition="Signed progress along starting torso heading; steady speed averages the last half of movement; "
                         "late speed and minimum sliding one-second speed expose a launch or late stall.",
        flight_definition="100 Hz contact samples with both soles >1 mm clear and no feet or other body part carrying "
                          ">2% body weight. Flight lasts >=30 ms from first to last clear sample. After excluding the first second, "
                          "running requires >=4 flights with opposite exclusive foot supports within 150 ms before/after, "
                          "both transfer directions, >=2 transfers in the latter half of that scored interval and no other body support.",
        stop_definition="Peak radial excursion from the release point. Settling is the earliest time after release when every "
                        "remaining 0.2s window has <=0.02 m/s planar path speed and <=8 deg/s yaw variation, all trunk "
                        "samples have height >=0.105 m and tilt <=15 deg, and at least 2s remain before the end of the 3.5s stop.",
        load_definition="Leg actuator force magnitude / model actuator torque limit, sampled at 100 Hz. Joint margins are "
                        "distances to the model mechanical ranges. These model outputs describe simulated loading.",
        groups={k:summarize(v) for k,v in groups.items()}, episodes=groups)
    write_json(out,report)
    print(json.dumps(report["groups"],indent=2,allow_nan=False))


if __name__ == "__main__":
    main()

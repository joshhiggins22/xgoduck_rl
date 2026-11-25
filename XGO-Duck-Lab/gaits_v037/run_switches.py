"""Exercise three Walk/Run graph transitions using the 0.3.7 inference and filters."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from statistics import fmean
from unittest.mock import patch

from run_audit import HZ, Sampler, _stop_metrics
from train_walk import THREAD_VARS, verify_model_file, write_json


SEQUENCES = ("walk_to_run_held", "run_to_walk_held", "run_to_walk_neutral")


@lru_cache(maxsize=4)
def _load(slot, path):
    # Separate stateful Policy instances even when the two files are byte-identical.
    from walk_bench import _runtime
    _, _, cls = _runtime()
    actor = cls()
    ok, message = actor.load(path)
    if not ok:
        raise ValueError(message)
    return actor


class Router:
    def __init__(self, policies, sequence, toggle_tick, start_tick, release_tick, switch):
        self.policies = policies
        self.sequence = sequence
        self.toggle_tick, self.start_tick, self.release_tick = toggle_tick, start_tick, release_tick
        self.switch = switch
        self.events = []

    def reset_action_filter(self):
        for actor in self.policies.values():
            actor.reset_action_filter()
        self.slot = "walk"
        self.tick = 0

    @property
    def prev_action(self):
        return self.policies[self.slot].prev_action

    def action_to_deg(self, action):
        return self.policies[self.slot].action_to_deg(action)

    def choice(self):
        if self.tick < self.start_tick:
            return "walk"
        if self.tick >= self.release_tick:
            # The neutral-control trial deliberately leaves the Run graph loaded
            # to isolate the effect of swapping graphs from the effect of releasing.
            return "run" if self.sequence == "run_to_walk_neutral" and not self.switch else "walk"
        starts_walk = self.sequence == "walk_to_run_held"
        if self.switch and self.tick >= self.toggle_tick:
            return "run" if starts_walk else "walk"
        return "walk" if starts_walk else "run"

    def infer(self, *args):
        import numpy as np
        old = self.policies[self.slot]
        destination = self.choice()
        actor = self.policies[destination]
        is_event = self.tick == self.toggle_tick or destination != self.slot
        if is_event:
            previous_raw = old.prev_action.copy()
            previous_filtered = old._act_filt.copy()
            # Predict the old graph's next output on this exact observation before
            # switching. This is a measurement only, and does not advance its state.
            obs = old.build_obs(*args)
            counter_raw = old.session.run(old._outputs, {old.in_name: obs[None]})[0][0]
            counter_filt = previous_filtered.copy()
            for sl, alpha in ((slice(0, 5), old.action_alpha), (slice(5, 9), old.action_alpha_head),
                              (slice(9, 14), old.action_alpha)):
                counter_filt[sl] *= alpha
                counter_filt[sl] += counter_raw[sl] * (1-alpha)
            if destination != self.slot:
                actor.prev_action[:] = previous_raw
                actor._act_filt[:] = previous_filtered
            filter_copy_error = float(np.max(np.abs(actor._act_filt - previous_filtered)))
        filtered = actor.infer(*args)
        if is_event:
            def deg(delta):
                return float(np.rad2deg(np.abs(delta).max()))
            self.events.append(dict(tick=self.tick, from_slot=self.slot, to_slot=destination,
                primary=self.tick == self.toggle_tick, raw_history_error_rad=float(np.max(np.abs(actor._obs[34:48]-previous_raw))),
                filter_copy_error_rad=filter_copy_error,
                peak_raw_step_deg=deg(actor.prev_action-previous_raw), peak_filtered_target_step_deg=deg(filtered-previous_filtered),
                extra_filtered_target_step_vs_old_graph_deg=deg(filtered-counter_filt),
                extra_raw_step_vs_old_graph_deg=deg(actor.prev_action-counter_raw)))
        self.slot = destination
        self.tick += 1
        return filtered


def episode(walk, run, case, sequence, before_ticks, switch):
    import walk_bench as bench
    env = bench._environment(case.foot, case.surface)
    start = bench._ticks(case.settle)
    release = start + bench._ticks(case.seconds)
    toggle = release if sequence == "run_to_walk_neutral" else start + before_ticks
    policies = {name: _load(name, path) for name, path in (("walk",walk), ("run",run))}
    router = Router(policies, sequence, toggle, start, release, switch)
    samples = Sampler(env, gait=True)
    with patch.object(bench, "_actor", return_value=router), patch.object(env, "advance", samples.advance):
        trial = bench.simulate(None, case, walk)
    points = samples.points
    start_i, toggle_i, release_i = 2*start, 2*toggle, 2*release
    floor_axis = None if len(points)<=start_i else (math.cos(points[start_i].yaw), math.sin(points[start_i].yaw))

    def speed(a,b):
        if floor_axis is None or not 0 <= a < b < len(points):
            return None
        return float(((points[b].x-points[a].x)*floor_axis[0]+(points[b].y-points[a].y)*floor_axis[1])*HZ/(b-a))

    stop = points[release_i:] if trial["fell_phase"] not in ("settle","move") else []
    transition = points[toggle_i:toggle_i+HZ//2+1]
    event = next((e for e in router.events if e["primary"]),None)
    state_exact = bool(event and all(e["raw_history_error_rad"] == e["filter_copy_error_rad"] == 0. for e in router.events))
    row = dict(
        case=trial["case"], sequence=sequence, switched=switch,
        primary_transition_s=(toggle-start)*bench.CONTROL_DT,
        source_state_carried_exactly=state_exact, events=router.events, primary_event=event,
        forward_m=trial["forward_m"], peak_cross_m=trial["max_cross_m"],
        speed_before_mps=speed(toggle_i-HZ,toggle_i),
        # Leave 0.5 s for the new gait or braking transient before the next 1 s window.
        speed_after_mps=speed(toggle_i+HZ//2,toggle_i+HZ//2+HZ),
        late_forward_speed_mps=speed(release_i-HZ,release_i),
        fell=trial["fell"], fell_phase=trial["fell_phase"],
        peak_trunk_rise_in_0p5s_m=max((p.z-transition[0].z for p in transition),default=None),
        peak_trunk_tilt_in_0p5s_deg=max((p.tilt for p in transition),default=None),
        airborne_samples_in_0p5s=sum(p.flight for p in transition), max_tilt_deg=trial["max_tilt_deg"],
    )
    row.update(_stop_metrics(stop,trial["completed"]))
    return row


def pair(job):
    walk, run, case, sequence, before = job
    return {key: episode(walk,run,case,sequence,before,change) for key,change in (("control",False),("switch",True))}


def summary(pairs):
    rows = [r["switch"] for r in pairs]
    events = [r["primary_event"] for r in rows if r["primary_event"]]
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return fmean(values) if values else None
    def delta(key):
        values = [r["switch"][key]-r["control"][key] for r in pairs if all(r[k][key] is not None for k in ("switch","control"))]
        return fmean(values) if values else None
    return dict(
        trials=len(rows), primary_events=len(events), exact_state_transfers=sum(r["source_state_carried_exactly"] for r in rows),
        falls=sum(r["fell"] for r in rows), new_falls=sum(r["switch"]["fell"] and not r["control"]["fell"] for r in pairs),
        stopped=sum(r["stop_ok"] for r in rows), control_stopped=sum(r["control"]["stop_ok"] for r in pairs),
        mean_speed_before_mps=mean("speed_before_mps"), mean_speed_after_mps=mean("speed_after_mps"),
        mean_paired_speed_after_delta_mps=delta("speed_after_mps"), mean_stop_peak_m=mean("stop_peak_m"),
        mean_paired_stop_peak_delta_m=delta("stop_peak_m"), mean_settle_s=mean("settle_s"),
        worst_extra_filtered_step_deg=max((e["extra_filtered_target_step_vs_old_graph_deg"] for e in events),default=None),
        worst_total_filtered_step_deg=max((e["peak_filtered_target_step_deg"] for e in events),default=None),
        worst_trunk_rise_in_0p5s_m=max((r["peak_trunk_rise_in_0p5s_m"] for r in rows if r["peak_trunk_rise_in_0p5s_m"] is not None),default=None),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walk",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trials",type=int,default=4,help="Paired seeds per surface and sequence")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--foot",choices=("v2-stock","v2-heel5","v2-heel10"),default="v2-heel5")
    p.add_argument("--surfaces",nargs="+",choices=("hard","carpet","hardwood","residential-carpet"),default=("hard","carpet"))
    p.add_argument("--seed",type=int,default=614100)
    args=p.parse_args()
    if min(args.workers,args.trials)<1:
        p.error("Workers and trials must be positive")
    for var in THREAD_VARS:
        os.environ[var]="1"
    from walk_bench import Case
    walk,run=(path.resolve(strict=True) for path in (args.walk,args.run))
    out=args.out.resolve()
    if out in (walk,run):
        p.error("Choose an output separate from the models")
    hashes={name:verify_model_file(path) for name,path in (("walk",walk),("run",run))}
    jobs,groups=[],{}
    for seq in SEQUENCES:
        for floor in args.surfaces:
            label=f"{seq}-{floor}"
            slot=len(groups)
            groups[label]=[]
            for i in range(args.trials):
                seed=args.seed+len(SEQUENCES)*len(args.surfaces)*i+slot
                rng=random.Random(seed)
                before,after=rng.randint(137,229),rng.randint(126,198)
                sec=(before+after)/50.
                case=Case(seed,foot=args.foot,surface=floor,vx=.20,seconds=sec,stop=3.5)
                jobs.append((walk,run,case,seq,before))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(pair,jobs):
            trial=row["switch"]
            groups[trial["sequence"]+"-"+trial["case"]["surface"]].append(row)
    if hashes!={name:verify_model_file(path) for name,path in (("walk",walk),("run",run))}:
        raise RuntimeError("A model changed while the switch audit was running")
    report=dict(sha256=hashes,seed=args.seed,
        definition="Simulate the proposed checkbox at a 50 Hz inference boundary using two 0.3.7 Policy instances. "
                   "Copy the source raw action history and filtered action before the destination computes its next step. "
                   "Keep the physical plant, IMU, heading target, home and filter settings continuous. For the neutral-release "
                   "control, continue the Run graph on the same zero command. For held switches the control keeps its original "
                   "graph during forward motion. Extra step compares the new and old outputs on the identical observation.",
        stop_definition="3.5 s at neutral; at least 2 s continuously upright with <=.02 m/s planar path speed in each 0.2 s "
                        "window and <=8 deg/s yaw variation per window, as in run_audit.",
        groups={k:summary(v) for k,v in groups.items()},episodes=groups)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,report)
    print(json.dumps(report["groups"],indent=2,allow_nan=False))


if __name__ == "__main__":
    main()

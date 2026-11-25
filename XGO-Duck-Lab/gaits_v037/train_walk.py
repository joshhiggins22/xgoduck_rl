"""Fit a bounded forward-command adapter to the v0.3.7 Walk actor on 5 mm heels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
from statistics import fmean, median

PRIMARY_FOOT = "v2-heel5"
SURFACES = ("hard", "carpet")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run_trial(job):
    from walk_bench import simulate
    params, case, model = job
    return simulate(params, case, onnx_path=model)


def run_batch(pool, settings, cases, model):
    """Spawn a process once; walk_bench reuses simulation and ONNX sessions in it."""
    cases = tuple(cases)
    jobs = ((params, case, model) for params in settings for case in cases)
    rows = iter(pool.map(run_trial, jobs, chunksize=1))
    return {p: list(itertools.islice(rows, len(cases))) for p in settings}


def cases_for(foot, seed_start, n, speed, seconds, stop):
    from walk_bench import Case
    return [Case(seed=seed, foot=foot, surface=surface, vx=speed, seconds=seconds,
                 settle=1., stop=stop, calibrated=True)
            for surface in SURFACES for seed in range(seed_start, seed_start + n)]


def summarize(rows):
    stops = [r for r in rows if r["stop_speed_mps"] is not None]
    dists = [r["stop_dist_m"] for r in rows if r["stop_dist_m"] is not None]
    return {
        "episodes": len(rows), "falls": sum(bool(r["fell"]) for r in rows),
        "completed": sum(bool(r["completed"]) for r in rows),
        "speed_mps": fmean(r["speed_mps"] for r in rows),
        "rms_cross_m": fmean(r["rms_cross_m"] for r in rows),
        "max_cross_m": fmean(r["max_cross_m"] for r in rows),
        "max_yaw_deg": fmean(r["max_yaw_deg"] for r in rows),
        "stop_measured": len(stops),
        "stop_speed_mps": fmean(r["stop_speed_mps"] for r in stops) if stops else None,
        "stop_dist_m": fmean(dists) if dists else None,
    }


def paired_group(stock_rows, cand_rows):
    if [r["case"] for r in stock_rows] != [r["case"] for r in cand_rows]:
        raise ValueError("Unpaired stock/candidate trials")
    base = summarize(stock_rows)
    cand = summarize(cand_rows)
    delta = {key: cand[key] - base[key] for key in base
             if cand[key] is not None and base[key] is not None}
    return {
        "stock": base, "candidate": cand, "delta_candidate_minus_stock": delta,
        "candidate_only_failures": sum(a["completed"] and not b["completed"]
                                       for a, b in zip(stock_rows, cand_rows)),
        "stock_only_failures": sum(b["completed"] and not a["completed"]
                                   for a, b in zip(stock_rows, cand_rows)),
        "faster_seeds": sum(b["speed_mps"] > a["speed_mps"] for a, b in zip(stock_rows, cand_rows)),
        "straighter_seeds": sum(b["rms_cross_m"] < a["rms_cross_m"] for a, b in zip(stock_rows, cand_rows)),
        "faster_and_straighter_seeds": sum(
            b["speed_mps"] > a["speed_mps"] and b["rms_cross_m"] < a["rms_cross_m"]
            for a, b in zip(stock_rows, cand_rows)),
        "median_paired_speed_delta_mps": median(b["speed_mps"] - a["speed_mps"]
                                               for a, b in zip(stock_rows, cand_rows)),
    }


def compare(stock_rows, cand_rows):
    groups = {}
    for surface in SURFACES:
        a = [r for r in stock_rows if r["case"]["surface"] == surface]
        b = [r for r in cand_rows if r["case"]["surface"] == surface]
        groups[surface] = paired_group(a, b)
    return groups


def ranking(groups):
    # Pair failures seed by seed. A new fall on one floor cannot be hidden by
    # surviving on an unrelated seed. For viable candidates, reward progress
    # and tracking together, putting extra weight on the weaker floor.
    failures = sum(g["candidate_only_failures"] for g in groups.values())
    scores = []
    for g in groups.values():
        d = g["delta_candidate_minus_stock"]
        speed = d["speed_mps"] / .02
        cross = -d["rms_cross_m"] / .012
        yaw = -d["max_yaw_deg"] / 5.
        # Saturation prevents one extreme metric dominating the other two.
        speed, cross, yaw = (max(-4., min(4., x)) for x in (speed, cross, yaw))
        tracking = .8 * cross + .2 * yaw
        scores.append(min(speed, tracking) + .4 * (speed + tracking))
    return (-failures, .6 * min(scores) + .4 * fmean(scores))


def bounded(params):
    from walk_adapter import BOUNDS
    return tuple(round(max(lo, min(hi, x)), 6)
                 for x, (lo, hi) in zip(params, BOUNDS.values()))


def screening(rng):
    from walk_adapter import BOUNDS, IDENTITY
    base = tuple(IDENTITY.values())
    points = [base]
    for i, (low, high) in enumerate(BOUNDS.values()):
        for value in ((low + high) / 2., low, high):
            p = list(base)
            p[i] = value
            points.append(tuple(p))
    # Stratify each coordinate, then shuffle their pairing reproducibly.
    axes = []
    for low, high in BOUNDS.values():
        axis = [low + (high - low) * (i + rng.random()) / 16 for i in range(16)]
        rng.shuffle(axis)
        axes.append(axis)
    points.extend(zip(*axes))
    return list(dict.fromkeys(bounded(p) for p in points))


def refinement(leaders, rng):
    from walk_adapter import BOUNDS
    points = list(leaders)
    for index, center in enumerate(leaders[:2]):
        widths = [(hi - lo) * (.13 if index == 0 else .22) for lo, hi in BOUNDS.values()]
        for axis, width in enumerate(widths):
            for sign in (-1, 1):
                p = list(center)
                p[axis] += sign * width
                points.append(bounded(p))
        for _ in range(4):
            points.append(bounded([x + rng.uniform(-w, w) for x, w in zip(center, widths)]))
    return list(dict.fromkeys(points))


def fit_stage(pool, name, points, cases, stock, out, identity):
    data = run_batch(pool, points, cases, stock)
    records = []
    for params, rows in data.items():
        groups = compare(data[identity], rows)
        rank = ranking(groups)
        records.append({"params": params, "rank": rank, "surfaces": groups, "episodes": rows})
    records.sort(key=lambda r: r["rank"], reverse=True)
    write_json(out / (name + ".json"), records)
    print(f"{name}: {len(points)} candidates x {len(cases)} paired episodes. "
          f"Leading parameters {records[0]['params']}", flush=True)
    return records


def verify_model_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path,
                        help="Verified v0.3.7 Python directory containing policy.py")
    parser.add_argument("--out", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--search-seed", type=int, default=3715)
    parser.add_argument("--train-seed", type=int, default=37100)
    parser.add_argument("--holdout-seed", type=int, default=57300)
    parser.add_argument("--holdout-count", type=int, default=10,
                        help="Fresh full-speed trial seeds for each primary surface")
    args = parser.parse_args()
    if args.workers < 1 or args.holdout_count < 1:
        parser.error("workers and holdout-count must be positive")
    if set(range(args.train_seed, args.train_seed + 24)) & set(range(
            args.holdout_seed, args.holdout_seed + max(202, args.holdout_count))):
        parser.error("Training and held-out seed ranges must be disjoint")
    for var in THREAD_VARS:
        os.environ[var] = "1"

    from export_walk import export_walk
    from walk_adapter import BOUNDS, IDENTITY
    rng = random.Random(args.search_seed)
    stock = args.stock.resolve()
    runtime = args.runtime_python.resolve()
    digest = verify_model_file(stock)
    if digest != verify_model_file(runtime / "xgoduck_walk.onnx"):
        parser.error("Stock and the supplied runtime Walk model differ")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = tuple(IDENTITY.values())
    print(f"Fitting on {PRIMARY_FOOT} using {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        first = fit_stage(pool, "screen", screening(rng), cases_for(
            PRIMARY_FOOT, args.train_seed, 2, .2, 4., .0), stock, out, identity)
        leaders = [r["params"] for r in first[:4]]
        points = list(dict.fromkeys([identity] + refinement(leaders, rng)))
        final = fit_stage(pool, "refine", points, cases_for(
            PRIMARY_FOOT, args.train_seed + 20, 4, .2, 5., .8), stock, out, identity)
        # Preserve the best experimental candidate even when unchanged stock ranks first.
        selected = next(r for r in final if tuple(r["params"]) != identity)
        params = dict(zip(BOUNDS, selected["params"]))
        write_json(out / "parameters.json", params)
        model = out / "walk-candidate.onnx"
        parity = export_walk(stock, params, model, runtime_python=runtime)
        print(f"Locked candidate {params}; evaluating its exported ONNX on fresh seeds.", flush=True)
        report = {
            "stock_sha256": digest, "candidate_sha256": verify_model_file(model),
            "params": params, "onnx_parity": parity, "search_seed": args.search_seed,
            "train_seed": args.train_seed, "holdout_seed": args.holdout_seed,
            "stock_ranked_first_in_refinement": tuple(final[0]["params"]) == identity,
            "protocol": "Training uses stock ONNX + parameterized commands; held-out episodes run the exported actor. "
                        "Mean speed includes startup and falls. Drift is the RMS sideways distance from the initial "
                        "line; maximum yaw is the largest heading excursion. Stop distance measures travel after release.",
            "checks": {},
        }
        tests = [("primary_full", PRIMARY_FOOT, .2, args.holdout_count, 0),
                 ("primary_partial", PRIMARY_FOOT, .1, 4, 100),
                 ("stock_feet_transfer", "v2-stock", .2, 2, 200),
                 ("15mm_transfer", "v2-heel15", .2, 2, 200)]
        for name, foot, speed, count, offset in tests:
            trials = cases_for(foot, args.holdout_seed + offset, count, speed, 7., 1.25)
            base = run_batch(pool, [None], trials, stock)[None]
            cand = run_batch(pool, [None], trials, model)[None]
            groups = compare(base, cand)
            report["checks"][name] = {
                "surfaces": groups, "stock_episodes": base, "candidate_episodes": cand,
            }
            write_json(out / "holdout.json", report)
            for floor, g in groups.items():
                d = g["delta_candidate_minus_stock"]
                print(f"{name}/{floor}: speed {d['speed_mps']:+.4f} m/s, "
                      f"rms drift {d['rms_cross_m']:+.4f} m, "
                      f"new failures {g['candidate_only_failures']}/{count}", flush=True)


if __name__ == "__main__":
    main()

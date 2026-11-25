"""Replay the continuous 15 mm heel sit and rise in the native MuJoCo viewer."""

import argparse
from contextlib import ExitStack
from itertools import repeat
import json
import time

import mujoco.viewer

from controlled_cycle import evaluate
from duck_env import ROOT, DT, DECIMATION


class WindowClosed(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3500)
    parser.add_argument("--surface", choices=["carpet", "hard", "hardwood", "residential-carpet"], default="carpet")
    parser.add_argument("--cycles", type=int, help="Close after this many replays")
    args = parser.parse_args()
    if args.cycles is not None and args.cycles < 1:
        parser.error("--cycles must be positive")
    path = ROOT / "runs/heel-cycle-descent/parameters.json"
    parameters = json.loads(path.read_text())["params"]
    seeds = repeat(args.seed) if args.cycles is None else repeat(args.seed, args.cycles)
    with ExitStack() as windows:
        viewer = None
        previous_frame = None

        def draw(env, tick):
            nonlocal viewer, previous_frame
            if viewer is None:
                viewer = windows.enter_context(mujoco.viewer.launch_passive(env.model, env.data))
                viewer.cam.distance = 0.48
                viewer.cam.lookat[:] = [0, 0, 0.11]
                viewer.cam.azimuth = 98
                viewer.cam.elevation = -12
            if not viewer.is_running():
                raise WindowClosed
            if previous_frame is not None:
                time.sleep(max(0.0, DT * DECIMATION - (time.monotonic() - previous_frame)))
            viewer.sync()
            previous_frame = time.monotonic()

        try:
            evaluate(parameters, args.surface, seeds, observer=draw)
        except WindowClosed:
            pass


if __name__ == "__main__":
    main()

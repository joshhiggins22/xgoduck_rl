"""Evaluate a 0.3.7 Walk actor in the native XGO-Duck-Lab simulation."""

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


LAB = Path(os.environ.get("DUCK_LAB", Path(__file__).resolve().parents[1])).resolve()
RUNTIME = Path(__file__).resolve().parent / "runtime"
CONTROL_DT = 0.02


@dataclass(frozen=True)
class Case:
    seed: int
    foot: str = "v2-stock"
    surface: str = "hard"
    vx: float = 0.2
    vy: float = 0.0
    wz: float = 0.0
    seconds: float = 5.0
    calibrated: bool = True
    settle: float = 1.0
    stop: float = 0.0


@lru_cache(maxsize=1)
def _runtime():
    if not (RUNTIME / "heading.py").is_file():
        raise FileNotFoundError(f"Expected the 0.3.7 runtime at {RUNTIME}. Set DUCK_LAB to the lab directory.")
    sys.path[:0] = [str(LAB), str(RUNTIME)]
    from duck_env import DuckSitStandEnv
    from heading import HeadingHold
    from policy import Policy
    return DuckSitStandEnv, HeadingHold, Policy


@lru_cache(maxsize=None)
def _environment(foot, surface):
    environment, _, _ = _runtime()
    return environment(foot=foot, surface=surface, start="stand", runtime_dir=RUNTIME)


@lru_cache(maxsize=8)
def _actor(path):
    _, _, policy_class = _runtime()
    policy = policy_class()
    ok, message = policy.load(path)
    if not ok:
        raise ValueError(message)
    return policy


class _IdentityCalibration:
    """Ideal simulated IMU with zero offsets, processed as a saved calibration."""

    @staticmethod
    def apply(acc, gyro):
        return list(acc), list(gyro)


def _pose(env):
    rotation = env.data.xmat[env.trunk]
    return env.data.xpos[env.trunk, :2].copy(), math.atan2(rotation[3], rotation[0])


def _ticks(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Episode durations must be finite and nonnegative")
    return round(seconds / CONTROL_DT)


def simulate(params, case, onnx_path=None, trace=None, *, finish_forward_m=None):
    """Settle, apply a fixed body-frame twist, then release. One simulation per worker.

    Speed is signed progress along the initial torso heading per full requested
    movement duration. Cross-track is perpendicular to that same floor line.
    Yaw is unwrapped independently from the onboard gyro estimate. Passing an
    exported actor with params=None replays just the bytes the board would load.
    A supplied trace receives the origin and each 50 Hz movement-window pose.
    A positive finish distance releases the command on the first tick after
    crossing, then runs the same neutral stop phase as a time-limited trial.
    """
    settle, move, stop = map(_ticks, (case.settle, case.seconds, case.stop))
    if move < 1 or not all(map(math.isfinite, (case.vx, case.vy, case.wz))):
        raise ValueError("Expected a finite twist and a positive movement duration")
    if finish_forward_m is not None and (not math.isfinite(finish_forward_m) or finish_forward_m <= 0):
        raise ValueError("Finish distance must be finite and positive")
    env = _environment(case.foot, case.surface)
    _, heading_class, _ = _runtime()
    path = Path(onnx_path or LAB / "runs/v037-v2-heel5-forward-walk/walk.onnx").resolve(strict=True)
    actor = _actor(path)
    actor.reset_action_filter()
    sensor_filter = env.feedback._filter
    sensor_filter.correction = _IdentityCalibration() if case.calibrated else None
    sensor_filter.command_acc_x_bias_ms2 = 0.0
    env.max_steps = settle + move + stop
    _, contact = env.reset(seed=case.seed)
    hold = heading_class()

    def observe_heading(data):
        state = env.feedback.state
        hold.observe(True, state.grav, state.gyro_rad, float(data.time))

    observe_heading(env.data)
    if params is not None:
        from walk_adapter import adapt_twist
    idle = (0.0, 0.0, 0.0)
    moving = (case.vx, case.vy, case.wz)
    # controller.py applies this trim using the requested command, before yaw hold.
    moving_bias = 1.5 if case.vx > 0 else -1.5 if case.vx < 0 or case.wz != 0 else 0.0
    origin, yaw_previous = _pose(env)
    axis = np.array([math.cos(yaw_previous), math.sin(yaw_previous)])
    perpendicular = np.array([-axis[1], axis[0]])
    progress = cross = max_cross = squared_cross = yaw = max_yaw = max_tilt = 0.0
    squared_roll = squared_tilt = peak_roll = 0.0
    movement_ticks = hold_ticks = 0
    correction_sum = 0.0
    stop_positions = []
    fell_phase = None
    release_tick = settle + move
    for tick in range(env.max_steps):
        if tick >= release_tick + stop:
            break
        is_moving = settle <= tick < release_tick
        if tick == settle:
            origin, yaw_previous = _pose(env)
            axis[:] = math.cos(yaw_previous), math.sin(yaw_previous)
            perpendicular[:] = -axis[1], axis[0]
            if trace is not None:
                trace.append(dict(t_s=0., forward_m=0., cross_m=0., yaw_deg=0.))
        if tick == release_tick:
            stop_positions.append(_pose(env)[0])
        twist = moving if is_moving else idle
        sensor_filter.command_acc_x_bias_ms2 = moving_bias if case.calibrated and is_moving else 0.0
        model_cmd = (twist[0], twist[1], hold.command(True, twist))
        if params is not None:
            model_cmd = adapt_twist(model_cmd, params)
        state = env.feedback.state
        filtered = actor.infer(state.grav, state.gyro_rad, state.angles, state.vels, model_cmd)
        target_deg = np.asarray(actor.action_to_deg(filtered))[env.feedback.core.POLICY_INDEX]
        _, _, fell, _, info = env.advance(np.deg2rad(target_deg), actor.prev_action, observe_heading)
        position, yaw_now = _pose(env)
        max_tilt = max(max_tilt, info["tilt_deg"])
        if is_moving:
            movement_ticks += 1
            mat = env.data.xmat[env.trunk]
            roll = math.degrees(math.atan2(mat[7], mat[8]))
            peak_roll = max(peak_roll, abs(roll))
            squared_roll += roll * roll
            squared_tilt += info["tilt_deg"] ** 2
            hold_ticks += int(hold.target is not None)
            correction_sum += abs(hold.correction)
            displacement = position - origin
            progress = float(displacement @ axis)
            cross = float(displacement @ perpendicular)
            max_cross = max(max_cross, abs(cross))
            squared_cross += cross * cross
            angle_change = yaw_now - yaw_previous
            yaw += math.atan2(math.sin(angle_change), math.cos(angle_change))
            max_yaw = max(max_yaw, abs(yaw))
            yaw_previous = yaw_now
            if trace is not None:
                trace.append(dict(t_s=movement_ticks * CONTROL_DT, forward_m=progress,
                                  cross_m=cross, yaw_deg=math.degrees(yaw)))
            if finish_forward_m is not None and progress >= finish_forward_m:
                release_tick = tick + 1
        elif tick >= release_tick:
            stop_positions.append(position)
        if fell:
            fell_phase = "move" if is_moving else "settle" if tick < settle else "stop"
            break

    stop_speed = stop_distance = None
    if len(stop_positions) >= 2:
        tail_ticks = min(25, len(stop_positions) - 1)
        tail_displacement = stop_positions[-1] - stop_positions[-1-tail_ticks]
        stop_speed = float(np.linalg.norm(tail_displacement) / (tail_ticks * CONTROL_DT))
        stop_distance = float(np.linalg.norm(stop_positions[-1] - stop_positions[0]))
    result = dict(
        case=asdict(case), contact=contact, fell=fell_phase is not None, fell_phase=fell_phase,
        completed=fell_phase is None, movement_s=movement_ticks * CONTROL_DT,
        forward_m=progress, speed_mps=progress / ((release_tick - settle) * CONTROL_DT),
        end_cross_m=cross, max_cross_m=max_cross,
        rms_cross_m=math.sqrt(squared_cross / max(1, movement_ticks)),
        end_yaw_deg=math.degrees(yaw), max_yaw_deg=math.degrees(max_yaw),
        rms_roll_deg=math.sqrt(squared_roll / max(1,movement_ticks)), peak_roll_deg=peak_roll,
        rms_tilt_deg=math.sqrt(squared_tilt / max(1,movement_ticks)),
        max_tilt_deg=max_tilt, hold_active_fraction=hold_ticks / max(1, movement_ticks),
        mean_abs_correction_rps=correction_sum / max(1, movement_ticks),
        stop_speed_mps=stop_speed, stop_dist_m=stop_distance,
    )
    if finish_forward_m is not None:
        result["stop_elapsed_s"] = max(0, len(stop_positions) - 1) * CONTROL_DT
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--params", nargs=3, type=float, metavar=("VX_GAIN", "VY_PER_VX", "WZ_PER_VX"))
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--foot", default="v2-stock")
    parser.add_argument("--surface", choices=("hard", "carpet", "hardwood", "residential-carpet"), default="hard")
    parser.add_argument("--vx", type=float, default=.2)
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--wz", type=float, default=0.)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--settle", type=float, default=1.)
    parser.add_argument("--stop", type=float, default=1.)
    parser.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=True)
    options = vars(parser.parse_args())
    onnx_path = options.pop("onnx")
    parameters = options.pop("params")
    print(json.dumps(simulate(parameters, Case(**options), onnx_path), indent=2))

"""Fit and export an opt-in, faster forward command on the verified XGO Walk pilot."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from export_walk import verify_runtime
from walk_adapter import STRAFE_LIMIT, YAW_LIMIT


PILOT_SHA256 = "15e8b8ea6fb69a98fdf3c028c3899859bfd2416bca6b8e2cea0b2ce409d5233c"
BOUNDS = {"target_vx": (.20, .40), "lateral": (-.045, .045), "yaw": (-.22, .22),
          "stride": (-.10, .30)}
NOMINAL_VX = np.float32(.20)
STOCK_MAX_VX = np.float32(.40)
# With the verified pilot's high-speed row gain=1.6 and clipped vx=.20.
MAX_PILOT_BOOST = (np.float32(1.6) - np.float32(1)) * NOMINAL_VX
MAX_TRIM_RAD = np.float32(.10)
SAGITTAL = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], np.float32)
PREFIX = "xgo_run_policy/"


@dataclass(frozen=True)
class SprintParams:
    target_vx: float = .20
    lateral: float = 0.0
    yaw: float = 0.0
    stride: float = 0.0


IDENTITY = asdict(SprintParams())


def _params(params):
    if isinstance(params, SprintParams):
        params = asdict(params)
    if params.keys() != BOUNDS.keys():
        raise ValueError(f"Expected sprint coefficients {tuple(BOUNDS)}")
    values = {}
    for name, (lo, hi) in BOUNDS.items():
        val = np.float32(params[name])
        if not np.float32(lo) <= val <= np.float32(hi):
            raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
        values[name] = val
    return values


def _axis_gate(vy, wz):
    return (max(np.float32(0), np.float32(1) - abs(wz) / YAW_LIMIT)
            * max(np.float32(0), np.float32(1) - abs(vy) / STRAFE_LIMIT))


def _forward_gate(twist):
    vx, vy, wz = twist
    return min(max(vx / NOMINAL_VX, np.float32(0)), np.float32(1)) * _axis_gate(vy, wz)


def adapt_command(twist, params):
    """Map the runtime command (after heading hold) to the unmodified pilot."""
    p = _params(params)
    original = np.asarray(twist, dtype=np.float32)
    gate = _forward_gate(original)
    if gate == 0:
        return original.copy()
    vx, vy, wz = original
    vy += p["lateral"] * gate
    wz += p["yaw"] * gate
    vx += (p["target_vx"] - NOMINAL_VX) * gate
    # Budget for the additional forward command the embedded stock adapter will
    # generate given these new lateral/turn inputs. At default console/phone
    # speeds, IDENTITY reproduces the pilot. High custom requests remain capped.
    vx = min(vx, STOCK_MAX_VX - MAX_PILOT_BOOST * _axis_gate(vy, wz))
    return np.array([vx, vy, wz], np.float32)


def combine_raw(obs, base_raw, params):
    """Add a bounded stride adjustment before runtime smoothing and prev_action."""
    p = _params(params)
    raw = np.asarray(base_raw, dtype=np.float32)
    gate = _forward_gate(np.asarray(obs, dtype=np.float32)[48:51])
    delta = np.clip(raw * p["stride"], -MAX_TRIM_RAD, MAX_TRIM_RAD)
    return raw + delta * SAGITTAL * gate


def _compose(network, p):
    graph = network.graph
    source, output = graph.input[0].name, graph.output[0].name
    # Redirect every pilot consumer, including its existing PPO leg residual,
    # to the same adapted observation. Preserve the original graph as a unit.
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == source:
                node.input[i] = PREFIX + "obs"
            elif name == output:
                node.input[i] = PREFIX + "pilot_actions"
        for i, name in enumerate(node.output):
            if name == output:
                node.output[i] = PREFIX + "pilot_actions"
    for info in graph.value_info:
        if info.name == output:
            info.name = PREFIX + "pilot_actions"
    base_nodes = list(graph.node)
    del graph.node[:]

    def const(name, value, dtype=np.float32):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), PREFIX + name))
        return PREFIX + name

    def op(kind, inputs, name, **attrs):
        dst = PREFIX + name
        graph.node.append(helper.make_node(kind, inputs, [dst], name=dst, **attrs))
        return dst

    zero, one = const("zero", 0), const("one", 1)
    vy_limit, wz_limit = const("lateral_limit", STRAFE_LIMIT), const("yaw_limit", YAW_LIMIT)
    prefix, vx, vy, wz, suffix = [PREFIX + n for n in ("prefix", "vx", "vy", "wz", "suffix")]
    split = const("observation_sizes", [48, 1, 1, 1, 10], np.int64)
    graph.node.append(helper.make_node("Split", [source, split], [prefix, vx, vy, wz, suffix],
                                       name=PREFIX + "split_obs", axis=1))

    def axis_gate(name, lateral, yaw):
        parts = []
        for axis, value, limit in (("yaw", yaw, wz_limit), ("strafe", lateral, vy_limit)):
            key = name + "_" + axis
            absolute = op("Abs", [value], key + "_abs")
            frac = op("Div", [absolute, limit], key + "_fraction")
            remaining = op("Sub", [one, frac], key + "_remaining")
            parts.append(op("Max", [zero, remaining], key + "_gate"))
        return op("Mul", parts, name)

    pos = op("Div", [vx, const("nominal", NOMINAL_VX)], "forward_fraction")
    pos = op("Clip", [pos, zero, one], "forward_gate")
    gate = op("Mul", [pos, axis_gate("user_axes", vy, wz)], "gate")
    proposed = []
    for axis, value, shift in (("vx", vx, p["target_vx"]-NOMINAL_VX),
                               ("vy", vy, p["lateral"]), ("wz", wz, p["yaw"])):
        delta = op("Mul", [const(axis + "_trim", shift), gate], axis + "_delta")
        proposed.append(op("Add", [value, delta], axis + "_proposed"))
    new_x, new_y, new_z = proposed
    pilot_gate = axis_gate("pilot_axes", new_y, new_z)
    boost = op("Mul", [const("max_inner_boost", MAX_PILOT_BOOST), pilot_gate], "inner_boost")
    limit = op("Sub", [const("stock_max", STOCK_MAX_VX), boost], "effective_limit")
    capped = op("Min", [new_x, limit], "capped_vx")
    active = op("Greater", [gate, zero], "forward_active")
    new_x = op("Where", [active, capped, vx], "final_vx")
    op("Concat", [prefix, new_x, new_y, new_z, suffix], "obs", axis=1)
    graph.node.extend(base_nodes)
    raw = PREFIX + "pilot_actions"
    delta = op("Mul", [raw, const("stride_scale", p["stride"])], "excursion")
    delta = op("Clip", [delta, const("negative_trim", -MAX_TRIM_RAD),
                        const("positive_trim", MAX_TRIM_RAD)], "bounded_excursion")
    delta = op("Mul", [delta, const("sagittal_mask", SAGITTAL)], "sagittal_excursion")
    delta = op("Mul", [delta, gate], "joint_trim")
    graph.node.append(helper.make_node("Add", [raw, delta], [output], name=PREFIX + "raw_action"))
    prop = network.metadata_props.add()
    prop.key = "xgo_run_policy"
    prop.value = json.dumps(dict(pilot_sha256=PILOT_SHA256, params={k: float(v) for k, v in p.items()},
                                 max_stock_forward_mps=.40, max_sagittal_trim_rad=.10), sort_keys=True)
    onnx.checker.check_model(network)


def _verify(base_path, path, p):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    base, candidate = [ort.InferenceSession(str(f), sess_options=options, providers=["CPUExecutionProvider"])
                       for f in (base_path, path)]
    original_meta = base.get_modelmeta().custom_metadata_map
    if any(candidate.get_modelmeta().custom_metadata_map.get(k) != v for k, v in original_meta.items()):
        raise ValueError("An underlying pilot metadata field changed")
    saved_profile = json.loads(original_meta["xgo_walk_adapter"])["params"]
    from walk_adapter import adapt_twist
    rng = np.random.default_rng(27011)
    commands = [(x, y, z) for x in (-.2, 0., .10, .15, .20, .25, .28, .32, .40)
                for y, z in ((0, 0), (.03, -.12), (-.08, 0), (0, .35), (.20, -.68))]
    commands.extend(rng.uniform([-.20, -.10, -.40], [.40, .10, .40], (96, 3)))
    maximum_error = maximum_cmd = max_joint_delta = 0.
    inactive = 0
    for command in commands:
        obs = rng.normal(0, .35, (1, 61)).astype(np.float32)
        obs[0, 5] -= .95
        obs[0, 48:51] = command
        adapted = obs.copy()
        adapted[0, 48:51] = adapt_command(obs[0, 48:51], p)
        final_x = adapt_twist(adapted[0, 48:51], saved_profile)[0]
        maximum_cmd = max(maximum_cmd, float(final_x))
        if final_x > STOCK_MAX_VX + 1e-6:
            raise ValueError("An upstream actor command exceeded its training range")
        raw = base.run(None, {"obs": adapted})[0][0]
        actual = candidate.run(None, {"obs": obs})[0][0]
        expected = combine_raw(obs[0], raw, p)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected-actual))))
        max_joint_delta = max(max_joint_delta, float(np.max(np.abs(raw-actual))))
        np.testing.assert_array_equal(actual[SAGITTAL == 0], raw[SAGITTAL == 0])
        if _forward_gate(obs[0, 48:51]) == 0:
            np.testing.assert_array_equal(adapted, obs)
            np.testing.assert_array_equal(actual, raw)
            inactive += 1
    if max_joint_delta > MAX_TRIM_RAD + 2e-6:
        raise ValueError("A sprint joint trim exceeded its bound")
    return dict(observations=len(commands), unchanged_command_cases=inactive,
                max_abs_action_error=maximum_error, max_stock_input_mps=maximum_cmd,
                max_joint_trim_rad=max_joint_delta)


def export_run_policy(base_onnx, params, out, *, verify=True, runtime_python=None):
    """Set verify=False for ephemeral simulations; full verification runs on selected graphs."""
    base, output = Path(base_onnx).resolve(strict=True), Path(out).resolve()
    if base == output or hashlib.sha256(base.read_bytes()).hexdigest() != PILOT_SHA256:
        raise ValueError("Use the verified pilot as source and a separate sprint output")
    p = _params(params)
    network = onnx.load(base)
    _compose(network, p)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=output.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        onnx.save(network, temp)
        result = dict(verified=False, base_sha256=PILOT_SHA256, export_sha256=hashlib.sha256(temp.read_bytes()).hexdigest())
        if verify:
            result.update(_verify(base, temp, p))
            release = Path(runtime_python) if runtime_python else Path(
                os.environ.get("DUCK_LAB", Path(__file__).resolve().parent)) / "runtime"
            verify_runtime(temp, release)
            result.update(verified=True, release_loader="loaded and passed 12 warm-up inferences")
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return result

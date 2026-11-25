"""Export and verify a fitted command adapter on the 0.3.7 stock Walk ONNX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from walk_adapter import (
    BOUNDS, FORWARD_LIMIT, SPEED_KNOTS, STRAFE_LIMIT, YAW_LIMIT, adapt_twist, coefficients,
)


STOCK_SHA256 = "1fc079512fe3fe23e274e7a9d8c9ad007e88439f0385ecf20c9596e57886249f"
PREFIX = "xgo_forward_adapter/"


def add_adapter(model, values):
    graph = model.graph
    input_name = graph.input[0].name
    for node in graph.node:
        for index, tensor in enumerate(node.input):
            if tensor == input_name:
                node.input[index] = PREFIX + "obs"

    def constant(name, data, dtype=np.float32):
        tensor = numpy_helper.from_array(np.asarray(data, dtype=dtype), PREFIX + name)
        graph.initializer.append(tensor)

    zero, one = "zero", "one"
    constant(zero, 0.0)
    constant(one, 1.0)
    constant("forward_limit", FORWARD_LIMIT)
    constant("yaw_limit", YAW_LIMIT)
    constant("strafe_limit", STRAFE_LIMIT)
    constant("split", [48, 1, 1, 1, 10], np.int64)
    nodes = [helper.make_node("Split", [input_name, PREFIX + "split"],
                              [PREFIX + s for s in ("prefix", "vx", "vy", "wz", "suffix")],
                              axis=1, name=PREFIX + "split_commands")]

    def op(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, [PREFIX + name for name in inputs],
                                      [PREFIX + output], name=PREFIX + output, **attributes))

    if values.ndim == 1:
        gain, lateral, yaw = values
        constant("gain_delta", gain - np.float32(1))
        constant("lateral", lateral)
        constant("yaw", yaw)
        params_metadata = dict(zip(BOUNDS, map(float, values)))
    else:
        # Across the two adjacent intervals the three weights are
        # [1-a, a-b, b]. Clipping saturates to the endpoint rows.
        for i, blend_name in enumerate(("a", "b")):
            constant(blend_name + "_start", SPEED_KNOTS[i])
            constant(blend_name + "_width", SPEED_KNOTS[i + 1] - SPEED_KNOTS[i])
            op("Sub", ["vx", blend_name + "_start"], blend_name + "_offset")
            op("Div", [blend_name + "_offset", blend_name + "_width"], blend_name + "_raw")
            op("Clip", [blend_name + "_raw", zero, one], blend_name)
        op("Sub", [one, "a"], "row0_weight")
        op("Sub", ["a", "b"], "row1_weight")
        for i, row_weight in enumerate(("row0_weight", "row1_weight", "b")):
            row = f"row{i}"
            constant(row, values[i][None, :])
            op("Mul", [row, row_weight], row + "_weighted")
        op("Add", ["row0_weighted", "row1_weighted"], "first_two_rows")
        op("Add", ["first_two_rows", "row2_weighted"], "coefficients")
        constant("coefficient_sizes", [1, 1, 1], np.int64)
        nodes.append(helper.make_node("Split", [PREFIX + "coefficients", PREFIX + "coefficient_sizes"],
                                      [PREFIX + name for name in ("gain", "lateral", "yaw")],
                                      axis=1, name=PREFIX + "split_coefficients"))
        op("Sub", ["gain", one], "gain_delta")
        params_metadata = {"speed_profile": values.tolist()}

    op("Clip", ["vx", zero, "forward_limit"], "forward")
    for value, limit in (("wz", "yaw_limit"), ("vy", "strafe_limit")):
        op("Abs", [value], value + "_abs")
        op("Div", [value + "_abs", limit], value + "_fraction")
        op("Sub", [one, value + "_fraction"], value + "_remainder")
        op("Clip", [value + "_remainder", zero, one], value + "_weight")
    op("Mul", ["forward", "wz_weight"], "heading_weighted")
    op("Mul", ["heading_weighted", "vy_weight"], "weight")
    for axis, param in (("vx", "gain_delta"), ("vy", "lateral"), ("wz", "yaw")):
        op("Mul", [param, "weight"], axis + "_delta")
        op("Add", [axis, axis + "_delta"], axis + "_adapted")
    op("Concat", ["prefix", "vx_adapted", "vy_adapted", "wz_adapted", "suffix"], "obs", axis=1)
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + original_nodes)
    metadata = model.metadata_props.add()
    metadata.key = "xgo_walk_adapter"
    info = {"stock_sha256": STOCK_SHA256, "params": params_metadata}
    if values.ndim == 2:
        info["command_vx_knots_mps"] = [round(float(speed), 2) for speed in SPEED_KNOTS]
    metadata.value = json.dumps(info, sort_keys=True)
    onnx.checker.check_model(model, full_check=True)


def verify_equivalence(stock, candidate, values):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sessions = [ort.InferenceSession(str(path), sess_options=options,
                                     providers=["CPUExecutionProvider"])
                for path in (stock, candidate)]
    original, wrapped = sessions
    for getter in ("get_inputs", "get_outputs"):
        def signature(session):
            return [(v.name, v.shape, v.type) for v in getattr(session, getter)()]
        if signature(original) != signature(wrapped):
            raise ValueError("Export changed the stock ONNX interface")
    metadata = original.get_modelmeta().custom_metadata_map
    wrapped_metadata = wrapped.get_modelmeta().custom_metadata_map
    if any(wrapped_metadata.get(key) != value for key, value in metadata.items()):
        raise ValueError("Export changed a stock runtime metadata field")
    rng = np.random.default_rng(3781)
    # Include direct axis commands, tiny turns, gradual gates and their endpoints.
    unaffected = [(0, 0, 0), (-.15, 0, 0), (-.2, .04, .1), (0, 0, .5),
                  (0, .2, 0), (.2, 0, .35), (.2, 0, -.5), (.2, .08, 0), (.2, -.1, .1)]
    active = [(.15, 0, 0), (.20, 0, 0), (.30, 0, 0), (.10, -.02, .15), (.20, .06, -.30)]
    trials = [(command, True) for command in unaffected] + [(command, False) for command in active]
    for speed in (.06, .10, .125, .15, .175, .20, .26):
        trials.extend([((speed, 0, 0), False), ((speed, .03, .18), False),
                       ((speed, -.08, .01), True), ((speed, 0, -.35), True)])
    for speed in SPEED_KNOTS:
        for direction in (-np.inf, np.inf):
            neighbor = np.nextafter(speed, np.float32(direction))
            trials.append(((neighbor, 0, 0), False))
    for _ in range(96):
        trials.append((rng.uniform([.01, -.065, -.34], [.32, .065, .34]), False))
    for _ in range(32):
        trials.append((rng.uniform([-.2, -.3, -1.0], [0, .3, 1.0]), True))
    max_error = 0.0
    unchanged_error = 0.0
    for command, unchanged in trials:
        obs = rng.normal(0, .3, (1, 61)).astype(np.float32)
        obs[0, 48:51] = command
        reference_obs = obs.copy()
        reference_obs[0, 48:51] = adapt_twist(obs[0, 48:51], values)
        if values.ndim == 2:
            # Independent float64 interpolation checks saturation, continuity
            # and assignment of left, middle and right rows in the NumPy path.
            vx = obs[0, 48]
            interpolated = [np.interp(vx, SPEED_KNOTS, col) for col in values.T]
            independently_adapted = adapt_twist(obs[0, 48:51], interpolated)
            np.testing.assert_allclose(reference_obs[0, 48:51], independently_adapted,
                                       rtol=2e-6, atol=2e-7)
        if unchanged:
            np.testing.assert_array_equal(reference_obs, obs)
        baseline = original.run(None, {"obs": reference_obs})[0]
        output = wrapped.run(None, {"obs": obs})[0]
        np.testing.assert_allclose(output, baseline, rtol=2e-6, atol=2e-6)
        error = float(np.max(np.abs(output - baseline)))
        max_error = max(max_error, error)
        if unchanged:
            unchanged_error = max(unchanged_error, error)
    return {"observations": len(trials), "max_abs_action_error": max_error,
            "unchanged_command_cases": sum(item[1] for item in trials),
            "unchanged_max_abs_action_error": unchanged_error,
            "stock_metadata_fields_preserved": len(metadata)}


def verify_runtime(path, runtime_python):
    runtime_python = Path(runtime_python).resolve()
    if not (runtime_python / "policy.py").is_file():
        raise ValueError("--runtime-python must point to the release directory containing policy.py")
    # Run the actual deployment loader in a separate interpreter so it imports
    # rl_core from the selected release even if a simulator imported its own.
    check = """import sys
sys.path.insert(0, sys.argv[1])
from policy import Policy
loaded, message = Policy().load(sys.argv[2])
if not loaded:
    raise RuntimeError(message)
print(message)
"""
    result = subprocess.run([sys.executable, "-c", check, str(runtime_python), str(path)],
                            check=True, text=True, capture_output=True)
    return result.stdout.strip()


def export_walk(stock, params, output, runtime_python=None):
    stock, output = Path(stock).resolve(), Path(output).resolve()
    if stock == output:
        raise ValueError("Choose a destination separate from the stock policy")
    if hashlib.sha256(stock.read_bytes()).hexdigest() != STOCK_SHA256:
        raise ValueError("Source does not match the verified XGO-Duck 0.3.7 stock Walk policy")
    values = coefficients(params)
    model = onnx.load(stock)
    add_adapter(model, values)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        onnx.save_model(model, temporary)
        result = verify_equivalence(stock, temporary, values)
        if runtime_python:
            verify_runtime(temporary, runtime_python)
            result["release_loader"] = "loaded and passed 12 warm-up inferences"
        result["export_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True,
                        help="JSON with three adapter parameters or three speed_profile rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, help="0.3.7 directory containing policy.py")
    args = parser.parse_args()
    print(json.dumps(export_walk(args.stock, json.loads(args.params.read_text()), args.output,
                                 args.runtime_python), indent=2))

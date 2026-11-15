"""A simulated rise followed by the frozen XGO policy. The host supplies a 50 Hz tick."""

import numpy as np
import onnxruntime as ort

CHANNELS = (2, 3, 4, 5, 11, 12, 13)
END_TICK = 35


def table(knots):
    """Three symmetric pose targets, linear between knots 0/7/18 and held to tick 25."""
    knots = np.asarray(knots, dtype=np.float32).reshape(3, 4)
    ticks = np.arange(END_TICK + 1)
    small = np.stack([np.interp(ticks, [0, 7, 18], column) for column in knots.T], axis=1)
    poses = np.zeros((len(ticks), 14), dtype=np.float32)
    poses[:, [2, 3, 4, 5]] = small
    poses[:, [11, 12, 13]] = -small[:, :3]
    weights = np.zeros_like(poses)
    weights[:, CHANNELS] = np.clip((END_TICK - ticks[:, None]) / 10.0, 0.0, 1.0)
    return poses, weights


class RisePolicy:
    """Simulation host: arm once per seated reset, then return to ordinary stock inference."""

    def __init__(self, session):
        self.session = session
        self.tick = END_TICK

    def reset(self, initial_obs):
        flexion = (initial_obs[9] - initial_obs[18]) * 0.5
        self.tick = 0 if flexion > 0.65 and not np.any(initial_obs[48:]) else END_TICK

    def __call__(self, obs):
        if np.any(obs[48:]):
            self.tick = END_TICK
        feed = {
            "obs": obs.astype(np.float32).reshape(1, 61),
            "rise_tick": np.array([self.tick], dtype=np.int64),
        }
        action = self.session.run(None, feed)[0][0]
        self.tick = min(END_TICK, self.tick + 1)
        return action


def load_policy(path):
    op = ort.SessionOptions()
    op.intra_op_num_threads = 1
    op.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), op, providers=["CPUExecutionProvider"])
    return RisePolicy(s)


def export(factory, knots, dest):
    """Keep all ONNX normalization and learned weights; append waypoint interpolation."""
    import onnx
    from onnx import TensorProto as T, helper as H, numpy_helper as N

    m = onnx.load(str(factory))
    poses, weights = table(knots)
    output = m.graph.output[0].name
    (writer,) = [node for node in m.graph.node if output in node.output]
    writer.output[list(writer.output).index(output)] = "factory_action"
    m.graph.input.append(H.make_tensor_value_info("rise_tick", T.INT64, [1]))
    for name, arr in [("rise_poses", poses), ("rise_weights", weights)]:
        m.graph.initializer.append(N.from_array(arr, name=name))
    m.graph.node.extend(
        [
            H.make_node("Gather", ["rise_poses", "rise_tick"], ["rise_pose"], axis=0),
            H.make_node("Gather", ["rise_weights", "rise_tick"], ["rise_weight"], axis=0),
            H.make_node("Sub", ["rise_pose", "factory_action"], ["rise_delta"]),
            H.make_node("Mul", ["rise_delta", "rise_weight"], ["rise_correction"]),
            H.make_node("Add", ["factory_action", "rise_correction"], [output]),
        ]
    )
    m.metadata_props.add(
        key="simulated_rise",
        value="v2-heel15 stand command; host supplies tick 0..35 at 50 Hz, 35 selects original actor",
    )
    onnx.checker.check_model(m)
    onnx.save(m, str(dest))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    old = ort.InferenceSession(str(factory), opts, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(str(dest), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(491)
    errors = []
    for tick in range(END_TICK + 1):
        for _ in range(4):
            obs = rng.normal(size=(1, 61)).astype(np.float32)
            a = old.run(None, {"obs": obs})[0]
            ref = a + (poses[tick] - a) * weights[tick]
            got = new.run(None, {"obs": obs, "rise_tick": np.array([tick], np.int64)})[0]
            errors.append(float(np.max(np.abs(ref - got))))
    if max(errors) > 2e-5:
        raise AssertionError(max(errors))
    return max(errors)

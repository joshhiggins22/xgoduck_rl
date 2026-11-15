"""CPU PPO with the deployed ONNX actor and its observation normalization."""


import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from duck_env import ROOT
from stock_policy import JOINT_NAMES

SOURCE = ROOT / "runtime/xgoduck_sitstand.onnx"


class Normalizer(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 61)
        self.register_buffer("mean", torch.zeros(61))
        self.register_buffer("std", torch.ones(61))

    def forward(self, obs):
        return (obs - self.mean) / self.std


def onnx_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def predict_onnx(session, obs):
    return session.run(None, {session.get_inputs()[0].name: obs[None].astype(np.float32)})[0][0]


def make_ppo(env, rollout, batch, lr, seed, log_std=-2.5):
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=lr,
        n_steps=rollout,
        batch_size=batch,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.97,
        clip_range=0.1,
        target_kl=0.02,
        policy_kwargs=dict(
            features_extractor_class=Normalizer,
            net_arch=dict(pi=[512, 256, 128], vf=[128, 64]),
            activation_fn=nn.ELU,
            log_std_init=log_std,
        ),
        seed=seed,
        verbose=0,
    )
    source = onnx.load(str(SOURCE))
    graph = source.graph
    params = {t.name: torch.from_numpy(onnx.numpy_helper.to_array(t).copy()) for t in graph.initializer}
    nodes = list(graph.node)
    if [n.op_type for n in nodes] != [
        "Sub",
        "Div",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
    ]:
        raise ValueError("Unexpected stock ONNX graph")
    meta = {x.key: x.value for x in source.metadata_props}
    if meta["joint_names"].split(",") != list(JOINT_NAMES):
        raise ValueError("Unexpected stock joint order")
    p = model.policy
    with torch.no_grad():
        p.features_extractor.mean.copy_(params[nodes[0].input[1]].reshape(61))
        p.features_extractor.std.copy_(params[nodes[1].input[1]].reshape(61))
        for dst, node in zip(
            [layer for layer in p.mlp_extractor.policy_net if isinstance(layer, nn.Linear)]
            + [p.action_net],
            [n for n in nodes if n.op_type == "Gemm"],
            strict=True,
        ):
            dst.weight.copy_(params[node.input[1]])
            dst.bias.copy_(params[node.input[2]])
    # Exercise real scale, low-variance channels and joint ordering before training.
    rng = np.random.default_rng(701)
    obs = (
        rng.normal(size=(128, 61)) * p.features_extractor.std.numpy()
        + p.features_extractor.mean.numpy()
    ).astype("float32")
    sess = onnx_session(SOURCE)
    pred = np.stack([predict_onnx(sess, x) for x in obs])
    with torch.no_grad():
        actual = p.get_distribution(torch.from_numpy(obs)).distribution.mean.numpy()
    err = float(np.max(np.abs(pred - actual)))
    if err > 2e-5:
        raise ValueError(f"Stock ONNX warm-start mismatch {err}")
    return model, err


class Actor(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.norm = p.features_extractor
        self.hidden = p.mlp_extractor.policy_net
        self.head = p.action_net

    def forward(self, obs):
        return torch.clamp(self.head(self.hidden(self.norm(obs))), -8.0, 8.0)


def export_onnx(model, path):
    actor = Actor(model.policy).eval()
    torch.onnx.export(
        actor,
        torch.zeros(1, 61),
        str(path),
        opset_version=13,
        input_names=["obs"],
        output_names=["actions"],
        do_constant_folding=True,
    )
    g = onnx.load(str(path))
    onnx.helper.set_model_props(
        g,
        {
            "joint_names": ",".join(JOINT_NAMES),
            "task": "simulation-only standing command",
            "action_scale": "1.0",
        },
    )
    onnx.checker.check_model(g)
    onnx.save(g, str(path))
    sess = onnx_session(path)
    rng = np.random.default_rng(119)
    obs = (rng.normal(size=(96, 61)) * actor.norm.std.numpy() + actor.norm.mean.numpy()).astype(
        "float32"
    )
    err = float(
        max(
            np.max(np.abs(predict_onnx(sess, x) - model.predict(x, deterministic=True)[0])) for x in obs
        )
    )
    if err > 2e-5:
        raise ValueError(f"Export mismatch {err}")
    return err

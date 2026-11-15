"""Check reproducibility, policy order, feedback timing and finite physics on this host."""

from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import mujoco
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from duck_env import DuckSitStandEnv, FEET, DT, DECIMATION, ROOT
from learner import make_ppo, export_onnx
from stock_policy import StockPolicy


def main():
    torch.set_num_threads(1)
    for foot in FEET:
        env = DuckSitStandEnv(foot, "carpet", start="sit")
        actor = StockPolicy(env.model, ROOT / "runtime")

        def replay():
            obs, _ = env.reset(seed=12345)
            actor.reset(env.data)
            trajectory = []
            for step in range(30):
                # Two independent adapter paths must send identical sensor histories and servo targets.
                raw_obs = actor.feedback.obs(actor.raw_action)
                np.testing.assert_array_equal(raw_obs, obs)
                target = actor.targets()
                np.testing.assert_allclose(target, env.actions.targets(actor.raw_action), atol=1e-6)
                obs, _, terminated, truncated, _ = env.advance(
                    target, actor.raw_action, sample_callback=actor.sample
                )
                trajectory.append(obs)
                if terminated or truncated:
                    break
            assert abs(env.data.time - len(trajectory) * DT * DECIMATION) < 1e-9
            return np.stack(trajectory)

        np.testing.assert_array_equal(replay(), replay())
        env.close()
        print(foot, "50 Hz policy, 100 Hz feedback, deterministic reset: OK")
    vec = DummyVecEnv([lambda: DuckSitStandEnv()])
    try:
        model, init_err = make_ppo(vec, 32, 16, 1e-5, 8)
        with TemporaryDirectory() as td:
            export_err = export_onnx(model, Path(td) / "test.onnx")
    finally:
        vec.close()
    print(
        f"MuJoCo {mujoco.__version__}, torch {torch.__version__}. ONNX warm-start max error {init_err:.2g}, export {export_err:.2g}."
    )


if __name__ == "__main__":
    main()

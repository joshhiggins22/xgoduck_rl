"""Train, evaluate and view an XGO standing policy on a CPU."""

import argparse
import json
from pathlib import Path
import time


from duck_env import DuckSitStandEnv, FEET, SURFACES, DT, DECIMATION
from learner import SOURCE, export_onnx, make_ppo, onnx_session, predict_onnx


def evaluate(env, predict, seeds, start):
    rows = []
    for seed in seeds:
        obs, reset = env.reset(seed=seed, options={"start": start})
        if hasattr(predict, "reset"):
            predict.reset(obs)
        total = 0.0
        peak = 0.0
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(predict(obs))
            steps += 1
            total += r
            peak = max(peak, info["upright_streak_s"])
            if term or trunc:
                break
        rows.append(
            dict(
                seed=seed,
                success=bool(not term and info["upright_streak_s"] >= 1.0),
                fell=info["fell"],
                final_upright_s=round(info["upright_streak_s"], 2),
                longest_upright_s=round(peak, 2),
                survival_s=round(steps * DT * DECIMATION, 2),
                final_tilt_deg=round(info["tilt_deg"], 2),
                return_sum=round(total, 2),
                **reset,
            )
        )
    return dict(
        start=start,
        foot=env.foot,
        surface=env.surface,
        episode_s=env.max_steps * DT * DECIMATION,
        successes=sum(r["success"] for r in rows),
        falls=sum(r["fell"] for r in rows),
        episodes=len(rows),
        trials=rows,
    )


def policy(path):
    if path.suffix == ".zip":
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    sess = onnx_session(path)
    if any(i.name == "rise_tick" for i in sess.get_inputs()):
        from rise import RisePolicy

        return RisePolicy(sess)
    return lambda obs: predict_onnx(sess, obs)


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    import torch

    if (
        args.steps <= 0
        or args.workers <= 0
        or args.rollout < 2
        or args.batch < 2
        or args.workers * args.rollout % args.batch
    ):
        raise ValueError("Choose positive steps/workers and a batch size dividing rollout × workers")
    args.out.mkdir(parents=True, exist_ok=False)
    vec = (DummyVecEnv if args.workers == 1 else SubprocVecEnv)(
        [
            lambda: DuckSitStandEnv(args.foot, args.surface, start=args.start)
            for _ in range(args.workers)
        ]
    )
    try:
        if args.resume:
            model = PPO.load(args.resume, env=vec, device="cpu")
            model.set_random_seed(args.seed)
            init_error = None
        else:
            model, init_error = make_ppo(
                vec, args.rollout, args.batch, args.lr, args.seed, args.log_std
            )
        actor_params = [
            *model.policy.mlp_extractor.policy_net.parameters(),
            *model.policy.action_net.parameters(),
            model.policy.log_std,
        ]
        before = torch.cat([p.detach().flatten() for p in actor_params]).clone()
        base_steps = model.num_timesteps

        class Progress(BaseCallback):
            def _on_rollout_start(self):
                if args.curriculum:
                    progress = min(1.0, (self.num_timesteps - base_steps) / (0.70 * args.steps))
                    vec.env_method("set_crouch_depth", 0.35 + 0.65 * progress)
                for p in actor_params:
                    p.requires_grad_(self.num_timesteps - base_steps >= args.freeze)

            def _on_step(self):
                return True

            def _on_rollout_end(self):
                elapsed = self.num_timesteps - base_steps
                if args.checkpoint and (
                    elapsed // args.checkpoint
                    != (elapsed - self.model.n_steps * self.model.n_envs) // args.checkpoint
                ):
                    self.model.save(args.out / f"checkpoint_{self.num_timesteps}.zip")
                print(f"rollout {self.num_timesteps} steps", flush=True)

            def _on_training_end(self):
                for p in actor_params:
                    p.requires_grad_(True)

        started = time.monotonic()
        model.learn(
            total_timesteps=args.steps, callback=Progress(), reset_num_timesteps=not bool(args.resume)
        )
        model.save(args.out / "policy.zip")
        loaded = PPO.load(args.out / "policy.zip", device="cpu")
        parity = export_onnx(loaded, args.out / "policy.onnx")
        change = float((torch.cat([p.detach().flatten() for p in actor_params]) - before).abs().max())
        result = dict(
            foot=args.foot,
            surface=args.surface,
            start=args.start,
            seed=args.seed,
            start_steps=base_steps,
            timesteps=model.num_timesteps,
            wall_s=round(time.monotonic() - started, 2),
            curriculum=args.curriculum,
            rollout=model.n_steps,
            workers=model.n_envs,
            batch_size=model.batch_size,
            source=None if not args.resume else str(args.resume),
            warmstart_max_abs_error=init_error,
            reloaded_export_max_abs_error=parity,
            max_actor_parameter_change=change,
        )
        (args.out / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        vec.close()


def show(args):
    import mujoco as mj

    env = DuckSitStandEnv(args.foot, args.surface, args.start, episode_s=args.seconds)
    obs, _ = env.reset(seed=args.seed)
    predict = policy(args.policy)
    if hasattr(predict, "reset"):
        predict.reset(obs)
    if args.command == "view":
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.distance = 0.48
            v.cam.lookat[:] = [0, 0, 0.11]
            v.cam.elevation = -12
            v.cam.azimuth = 95
            while v.is_running():
                t = time.monotonic()
                obs, _, term, trunc, _ = env.step(predict(obs))
                v.sync()
                time.sleep(max(0, DT * DECIMATION - (time.monotonic() - t)))
                if term or trunc:
                    obs, _ = env.reset(seed=args.seed)
                    if hasattr(predict, "reset"):
                        predict.reset(obs)
    else:
        import subprocess

        cam = mj.MjvCamera()
        cam.distance = 0.48
        cam.lookat[:] = [0, 0, 0.105]
        cam.elevation = -12
        cam.azimuth = 98
        opt = mj.MjvOption()
        opt.geomgroup[3:] = 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "640x480",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.out),
        ]
        with (
            mj.Renderer(env.model, height=480, width=640) as renderer,
            subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc,
        ):
            for step in range(env.max_steps):
                if step % 2 == 0:
                    renderer.update_scene(env.data, camera=cam, scene_option=opt)
                    proc.stdin.write(renderer.render().tobytes())
                obs, _, term, trunc, _ = env.step(predict(obs))
                if term or trunc:
                    break
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg export failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "eval", "view", "record"]:
        p = sub.add_parser(name)
        p.add_argument("--foot", choices=FEET, default="v2-heel15")
        p.add_argument("--surface", choices=SURFACES, default="mixed" if name == "train" else "carpet")
        p.add_argument("--seed", type=int, default=42 if name == "train" else 811)
        p.add_argument(
            "--start",
            choices=["sit", "stand", "mixed"] if name == "train" else ["sit", "stand"],
            default="mixed" if name == "train" else "sit",
        )
        if name == "train":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--steps", type=int, default=8192)
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--rollout", type=int, default=256)
            p.add_argument("--batch", type=int, default=64)
            p.add_argument("--lr", type=float, default=1e-5)
            p.add_argument(
                "--freeze",
                type=int,
                default=512,
                help="Initial critic-only sample count, rounded to rollout boundary",
            )
            p.add_argument("--resume", type=Path)
            p.add_argument(
                "--curriculum",
                action="store_true",
                help="Gradually deepen seated starts; retain full-depth trials",
            )
            p.add_argument(
                "--log-std",
                type=float,
                default=-2.5,
                help="Initial actor exploration, natural log standard deviation",
            )
            p.add_argument(
                "--checkpoint",
                type=int,
                default=32768,
                help="Save intermediate checkpoints at this step interval (0 disables)",
            )
        else:
            p.add_argument("--policy", type=Path, default=SOURCE)
            p.add_argument("--seconds", type=float, default=6.0)
            if name == "eval":
                p.add_argument("--episodes", type=int, default=10)
                p.add_argument("--out", type=Path)
            if name == "record":
                p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        result = evaluate(
            DuckSitStandEnv(args.foot, args.surface, args.start, args.seconds),
            policy(args.policy),
            range(args.seed, args.seed + args.episodes),
            args.start,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}))
    else:
        show(args)


if __name__ == "__main__":
    main()

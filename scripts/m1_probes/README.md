# M1 MuJoCo probes & warm-start prep

CPU-MuJoCo visual-check + warm-start tooling for the XGO-Duck. Runs on the M1 Studio
(`joshhiggins@10.5.0.76`, `~/duck-probe`). MJX cannot train on Apple Silicon, so these
probes diagnose physics and replay policies to SEE behavior rather than infer it from
reward numbers.

## Setup (M1)
```
uv venv --python /opt/homebrew/bin/python3.14 venv
uv pip install --python venv/bin/python mujoco numpy imageio imageio-ffmpeg torch onnx
# model: scp the repo's robot/xgoduck dir to ~/duck-probe/model (scene.xml + assets/)
```

## Scripts
- `settle_probe.py` — hold the STAND keyframe on nominal vs carpet-extreme ground
  (solref 0.02 / 0.06); logs trunk z + ground reaction, renders side-view mp4. Tests
  whether the ground-compliance DR makes standing unreachable.
- `replay_policy.py` — replay a trained rsl_rl checkpoint (`model_*.pt`) in CPU MuJoCo:
  exact 61-D obs + obs-normalizer (handles zero-variance dims) + ELU MLP. Caveat: raw
  scene.xml uses position servos, not the BAM actuators used in training.
- `replay_official.py` — replay the deployed official ONNX (`xgoduck_sitstand.onnx`).
- `warmstart_prep.py` — extract the official ONNX actor -> `official_warmstart.pt`
  (rsl_rl `actor_state_dict`, `infos.warm_start_actor_only=True`) and validate by replay.

## Warm-start training (needs a GPU — vast.ai / HF Jobs)
Place `official_warmstart.pt` at `logs/rsl_rl/<experiment>/warmstart/model_0.pt`, then:
```
uv run train Mjlab-SitStand-Carpet-XgoDuck --agent.resume \
    --agent.load-run warmstart --agent.load-checkpoint model_0.pt <train args>
```
`src/mjlab_microduck/warm_start_hook.py` detects the flag and loads ONLY the actor
(fresh critic / optimizer / iteration). Unit-tested in `tests/test_warm_start_hook.py`.

## Visual-check calibration
Ground posture claims in trunk **z + pitch + roll** (roll matters — pitch alone misses
sideways tips). The duck's natural digitigrade crouch reads as "collapse" to side-view
vision without context; give the vision model the measured angles. See the
`mujoco-m1-gait-training` skill, `references/mjlab-policy-replay.md`.

# XGO Duck · local MuJoCo training lab

This lab runs the V2 duck on a CPU. It trains the zero-command rise and standing hold, plays the factory ONNX models, and lets you change the physical foot and the floor independently. The robot can be off. This copy was built and exercised on an Intel Mac running macOS 15.7.7 with Python 3.12.

**On this Mac**, double-click **Watch Duck.command** to open the interactive MuJoCo viewer. Drag to orbit, scroll to zoom, and close the window to stop. The default shows the 49,152-step PPO candidate rising on simulated carpet with the stock V2 feet. Double-click **Train Duck.command** to reproduce the 49,152-step stock-foot PPO experiment and evaluate its exported actor from a seated start. Each run gets its own directory under `runs/`. Double-click **Watch Heel Cycle.command** to see the 15 mm duck sit down and rise under its separately trained controller. [Side-by-side videos and measured results](RESULTS.md).

**On another Mac or after moving the directory**, unpack the portable archive and run `./setup.sh` from this folder using CPython 3.12. Set `XGO_PYTHON` to the full interpreter path if needed. The installed virtual environment in the original folder uses absolute paths. The archive carries source, pretrained models, robot meshes and results. Python dependencies download during installation. For an exact Intel Mac reinstall, the full tested set is in `requirements-intel-mac.lock`. Neither an Arduino connection nor a cloud account is required.

The [stock-foot robot trial guide](ROBOT-TRIAL.md) includes the model file, the correct runtime slot, backup and restore steps. The November Walk reference at `runtime/walk-4ce3315c.onnx` matches the later feedback archive. [Pinned Walk source](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/4ce3315c86190a632f4d75a2464a4ed31c9af69a/python/xgoduck_walk.onnx).

## Run an experiment

Run the following in Terminal **from the lab folder**. A new `--out` for training protects earlier checkpoints.

```bash
# Watch a rise attempt. macOS GUI playback uses mjpython.
.venv/bin/mjpython lab.py view --foot v2-heel15 --surface carpet --start sit

# Train from the bundled factory SitStand actor. Two CPU workers generate samples.
.venv/bin/python lab.py train --workers 2 --steps 16384 --foot v2-heel15 \
  --surface mixed --out runs/my-heel-test

# Measure the learned rise and the ability to hold upright separately.
.venv/bin/python lab.py eval --policy runs/my-heel-test/policy.onnx \
  --foot v2-heel15 --surface carpet --start sit --episodes 10 --out runs/my-heel-test/rise.json
.venv/bin/python lab.py eval --policy runs/my-heel-test/policy.onnx \
  --foot v2-heel15 --surface carpet --start stand --episodes 10 --out runs/my-heel-test/hold.json

# Continue a trusted local SB3 checkpoint. --steps counts additional samples.
.venv/bin/python lab.py train --workers 2 --resume runs/my-heel-test/policy.zip \
  --steps 8192 --freeze 0 --foot v2-heel15 --surface mixed --out runs/my-heel-test-2

# Compare factory and candidate on identical seeds, both floors and both reset poses.
.venv/bin/python compare.py --candidate runs/my-heel-test/policy.onnx --out results/my-comparison.json

# Record a six-second rollout. Video export also needs ffmpeg on PATH.
.venv/bin/python lab.py record --policy runs/my-heel-test/policy.onnx --foot v2-heel15 \
  --surface carpet --start sit --out runs/my-heel-test/rise.mp4
```

For a rise curriculum, add `--curriculum`. This progressively deepens a portion of the training resets, while retaining complete seated starts. Evaluation always uses the full requested pose. Resume restores the checkpoint's network, optimizer, rollout and batch settings. Supply the desired foot, surface, workers, reset mix and additional training duration. A curriculum on resume starts a fresh schedule. A checkpoint includes the observation normalizer along with the weights. ONNX exports accept raw observations. PPO clips exceptionally large absolute action offsets at 8 rad before the runtime's smoothing and simulated encoder limits.

## What this measures

- **Robot and timing.** XGO link geometry, inertia and HLS1910 fit with vendored BAM. Four 5 ms physics steps per action give 50 Hz decisions. The named gyro and simulated accelerometer go through the deployment feedback filter at 100 Hz. The previous-action observation is the raw 14-value policy output. The deployed leg and head smoothing then converts it to targets. The HLS1910 fit, supply sag, command lag and motor dynamics are approximations. This CPU task uses the fork's full-contact robot without the newer upstream backlash model.
- **Foot geometry.** `upstream` uses the original RL meshes. `v2-stock`, `v2-heel5`, `v2-heel10` and `v2-heel15` register the original hardware or the respective Fusion extension to the same ankle frames. Rigid foot and TPU visuals are separate. Each foot has one convex contact hull; underfoot screw recesses and TPU deformation are absent. The extension includes an estimated mass and inertia change. See [foot registration and mass calibration](v2_feet/README.md).
- **Surfaces.** The carpet condition samples friction and contact softness at reset. Its MuJoCo contact time constant is 35–65 ms, compared with 15 ms on the harder floor. This is an uncalibrated proxy for a compressible floor. At present it does not reproduce spatial changes in pile depth, snagging, or time-dependent fabric deformation.
- **Task and score.** Most training starts are seated. All actors in the basic trainer receive a zero stand command. The separate factory replay supports a sit command and a timed handoff to Walk. A six-second benchmark passes only if the duck avoids the fall threshold and is upright for the final full second: trunk at least 124 mm high, tilt under 15 degrees and both knees within 0.40 rad. A high return or a short peak in height alone does not qualify. A trial starting upright measures the hold. Each floor/foot comparison uses the same seeds and the same starting joint pose, with small initial attitude and rate perturbations. Floor placement accounts for each foot's actual lowest point.
- **Deployment scope.** The factory and PPO actors accept 61 floats and output 14 joint offsets in policy order. The timed heel maneuver also requires a 50 Hz integer counter from its host. These experiments train the standing command only. Sit, transitions to walking, recovery and hardware transfer require their own tests. No candidate has been installed on the duck.

To replay the handoff or a complete factory Sit→Stand cycle:

```bash
.venv/bin/python stock_eval.py --foot v2-heel15 --surface carpet --start sit \
  --trials 8 --seed 1100 --handoff-seconds 1 --json results/handoff.json
.venv/bin/python stock_eval.py --foot v2-stock --surface carpet --start stand \
  --sit-dwell 3 --trials 4 --seed 1100 --json results/commanded-cycle.json
```

To evaluate the learned descent and rise together on held-out carpet starts, run the following from the lab directory.

```bash
.venv/bin/python controlled_cycle.py --surface carpet --seed 3500 --episodes 12
.venv/bin/mjpython view_cycle.py --surface carpet --seed 3500 --cycles 1
```

The [full heel cycle notes](runs/heel-cycle-descent/README.md) include the descent training command and the recorded failure case. The [15 mm maneuver notes](runs/heel-rise-bounded/README.md) include commands to retrain its three poses and evaluate the exported model.

The scale in the lab is 1.0 unless you supply `--scale` to `stock_eval.py`. The pinned upstream architecture document specifies a separate 1.1 SitStand multiplier. The installed controller and settings determine which replay matches your duck. A one-second handoff represents the upstream timed behavior. To reproduce the pose-based switch in the November 4 local controller, replace `--handoff-seconds 1` with `--pose-handoff`. [Matched handoff results](results/HEEL-LENGTH-AND-HANDOFF.md). An optional `--sitstand-onnx runs/stock-49k/policy.onnx` uses the trained stock-foot actor for both Sit and Stand, with factory Walk after the switch. A commanded sit begins upright, keeps the sensor/action history across the command change and qualifies a rise only after a measured seated posture. Use that test when comparing against a video of the real duck.

## Sources and validation

This lab adapts the prior native trainer on [the XGO carpet-standup fork](https://github.com/joshhiggins22/xgoduck_rl/tree/c1091f58a87d635971a249950f429f27a1f25985/sb3_mac). It corrects the decision period, angular-velocity frame, reset state of the motor model, raw/filtered action history, and action-rate reward. The vendored BAM controller matches MuJoCo friction constraints using DOF addresses. Rear contact position is measured from active foot-floor contact points. The warm start reads and verifies the deployed ONNX graph directly. Intermediate PPO checkpoints support longer runs.

The geometry and servo inputs come from [XGO-Duck RL](https://github.com/LuwuDynamics/xgoduck_rl), the actuator model from [BAM](https://github.com/Rhoban/bam), and the runtime math and factory policies from [the Arduino runtime](https://github.com/LuwuDynamics/xgoduck_runtime_arduino). The runtime snapshot here is `b401d85d20bb5bfd0237242a8a5ab6847fc962a2`; the hardware design is [52e237f](https://github.com/joshhiggins22/xgoduck_hardware/tree/52e237f0d53199b7ef697b7ce9e0662e7a662ffe/mods/v2-heel-fusion). License provenance for models and hardware is described by each source. The bundled runtime notices apply to the corresponding upstream files.

[Pollen Robotics' MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) is the parent recipe for the XGO fork. [MicroDuck Lab](https://github.com/jonathanhawkins/microduck-lab) demonstrates native CPU MuJoCo and PPO on a Mac. The [HD1910 sit/stand example](https://github.com/Ley-WL/microduck-rl/blob/main/docs/sitstand-official-scratch-20261021.md) is another relevant experiment with different hardware. A board swap would add driver and sensor integration work. The UNO Q already runs the actor independently of training. [XGO runtime architecture](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/docs/architecture.md).

Run `.venv/bin/python verify_lab.py` for a replay and ONNX parity check, or `.venv/bin/python -m v2_feet.verify robot` for the heel contact fixture. See [experiment results](RESULTS.md) for the trials shipped with this lab.

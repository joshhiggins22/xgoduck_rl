# 15 mm heel: joint-bounded rise maneuver

This is the preferred **simulation** controller for rising from the lab’s seated reset with the V2 15 mm heel. Starting upright uses the factory sit/stand actor. Starting seated adds a learned 0.70-second maneuver, then continues with that same factory actor. Both feet and the carpet proxy use the lab’s standard definitions.

The optimizer searched three poses for hip pitch, knee, ankle and neck pitch. Leg channels are symmetric, and the other seven servo commands stay under factory feedback throughout. The three poses are keyed at 0, 0.14 and 0.36 s and fade into factory feedback between 0.50 and 0.70 s. The search constrained the seven scripted target channels inside the simulated mechanical stops. These limits apply to the scripted poses. They are not a test of physical current draw, servo heating, carpet interaction or impact load.

Cross-entropy search used eight generations of 40 candidates with two CPU workers. It initialized from the earlier unconstrained experiment. Candidate selection used six-second trials on separate seeds (2000–2001). The single exported candidate was then compared to factory on fresh paired seeds 2460–2471. From sitting, the bounded candidate achieved 12/12 sustained rises on hard floor and 12/12 on the carpet proxy. Factory stayed crouched (0/12 in both). Both actors achieved 12/12 in both standing-start conditions. No falls occurred in that paired grid. Results are in `results/heel-rise-bounded-held-out.json`; success requires one full final second upright with no earlier fall. This protocol starts from the simulator’s seated reset. A real Sit → Stand command retains servo, body, IMU and previous-action state.

The portable ONNX includes the complete frozen factory actor and fitted waypoint correction. `rise.RisePolicy.reset(initial_obs)` arms on flexed knees and a stand command. The simulation host supplies the explicit integer `rise_tick` at 50 Hz. The existing single-input hardware loader does not support this additional input. Automatic handoff to Walk is a separate controller behavior and is outside the paired grid above.

From the lab directory:

```sh
.venv/bin/python lab.py eval --policy runs/heel-rise-bounded/rise.onnx --foot v2-heel15 --surface carpet --start sit --episodes 12 --seed 2460
.venv/bin/mjpython lab.py view --policy runs/heel-rise-bounded/rise.onnx --foot v2-heel15 --surface carpet --start sit --seed 2460
.venv/bin/python train_rise.py --out runs/new-bounded-rise --from-json runs/heel-rise-cem/rise.json --joint-bounds --generations 8 --population 40 --seed 338
```

The outputs were independently checked against the frozen actor plus the same pose interpolation at every tick 0–35. Maximum absolute action difference across 144 random 61-value inputs was zero for this export.

## Commanded Sit → Stand on the same 15 mm feet

In a continuous simulation starting upright, the factory actor was given the Sit command for three seconds. A Stand command then re-armed the learned maneuver from the current measured knees while preserving the body, actuator, IMU, smoothing, and previous-action state. Twelve fresh carpet seeds (2610–2621) all failed to establish a deep seated pose during the Sit phase. The torso ended at **125.5–133.3 mm** and the less-flexed knee at **0.129–0.590 rad**. For comparison, the benchmark’s seated reset placed the torso at **99.2–103.8 mm** with both knees at **1.35 rad**. As a check independent of height, no trial sustained both knees above 0.8 rad with tilt below 25° for 0.24 s. The failed phase was the factory sit-down on the longer feet.

A full cycle counted success only if the robot first attained a seated pose. All three conditions scored **0/12 complete cycles**: continuous SitStand, a one-second handoff to Walk, and a switch triggered by knee angle and trunk tilt. Those full-cycle results and the reset-start success above refer to different initial conditions. The host can arm explicitly on a command via `RisePolicy.reset(obs)`; on these incomplete sits, the knee trigger can decline to start the maneuver. See `results/rise-bounded-cycle*.json` and `results/heel-commanded-seated-posture.json`.

```sh
.venv/bin/python rise_cycle.py --episodes 12 --seed 2610 --handoff none
.venv/bin/python rise_cycle.py --episodes 12 --seed 2610 --handoff timer
.venv/bin/python rise_cycle.py --episodes 12 --seed 2610 --handoff pose
```

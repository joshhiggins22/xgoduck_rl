# XGO Duck V2 stand and heel experiments

Start with the stock V2 feet for a duck that struggles to rise from sitting. The Mac-trained stock-foot actor improved the rise on unseen simulated starts and on a commanded Sit → Stand cycle. The extended heel addresses a different failure. It gave the standing duck more margin against a backward push, and it required a different maneuver to rise from a deep crouch.

## Choosing a stock-foot policy for a robot trial

The [stock PPO SitStand](runs/stock-49k/policy.onnx) accepts the original 61-value observation and outputs 14 joint offsets. The November Uno Q Python loader loaded and warmed this file on the Mac. For these comparisons the robot starts upright, receives Sit for three seconds, and then receives Stand. A 0.10-second rearward pulse at the torso begins 0.20 s after Stand. Both controllers keep their sensor history and switch to the same November Walk model when the measured pose is ready.

| SitStand action scale | Backward impulse / mass (N·s/kg) | Factory SitStand | Stock PPO SitStand |
| --- | ---: | ---: | ---: |
| 1.0 | 0.35 | 0/12 | 12/12 |
| 1.0 | 0.75 | 0/12 | 1/12 |
| 1.1 | 0.35 | 12/12 | 12/12 |
| 1.1 | 0.75 | 2/12 | 9/12 |

All 12 paired seeds (7500–7511) achieved a measured sit before Stand and received the full push. A pass requires the final second upright without a fall. These outcomes depend on the timing, force, scale and simulated carpet. The board's installed controller and settings remain to be checked. [Scale 1.0 episodes](results/stock-backward-rise.json) · [Scale 1.1 episodes](results/stock-backward-rise-gain11.json) · [Trial and rollback on the robot](ROBOT-TRIAL.md)

When SitStand was kept active continuously after a standing reset, the PPO also recovered more often at 0.75 N·s/kg. With scale 1.0 it recovered in 10/12 trials versus 0/12 for the factory actor. Both scored 0/12 at 1.0 N·s/kg. The controller usually switches to Walk before a push this late would occur. [Factory standing trials](results/stock-push-sitstand-factory.json) · [PPO standing trials](results/stock-push-sitstand-ppo.json)

To reproduce the 0.35 N·s/kg comparison from the lab directory, run both commands with the same seed, duration, gain and Walk model.

```sh
.venv/bin/python stock_eval.py --foot v2-stock --surface carpet --start stand --sit-dwell 3 --pose-handoff --rise-push .35 --scale 1.0 --seed 7500 --trials 12 --walk-onnx runtime/walk-4ce3315c.onnx
.venv/bin/python stock_eval.py --foot v2-stock --surface carpet --start stand --sit-dwell 3 --pose-handoff --rise-push .35 --scale 1.0 --seed 7500 --trials 12 --walk-onnx runtime/walk-4ce3315c.onnx --sitstand-onnx runs/stock-49k/policy.onnx
```

### Balance after the switch to Walk

The [second Walk candidate](runs/stock-walk-recovery-stress/walk.onnx) is based on the owner's archived Walk actor at revision `4ce3315c`. With stock legs and feet it adds a correction for backward pitching while the movement command is zero. At a 0.90 N·s/kg backward push on simulated carpet, recoveries improved from **0/10 to 8/10** when Walk had been active throughout, and **3/10 to 10/10** when the duck started upright under SitStand and then switched to Walk. In a separate three-second commanded Sit, rise, measured switch and push, recovery improved from **5/8 to 6/8** at the same severity. Factory SitStand was used throughout the sit and rise in this last comparison. The new Walk rescued two seeds and fell on one seed the factory Walk recovered. Keep the two hardware trials separate to identify which change helped. [Run notes](runs/stock-walk-recovery-stress/README.md) · [Continuous Walk](results/stock-walk-recovery-stress-continuous.json) · [Upright handoff](results/stock-walk-recovery-stress-handoff.json) · [Commanded cycle](results/stock-walk-recovery-stress-cycle.json)

Both the November Walk model and this candidate also recovered in 8/8 forward, left and right push trials at each of 0.35 and 0.65 N·s/kg, on the same eight seeds (9500–9507) with zero commanded velocity. The graph preserves the archived model's output for the same observation when the movement command is nonzero. Joystick walking, actual carpets and physical servo loads still need testing on the robot. The earlier [moderate-push Walk candidate](runs/stock-walk-recovery/README.md) regressed in some stationary trials and is included for simulation comparison only. [Directional checks](results/stock-walk-other-directions.json)

## Stock V2 feet

The factory SitStand ONNX initialized the PPO actor. Two CPU workers trained on a mixture of hard and simulated carpet for 49,152 samples in 165 seconds on this Intel Mac. Both actors received identical seeds, servo filtering and timing. A successful six-second episode ended with at least one continuous second upright and no earlier fall.

| Test | Factory SitStand | Mac-trained PPO |
| --- | ---: | ---: |
| Simulated carpet, seated reset, fresh seeds 4100–4119 | 13/20 | 20/20 |
| Simulated carpet, factory Sit for 3 s, then Stand for 6 s, fresh seeds 3100–3111 | 10/12 | 12/12 |
| Hard floor, seated reset, initial seeds 811–820 | 3/10 | 10/10 |
| Simulated carpet, seated reset, initial seeds 811–820 | 4/10 | 10/10 |

In the commanded trials, all 12 ducks reached the measured seated posture before the Stand command. The test carries joint, motor and observation history across that command boundary. The stand actor then controls the robot for the entire six seconds. Walking uses a separate policy. [Fresh-seed episodes](results/stock-fresh-seeds.json) · [Commanded cycles](results/stock-commanded-cycle.json) · [Training record](runs/stock-49k/training.json) · [Side-by-side replay](results/stock-policy-comparison.mp4)

A controller-only change also helped the original factory actors. On a separate set of 12 commanded cycles, a one-second switch from SitStand to Walk passed 8/12. Waiting for measured knee angles and torso tilt to settle passed 12/12. The owner's November 4 controller source already has this logic. The firmware version currently installed on the duck needs verification. [Timing, source details and reproduction](results/HEEL-LENGTH-AND-HANDOFF.md)

Using the trained PPO actor for **both Sit and Stand** followed by factory Walk, the stock duck completed 12/12 cycles with either a fixed one-second switch or the pose-based switch on these same 12 seeds. All 12 reached the seated criterion before the rise in both tests. The Walk actor received a zero velocity command. Actual walking at speed remains untested. [Timed switch](results/stock-ppo-timed-handoff.json) · [Pose-based switch](results/stock-ppo-full-handoff.json)

Most trials use an action multiplier of 1.0, the default in the November 4 local controller source. The vendor architecture document for the bundled runtime describes 1.1 for SitStand. With a one-second Walk handoff and a multiplier of 1.1, both the stock-foot factory actor and PPO completed 12/12 commanded cycles on seeds 5700–5711. On the original 20 fresh seated starts, the factory and PPO actors also both achieved 20/20 at 1.1. The improvement at 1.0 is sensitive to this setting. [Factory seated starts at 1.1](results/stock-factory-rise-gain11.json) · [PPO seated starts at 1.1](results/stock-ppo-rise-gain11.json). On 15 mm feet, the factory policy still achieved 0/12 commanded sits at 1.1. [Factory V2 at 1.1](results/stock-factory-gain11.json) · [PPO V2 at 1.1](results/stock-ppo-gain11.json) · [Factory 15 mm at 1.1](results/heel-factory-gain11.json) · [Vendor controller](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/docs/architecture.md)

## Extended heels

At the highest backward push tested (1.0 N·s/kg over 0.10 s), factory Walk at zero speed recovered in 0/8 trials with stock feet, 3/8 with 5 mm heels, 7/8 with 10 mm heels and 7/8 with 15 mm heels. Each duck had already settled upright. At a smaller 0.35 N·s/kg push, stock and 15 mm both recovered in all 12 trials. The stronger pulse was chosen after testing lighter pushes. [Standing push protocol and trials](results/STANDING-PUSH.md)

In a deep crouch, the 15 mm rear extension points almost straight down relative to the floor. It raises the body at first contact and changes the starting posture for the rise. The stock-foot PPO actor transferred directly to those feet achieved 0/10 carpet rises. Separate 16,384-step and 65,536-step PPO runs using those feet also scored 0/10 rises, although each held an upright reset for 10/10 trials. [Foot geometry](v2_feet/README.md) · [Transfer](results/stock-49k-on-heel.json) · [Longer PPO run](results/heel-curriculum.json)

A second training method searched for three successive joint poses on the 15 mm model. The learned maneuver lasts 0.70 s and blends back into factory SitStand. It rose from a seated reset in 12/12 fresh carpet trials and 12/12 fresh hard-floor trials. Factory SitStand alone scored 0/12 from sitting on each surface. Both held standing starts in all trials. The seven scripted joint targets are bounded by modeled mechanical stops. The earlier unrestricted search is included to reproduce initialization. Its best pose requested a knee target beyond the modeled stop. Use the bounded result for playback. The timed graph needs a second input from the host and is a simulation controller. [Bounded maneuver and training command](runs/heel-rise-bounded/README.md) · [Episode data](results/heel-rise-bounded-held-out.json) · [Side-by-side replay](results/heel-policy-comparison.mp4)

The full Sit command revealed another obstacle. Starting upright on 15 mm feet, the factory Sit command failed to flex both knees into a deep sit in all 12 simulated carpet trials. At the next Stand command the less-flexed knee was 0.13–0.59 rad, compared with 1.35 rad in the seated reset. Even removing the height criterion gave 0/12 measured sits. This separates the ability to rise from the ability to get into the starting pose. [Measured knee angles and height](results/heel-commanded-seated-posture.json) · [Commanded cycles](results/rise-bounded-cycle.json)

A second search trained the **descent** on those same extended feet. It gradually bends the legs for a 1.8-second Sit command, then arms the bounded rise from the actual joint and sensor state. Each pass requires a measured deep sit for the final 0.24 seconds of Sit and a full second upright at the end of the following six seconds. With independent seeds 3500–3511, the full controller completed **11/12 simulated carpet cycles and 12/12 hard-floor cycles**. On carpet seed 3503 the duck tipped during descent at about 1.5 s. The successful trials end with a standing hold. A separate 3.0-second Sit command completed 12/12 carpet cycles on seeds 3700–3711. [Longer commanded sits](results/controlled-long-seat-carpet.json). [Continuous cycle video](results/heel-cycle.mp4) · [Independent episodes](results/heel-cycle-search-held-out.json) · [Fitted descent parameters](runs/heel-cycle-descent/parameters.json)

## Compare with the physical duck

The carpet proxy varies friction and contact softness. TPU compression, carpet pile catching an edge, individual servo behavior, and the actual installed controller can change the result. For a first comparison, use the original feet and film one Sit → Stand from the side on the usual carpet and another on a hard floor. Include the whole foot, knees and torso, and keep recording through the first second of Walk. Whether the duck stalls while folded, tips backward once upright, or falls at the switch determines which simulation result to test against. The supplied actors have been evaluated locally in simulation.

# Rising from a sit with the factory policies

On the simulated carpet, the stock V2 feet rose and held a stand most consistently when the controller waited for the knees and torso to straighten before switching from SitStand to Walk. The 5 mm heel passed two of eight rises with that controller. The 10 mm and 15 mm heels never met the switch criteria on these seeds. This compares the factory policies at zero movement command. A separately trained actor can behave differently.

## Seated starts across four CAD sizes

Each foot received the same eight seeds (1100–1107), including the same initial attitude perturbations and synthetic carpet parameter draws. Floor placement accounted for the foot geometry. Cells give successful six-second trials. Success requires a final continuous second with trunk height ≥124 mm, torso tilt <15°, and both knees within 0.40 rad. A fall or a handoff timeout ends a trial.

| V2 foot | Stay on SitStand | Switch to Walk at 1 s | Switch when pose is ready | Trials meeting switch criteria |
| --- | ---: | ---: | ---: | ---: |
| Stock | 3/8 | 4/8 | 8/8 | 8/8 |
| +5 mm | 0/8 | 0/8 | 2/8 | 5/8 |
| +10 mm | 0/8 | 0/8 | 0/8 | 0/8 |
| +15 mm | 0/8 | 0/8 | 0/8 | 0/8 |

The extensions change floor clearance as the ankle rotates. On the training model, a point displaced 15 mm toward the rear in standing moved about 14.7 mm downward and 2.8 mm forward at the folded joint angles. The 10 and 15 mm feet stayed low on the continuous SitStand controller and fell after the fixed-time switch to Walk. The same feet can recover from a stronger backward push once already upright. [Timed and continuous episode data](../runs/diagnostics/short-heel-sweep.json) · [Pose-based episode data](../runs/diagnostics/pose-handoff.json) · [Backward push results](STANDING-PUSH.md)

## Held-out check on stock V2 feet

These tests used 12 fresh paired seeds (5700–5711), the same simulated carpet range and the same six-second rise horizon. The second row starts upright, runs the factory Sit command for three seconds, then preserves all joint, motor and feedback state into the Stand command. All 12 trials in both controller conditions reached a sustained seated pose before the command changed. The seating check requires both knees flexed >0.8 rad, trunk height <115 mm and tilt <25° over the last 0.24 s of the Sit command.

| Starting protocol | Switch at 1 s | Switch when pose is ready | Pose-based switch time |
| --- | ---: | ---: | ---: |
| Reset in seated configuration | 4/12 | 12/12 | 0.36 s |
| Command Sit, then Stand | 8/12 | 12/12 | 0.36–0.38 s |

The switch check reproduces the thresholds in the owner’s November 4 local controller revision. At 50 Hz, it requires three consecutive fresh samples with gravity-derived tilt ≤15° and both measured knee angles within 25°. Stand times out at 3 s. The simulation ends on timeout. The local controller releases the motors. The installed firmware and the robot’s physical carpet performance were unverified. These trials vary seeds within one contact model. [Held-out episodes and ONNX hashes](../runs/diagnostics/pose-handoff-holdout.json)

## Reproduce from the lab folder

The first command tests the pose-based switch on the 5 mm CAD design. The next two compare the commanded cycle on stock V2 feet with paired held-out seeds.

```sh
.venv/bin/python stock_eval.py --foot v2-heel5 --surface carpet --start sit \
  --pose-handoff --seed 1100 --trials 8 --seconds 6
.venv/bin/python stock_eval.py --foot v2-stock --surface carpet --start stand \
  --sit-dwell 3 --handoff-seconds 1 --seed 5700 --trials 12 --seconds 6
.venv/bin/python stock_eval.py --foot v2-stock --surface carpet --start stand \
  --sit-dwell 3 --pose-handoff --seed 5700 --trials 12 --seconds 6
```

Use the other CAD foot names from the table with the first command (`v2-stock`, `v2-heel10`, `v2-heel15`). Omit the handoff flag to continue on SitStand. Add `--json FILE` to save episode data.

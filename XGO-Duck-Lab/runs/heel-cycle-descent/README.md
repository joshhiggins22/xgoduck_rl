# 15 mm heel commanded sit and stand

The simulated duck starts standing on the 15 mm V2 feet. A Sit command runs the fitted, symmetric descent; the Stand command starts the separately fitted joint-bounded rise from the measured crouch. The body, actuator filters, motor history and IMU state carry across both commands. The demonstration keeps the factory SitStand actor active after recovery.

At 50 Hz, the descent interpolates from home to a first pose at 0.6 s and a second pose at 1.2 s. It holds the second pose until the Stand command. The first eight entries of `parameters.json` specify hip pitch, knee, ankle and neck pitch relative to home, in radians, for those two poses. The last two entries adjust hip pitch and ankle using the fore-aft component of filtered gravity. The harness mirrors the pitch-chain targets across the legs, interpolates hip roll and maintains head pitch at home. Other channels use the factory actor's output.

## Results

The nominal protocol issues Stand after 1.8 seconds, then runs for six seconds. At the switch the duck must have spent 12 consecutive ticks (0.24 s) below 115 mm torso height, below 25° tilt, with both knees flexed beyond 0.8 rad. The final upright check requires at least one second continuously above the lab's 124 mm height threshold, below 15° tilt, with both knees below 0.4 rad. Any fall ends the trial. These are the same criteria used to test factory descent.

| Surface and command | Independent seeds | Complete cycles |
| --- | --- | ---: |
| Hard floor, 1.8 s Sit | 3500–3511 | 12/12 |
| Carpet proxy, 1.8 s Sit | 3500–3511 | 11/12 |
| Carpet proxy, 3.0 s Sit | 3700–3711 | 12/12 |

At Stand in the eleven successful nominal carpet trials, the torso was 95.0–96.5 mm high, the less-flexed knee measured 1.287–1.303 rad, and tilt was 1.20–2.40°. Those trials finished with 5.20–5.24 seconds continuously upright. On carpet seed 3503 the duck fell during descent, 1.48 s after Sit (zero-based tick 73). Its torso reached 67.7 mm and tilted 67.4°. The three-second trial uses different initial conditions, so it cannot establish that a longer dwell fixes that failure. Check `results/controlled-cycle-hard.json`, `results/controlled-cycle-carpet.json` and `results/controlled-long-seat-carpet.json` for individual trials. The fit's per-seed outputs, including the failed descent's pose, are in `results/heel-cycle-search-held-out.json`.

The trainer sampled 36 trajectories per generation for seven generations. It trained on one seed per generation (3100–3106), alternating hard floor and carpet. It evaluated six finalists independently on seeds 3300–3301 on both surfaces, selecting the one with the most completed cycles and then the highest worst-case score. The winner passed all four validation trials. It evaluated the selected parameters on the 24 nominal held-out trials above. The carpet uses randomized MuJoCo contact parameters. Hardware validation and the transition to walking remain separate experiments.

## Reproduce

From the lab directory, use the installed native Python environment.

```sh
.venv/bin/python controlled_cycle.py --surface hard --seed 3500 --episodes 12
.venv/bin/python controlled_cycle.py --surface carpet --seed 3500 --episodes 12
.venv/bin/python controlled_cycle.py --surface carpet --sit-seconds 3 --seed 3700 --episodes 12
.venv/bin/python train_descent.py --out runs/new-descent
.venv/bin/mjpython view_cycle.py --surface carpet --seed 3500
```

To record one trial, pass an observer to `controlled_cycle.evaluate(parameters, surface, [seed], observer=callback)`. The callback receives `(env, tick)` immediately before each 50 Hz step. Stand activates at tick 90 under the nominal command.

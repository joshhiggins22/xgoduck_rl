# Stock V2 Walk recovery under stronger backward pushes

The base is the original archived Walk actor from revision `4ce3315c`, bundled as `runtime/walk-4ce3315c.onnx`. This candidate takes a single 61-float observation and returns 14 joint offsets. It loads through the November 4 Python runtime's original ONNX loader. The two models' SHA-256 digests are recorded in the comparison JSONs. The candidate digest is `66d4f875a140c2bfed31d232528cb2b45404cbb24acf3cd183f5fc1bc9cbfd92`.

At zero forward, lateral and yaw command, the actor estimates impending backward tilt from the fore-aft component of filtered gravity and 0.25 s of pitch rate. For a projected signal greater than 0.0576 it scales up a correction, reaching full strength at 0.1857. At full strength it adds +0.031 rad hip pitch, +0.380 rad knee and −0.378 rad ankle on the left leg, mirrored on the right, and +0.190 rad neck pitch. The largest leg correction is 21.8° before the normal servo action filter. It leaves the archived action intact for nonzero twist commands and for a projected signal in the inactive region. The host's Walk action history continues to feed back into the actor.

The stronger search used seven generations of 30 candidate corrections. Each was tested after a SitStand-to-Walk handoff and from continuous Walk on a different seed at 0.70–1.00 N·s/kg. Six validation scenarios per startup mode paired a fresh seed with one of six impulse levels spanning 0.55–1.00. Selection maximized recoveries in the worst startup mode, then overall recoveries, then score. The selected candidate recovered 11/12 validation trials. The archived actor recovered 5/12. Final evaluation used the exported ONNX, fresh seeds, stock V2 geometry, randomized carpet contact, the standard actuator model, and an action multiplier of 1.0. A pulse lasts 0.10 s. Recovery requires no fall and one full final second upright.

The export was checked on 450 synthetic observations. All 150 sampled nonzero-command and 150 benign inputs produced the same outputs as the archived actor, bit for bit. The maximum absolute error against the specified correction was 5.97e−8 rad. The other 150 random states exercised the correction. Equal outputs given identical commanded-walking observations do not establish identical trajectories after a previous intervention.

## Independent carpet results

The first two protocols use ten fresh seeds (8400–8409). In each cell, the left count is the archived Walk actor and the right count is the stress-trained export. Each push level reuses the same seeds for matched comparisons.

| Backward impulse / mass (N·s/kg) | SitStand → Walk, from upright | Continuous Walk, from upright |
| ---: | ---: | ---: |
| 0.00 | 10 → 10 | 10 → 10 |
| 0.55 | 10 → 10 | 10 → 10 |
| 0.75 | 8 → 10 | 5 → 10 |
| 0.90 | 3 → 10 | 0 → 8 |
| 1.00 | 2 → 10 | 0 → 6 |

At 0.90, the median peak tilt fell from 66.2° to 8.6° after handoff and from 66.8° to 13.7° in continuous Walk. On these two protocols the candidate rescued 36 trials and preserved every archived-actor success. Its continuous-Walk runs still included two falls at 0.90 and four falls at 1.00.

A third protocol began with a real three-second Sit command to the archived SitStand actor. Every duck attained a deep seat before Stand. The state-based handoff selected Walk 0.36–0.38 s after Stand and the push began one second after that switch. Across eight fresh seeds (8500–8507), the archived actor and candidate recovered 8/8 and 8/8 without a push, 7/8 and 8/8 at 0.75, and 5/8 and 6/8 at 0.90. At 0.90 two original falls became recoveries, one original recovery (seed 8503) became a fall and one seed fell with either actor. This comparison preserves the state of the simulator and IMU through the command sequence. The Walk filters reset at handoff, as on the archived controller.

A separate check applied forward, left and right pulses at 0.35 and 0.65 N·s/kg on seeds 9500–9507. Each actor recovered 8/8 at each of the six combinations. See `results/stock-walk-other-directions.json`.

Individual outcomes are in `results/stock-walk-recovery-stress-handoff.json`, `results/stock-walk-recovery-stress-continuous.json` and `results/stock-walk-recovery-stress-cycle.json`. These are simulated straight-back pulses of known duration on a carpet proxy. Real-floor traction, fall response, actuator loading, and policy behavior under other disturbances require physical validation. This experiment made no changes to the duck.

## Reproduce on the Mac

Start from the lab directory.

```sh
.venv/bin/python walk_recovery.py fit-stress --run runs/retrained-walk-stress
.venv/bin/python walk_recovery.py evaluate --run runs/stock-walk-recovery-stress --seed 8400 --episodes 10 --impulses 0 .55 .75 .9 1.0 --out results/replay-walk-handoff.json
.venv/bin/python walk_recovery.py evaluate --run runs/stock-walk-recovery-stress --continuous --seed 8400 --episodes 10 --impulses 0 .55 .75 .9 1.0 --out results/replay-walk-continuous.json
.venv/bin/python walk_recovery.py evaluate --run runs/stock-walk-recovery-stress --full-cycle --seed 8500 --episodes 8 --impulses 0 .75 .9 --out results/replay-walk-cycle.json
```

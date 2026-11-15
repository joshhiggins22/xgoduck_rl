# Standing on simulated carpet under a backward push

The unmodified factory Walk ONNX received a zero movement command from an upright reset. Each pulse applied a constant backward horizontal force at the trunk center of mass for 0.10 s, along the heading at push onset. Column headings below give **impulse divided by total robot mass** (N·s/kg, numerically m/s). Consequently the 1.0 level applied about 7.98 N to stock or 8.16 N to the 15 mm model. These are fixed simulation challenges, not physical estimates of a carpet-induced disturbance.

Each trial gets 2 s to settle, must meet the lab’s upright definition throughout the final 0.4 s before the pulse, and then gets 3 s after the pulse to recover. Success requires no fall and continuous upright posture for at least the final 1 s. The same seeds yield the same carpet contact parameters and reset perturbations across feet. Each simulated heel includes its estimated mass change. Raw JSON records pre-push falls or failures to settle separately from falls after the pulse.

## Initial force range

Seeds 2100–2111. Cells show recoveries / all trials. See the [episode-level data](standing-push-walk.json).

| Foot | 0.00 | 0.10 | 0.22 | 0.35 |
| --- | ---: | ---: | ---: | ---: |
| v2-stock | 12/12 | 12/12 | 12/12 | 12/12 |
| v2-heel15 | 12/12 | 12/12 | 12/12 | 12/12 |

## Higher forces, fresh paired seeds

Seeds 2200–2207. Cells show recoveries / all trials. See the [episode-level data](standing-push-stress.json).

| Foot | 0.00 | 0.50 | 0.75 | 1.00 |
| --- | ---: | ---: | ---: | ---: |
| v2-stock | 8/8 | 8/8 | 8/8 | 0/8 |
| v2-heel15 | 8/8 | 8/8 | 8/8 | 7/8 |

## All four heel lengths at the high-stress setting

Seeds 2200–2207. Cells show recoveries / all trials. See the [episode-level data](standing-push-sizes.json).

| Foot | 0.00 | 1.00 |
| --- | ---: | ---: |
| v2-stock | 8/8 | 0/8 |
| v2-heel5 | 8/8 | 3/8 |
| v2-heel10 | 8/8 | 7/8 |
| v2-heel15 | 8/8 | 7/8 |

The size comparison deliberately reuses the stronger-pulse seeds. The stock and 15 mm rows in the last two tables are repeated trials, not independent replication. The largest pulse was selected for the size comparison after observing it separated stock from 15 mm. In the initial comparison at 0.35 N·s/kg, both designs recovered in all 12 trials; median peak trunk tilt after push onset was 7.85° with stock feet and 1.39° with the 15 mm feet.

These results concern one policy holding a stand in a rigid-foot contact approximation on a synthetic compliant plane. Successful sit-to-stand, backward stepping, and physical performance on a particular carpet require separate tests.

## Reproduce from the lab root

```sh
.venv/bin/python heel_push.py --json results/standing-push-walk.json
.venv/bin/python heel_push.py --seed 2200 --trials 8 --impulses 0 .50 .75 1.00 --json results/standing-push-stress.json
.venv/bin/python heel_push.py --seed 2200 --trials 8 --candidate v2-heel5 v2-heel10 v2-heel15 --impulses 0 1.0 --json results/standing-push-sizes.json
```

Use `--geometry-only` to disable added heel mass or `--policy sitstand` for a separate stand-hold controller comparison.

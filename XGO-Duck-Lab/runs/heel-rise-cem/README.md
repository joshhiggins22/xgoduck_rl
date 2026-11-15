# Seated-rise search on the 15 mm heel

This simulation experiment holds the stock sit/stand ONNX weights fixed. Cross-entropy search fitted three symmetric poses for the two hip pitch, knee and ankle pairs, and neck pitch. At 50 Hz the poses are keyed at 0, 0.14, and 0.36 seconds. During 0.50–0.70 s, the controller fades those seven channels back to the stock policy. The other seven channels use stock feedback for the whole maneuver.

The saved ONNX contains the original actor and the learned trajectory. Its inputs are the raw 61-value observation and an integer `rise_tick`. `rise.RisePolicy.reset(initial_obs)` examines the knee angles at a fresh stand-command reset. A seated start arms the maneuver; an upright start and nonzero commands select the original actor. The explicit simulation host supplies the 50 Hz tick. After 0.70 s the controller remains in the stock sit/stand actor. It can initiate the maneuver again at a subsequent environment reset. Device integration would need to trigger this host state on the real sit-to-stand transition and test the handoff to walking.

This first experiment allows offsets up to ±2 radians. Its selected first knee target is about 109°, above the modeled 90° stop, and its final ankle target also exceeds the modeled range. MuJoCo's mechanical constraints enforce travel in the simulation, and the controller may push against a stop. Keep this model on the simulation side.

At six seconds the exported graph passed 12/12 seated rises on both the simulated hard floor and carpet proxy on fresh seeds 2300–2311. The stock actor achieved 0/12 in both seated conditions. Both actors held 12/12 standing starts on both surfaces. Success requires the last full second upright with no fall. See `results/heel-rise-held-out.json` and `rise.json`.

From the lab directory:

```sh
.venv/bin/python lab.py eval --policy runs/heel-rise-cem/rise.onnx --foot v2-heel15 --start sit --surface carpet --seed 2300 --episodes 12
.venv/bin/mjpython lab.py view --policy runs/heel-rise-cem/rise.onnx --foot v2-heel15 --start sit --surface carpet --seed 2300
.venv/bin/python train_rise.py --out runs/new-rise-experiment --seed 149
```

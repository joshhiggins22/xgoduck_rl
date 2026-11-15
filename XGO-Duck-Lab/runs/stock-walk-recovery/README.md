# Stock V2 Walk recovery, moderate-pulse experiment

This experimental single-input ONNX starts with the owner's archived Walk actor from revision 4ce3315c. With the movement command at zero, it uses the filtered fore-aft gravity component and pitch rate to detect backward tipping. At full activation it adds at most 0.205 rad of hip pitch, 0.098 rad of knee, 0.333 rad of ankle and 0.171 rad of neck pitch relative to the archived actor. The two legs mirror each other. When any twist command is nonzero, all 14 model outputs equal the archived actor on the same observation. It also outputs the original action while the projected backward tilt signal stays below its threshold.

A seven-generation cross-entropy search trained on impulses of 0.38–0.50 N·s/kg, with separate validation seeds 6200–6201. The exported graph uses the standard 61-input/14-output float interface, and loaded in the November 4 controller's Python policy loader on the Mac. Among 450 synthetic comparison observations, all 150 nonzero-command and 150 benign-state cases were bit-identical to the archived Walk actor. Peak error against the intended formula was below 0.000001 rad. For both actors, the robot's state evolves over time, so equality for identical input does not by itself establish equal physical walking performance.

On 12 held-out carpet trials per severity (6300–6311), with a one-second SitStand-to-Walk handoff from standing and a zero-velocity command, both actors recovered 12/12 at 0, 0.35 and 0.55 N·s/kg. At 0.75 the original recovered 8/12 and the candidate 9/12. At 0.90 the counts were 1/12 and 4/12. At 1.00 each recovered once. In a separate continuous-Walk test on seeds 7400–7407 the candidate regressed, scoring 6/8 versus the original 8/8 at 0.55 and 3/8 versus 4/8 at 0.75. At 1.00 both scored 0/8. This first graph is retained as a simulation experiment. See `results/stock-walk-recovery-held-out.json`, `results/stock-archived-walk-push.json` and `results/stock-updated-walk-push.json`.

From the lab directory:

```sh
.venv/bin/python walk_recovery.py evaluate --episodes 12 --seed 6300 --out results/replay-walk.json
.venv/bin/python walk_recovery.py fit --run runs/new-moderate-walk
```

The original actor is bundled as `runtime/walk-4ce3315c.onnx`, SHA-256 `1fc079512fe3fe23e274e7a9d8c9ad007e88439f0385ecf20c9596e57886249f`. This export has SHA-256 `301d88104823a5e6697f3384c475645e101d63c4a58b7c36bc8370da0fae8b45`. The fitted parameters are in `parameters.json`.

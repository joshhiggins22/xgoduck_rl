# v0.3.7 forward Walk for V2 feet with 5 mm heels

[`walk.onnx`](walk.onnx) is the Walk actor trained on an M1 Max MacBook Pro for XGO-Duck V2 feet with 5 mm heel extensions. It uses the Arduino v0.3.7 Walk interface, with 61 observations and 14 joint actions. The v0.3.7 runtime loader accepted the export and completed its 12 warm-up inferences.

The base was the stock v0.3.7 Walk actor with a forward-command correction fitted at 0.10, 0.15 and 0.20 m/s. PPO then learned bounded corrections to the ten leg actions. This export comes from seed 37409 at 258,048 PPO steps. It passed its held-out comparison with the tuned base.

| Model | SHA-256 |
| --- | --- |
| Stock v0.3.7 Walk | `1fc079512fe3fe23e274e7a9d8c9ad007e88439f0385ecf20c9596e57886249f` |
| Command-corrected base | `8577fcb99d0f64a43ba1c98612a7ab945cba025f4035bd763f251e461b19c297` |
| This Walk export | `15e8b8ea6fb69a98fdf3c028c3899859bfd2416bca6b8e2cea0b2ce409d5233c` |

## 50 m simulation

Both models received the same 0.20 m/s forward command and heading hold on six paired starts per floor (seeds 163100–163111). The simulation used 5 mm heels, a calibrated IMU, and a 1.26 s neutral-command period after crossing 50 m forward. Both models reached the finish and stopped upright in all 12 trials.

| Surface | Mean forward speed, stock → this actor | Mean peak sideways distance from the start line, stock → this actor |
| --- | ---: | ---: |
| Hard floor | 0.166 → 0.234 m/s (+41%) | 3.14 → 5.21 m (+66%) |
| Simulated carpet | 0.188 → 0.259 m/s (+38%) | 3.95 → 2.60 m (−34%) |

The actor finished faster in every pair. Its peak sideways deviation fell in five of six carpet pairs and rose in five of six hard-floor pairs. Physical speed, tracking and stopping on the robot remain unverified for this export.

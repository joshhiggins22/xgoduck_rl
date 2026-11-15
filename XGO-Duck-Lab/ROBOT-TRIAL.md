# Trying a stock-foot model on the Uno Q

The stock-foot SitStand candidate is [runs/stock-49k/policy.onnx](runs/stock-49k/policy.onnx). It occupies **only** the `xgoduck_sitstand.onnx` slot. The controller calls this file for both Sit and Stand, then hands balance to the separate Walk model. With the Walk weights from the November 12 archive, both this candidate and factory SitStand completed 12/12 unperturbed simulated carpet cycles. [Archived Walk trials](results/archived-walk-handoff.json) The simulation uses its own sensor and contact models. Hardware fit, motor load and carpet behavior still need a physical trial. The 15 mm heel maneuver has a different host interface.

## Identify and save the installed pair

The November 4 repair log recorded the Linux host as `xgoduck1`, project `/home/arduino/ArduinoApps/archive` and Python container `archive-main-1`. The project README documents Arduino App Lab **Run** and the control interface on port 9527. Verify the board address, active project and container in your App Lab session. Press **Torque off** with the duck supported. On the board, inspect the running containers and their mounts. Substitute your active container and source project below if they have changed.

```sh
docker ps --format '{{.Names}} {{.Ports}}'
duck_container=archive-main-1
docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' "$duck_container"
duck_project=/home/arduino/ArduinoApps/archive
```

From a shell on the board, create a fresh backup outside the project. Keep its printed path. The factory SitStand in both local snapshots began with SHA-256 `0edc3518`. Two different Walk networks appear under the same filename: the vendor and early board capture begin with `5eaae52c`, and the owner’s November 12 feedback archive begins with `1fc07951`. The weights differ. The lab includes the [November 12 Walk network](runtime/walk-4ce3315c.onnx) for simulated comparison. The backup of the *installed pair* is the rollback source.

```sh
duck_backup=$(mktemp -d "$HOME/duck-model-backup.XXXXXXXX") &&
  cp -p "$duck_project/python/xgoduck_sitstand.onnx" "$duck_project/python/xgoduck_walk.onnx" "$duck_backup/" &&
  (cd "$duck_backup" && sha256sum xgoduck_sitstand.onnx xgoduck_walk.onnx > originals.sha256 && sha256sum -c originals.sha256) &&
  printf 'Save this backup path: %s\n' "$duck_backup"
```

The newer local controller reports `sitstand.stand_timeout_s=3` and `sitstand.action_scale` in `/api/status`. It hands off when both knees and the torso are upright, normally with scale 1.0. An earlier capture uses a 1 s timer and fixed scale 1.1. Record the installed settings and the stock Sit → Stand behavior before swapping either file. Use the same carpet, original V2 feet, scale and smoothing for the comparison. [Simulated timing and gain comparisons](RESULTS.md)

## Replace SitStand

From the lab folder **on the Mac**, substitute the board’s current SSH address and upload the candidate to the board account’s home directory.

```sh
duck_board=arduino@xgoduck1
shasum -a 256 runs/stock-49k/policy.onnx
scp runs/stock-49k/policy.onnx "$duck_board:duck-sitstand-trial.onnx"
```

On the board, in the same shell as the backup, verify and install the staged file. A temporary sibling and rename keep the active filename complete. The historical project reloads models on a Python container restart. If App Lab manages a different project or container on your unit, use its Stop/Run procedure for that active project.

```sh
duck_slot=xgoduck_sitstand.onnx
duck_hash=007e81ebcaec0a262490e153a3ea4e09fed9a3183bf7d64e6a512f5be333f590
duck_staged="$HOME/duck-sitstand-trial.onnx"
install_duck_model() (
  set -e
  test -f "$duck_backup/originals.sha256"
  (cd "$duck_backup" && sha256sum -c originals.sha256)
  printf '%s  %s\n' "$duck_hash" "$duck_staged" | sha256sum -c -
  duck_tmp=$(mktemp "$duck_project/python/.model-trial.XXXXXXXX")
  trap 'rm -f "$duck_tmp"' EXIT
  cp "$duck_staged" "$duck_tmp"
  chmod --reference="$duck_project/python/$duck_slot" "$duck_tmp"
  mv "$duck_tmp" "$duck_project/python/$duck_slot"
  printf '%s  %s\n' "$duck_hash" "$duck_project/python/$duck_slot" | sha256sum -c -
  docker restart "$duck_container"
)
install_duck_model
```

Open the control page. Verify **Inference only**, motors unloaded, 15/15 servos and IMU online before enabling a trial. The model name in the UI stays the same. Check the bytes loaded from the container mount using its actual path, for example `docker exec "$duck_container" sha256sum "/actual/container/python/$duck_slot"`, substituting the mapped Python directory. With the duck supported and **Torque off** in reach, test one stationary Sit → Stand cycle. Observe whether both knees straighten, whether it hands over to Walk, and whether it holds after handoff. If it tips, binds or times out, support it and unload. Record any missing servo IDs, supply voltage, handoff and timeout messages.

## Restore the tested slot

With the duck supported and torque off, reuse the same board shell. Set `duck_slot` to the model being restored. If reconnecting, first set `duck_backup`, `duck_project` and `duck_container` to the previously recorded values. The rollback verifies the installed pair against the original hashes.

```sh
restore_duck_model() (
  set -e
  (cd "$duck_backup" && sha256sum -c originals.sha256)
  duck_tmp=$(mktemp "$duck_project/python/.model-restore.XXXXXXXX")
  trap 'rm -f "$duck_tmp"' EXIT
  cp -p "$duck_backup/$duck_slot" "$duck_tmp"
  mv "$duck_tmp" "$duck_project/python/$duck_slot"
  (cd "$duck_project/python" && sha256sum -c "$duck_backup/originals.sha256")
  docker restart "$duck_container"
)
duck_slot=xgoduck_sitstand.onnx
restore_duck_model
```

Confirm **Inference only** and feedback after the restart. The same process works for `duck_slot=xgoduck_walk.onnx` after a Walk-only trial.

## If the duck gets upright and then tips backward

Once Stand switches to Walk, the Walk model controls balance even with the joystick centered. A second [stock-foot Walk candidate](runs/stock-walk-recovery-stress/walk.onnx) starts from the November archive's Walk weights and adds a response when the duck begins pitching backward at zero movement command. It uses the same single-input interface. The simulation results and failure cases are in the [Walk experiment notes](runs/stock-walk-recovery-stress/README.md). Physical walking and recovery remain unverified.

Try this file in a separate session with the original SitStand restored. Compare the installed Walk SHA-256 against its training base, `1fc079512fe3fe23e274e7a9d8c9ad007e88439f0385ecf20c9596e57886249f`. If they differ, the swap would also change the underlying gait. Keep the installed Walk for the first SitStand trial. With torque off, make a new original-pair backup using the steps above.

From the Mac lab directory, stage the Walk file after verifying its digest.

```sh
duck_board=arduino@xgoduck1
shasum -a 256 runs/stock-walk-recovery-stress/walk.onnx
scp runs/stock-walk-recovery-stress/walk.onnx "$duck_board:duck-walk-trial.onnx"
```

In the board shell with the backup path and functions above, select only the Walk slot. This checks the installed base before changing it.

```sh
duck_slot=xgoduck_walk.onnx
duck_staged="$HOME/duck-walk-trial.onnx"
duck_hash=66d4f875a140c2bfed31d232528cb2b45404cbb24acf3cd183f5fc1bc9cbfd92
printf '%s  %s\n' 1fc079512fe3fe23e274e7a9d8c9ad007e88439f0385ecf20c9596e57886249f "$duck_project/python/$duck_slot" | sha256sum -c - && install_duck_model
```

Check the restarted container and feedback as above. With support available, try a stationary stand with the joystick centered. Record whether any tip starts before or after the switch. Restore using `duck_slot=xgoduck_walk.onnx` and `restore_duck_model` with torque off.

"""Use the shipped Run gravity path during a simulation of one selected Run actor."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch


@contextmanager
def run_policy_trim(plant, actor, enabled=True):
    """Enter before any apply_mismatch context so both filters see the same sensors."""
    if not enabled:
        yield
        return
    from walk_bench import _runtime
    _, heading, _ = _runtime()
    from run_gravity import RunGravity
    physical = plant.feedback._filter
    trim = RunGravity()
    trim.set_imu_filters(gyro_alpha=physical.gyro_alpha, gravity_tau_s=physical.gravity_tau_s)
    update, reset, infer, command = physical.update, physical.reset, actor.infer, heading.command
    requested = None

    def reset_filters():
        trim.reset()
        return reset()

    def update_filters(*args, **kwargs):
        packet = update(*args, **kwargs)
        if physical.correction is None:
            trim.update(*args, **kwargs)
        return packet

    def heading_command(hold, walking, twist):
        nonlocal requested
        requested = tuple(twist)
        return command(hold, walking, twist)

    def run_infer(grav, gyro, angles, vels, cmd, *args, **kwargs):
        observed = trim.select(physical.correction is None, requested if requested is not None else cmd, grav)
        return infer(observed, gyro, angles, vels, cmd, *args, **kwargs)

    with ExitStack() as scope:
        scope.enter_context(patch.object(physical, 'reset', reset_filters))
        scope.enter_context(patch.object(physical, 'update', update_filters))
        scope.enter_context(patch.object(heading, 'command', heading_command))
        scope.enter_context(patch.object(actor, 'infer', run_infer))
        yield

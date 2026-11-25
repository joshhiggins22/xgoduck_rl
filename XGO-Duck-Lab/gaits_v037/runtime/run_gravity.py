"""Policy-only forward Run trim for v0.3.7 when no saved IMU calibration exists."""
from proc import SignalFilter


class RunGravity(SignalFilter):
    """A separate receive-side filter. Only its gravity vector reaches the Run actor."""

    def reset(self):
        super().reset()
        self.command_acc_x_bias_ms2 = 0.
        self._release = False
        self.gravity = None

    def correct_imu(self, acc, gyro):
        # Keep the release's uncalibrated board-to-body transform on both streams.
        acc, gyro = super().correct_imu(acc, gyro)
        return [acc[0] + self.command_acc_x_bias_ms2, acc[1], acc[2]], gyro

    def update(self, *args, **kwargs):
        values = super().update(*args, **kwargs)
        self.gravity = values['grav'] if values['imu_fusion_ready'] else None

    def select(self, eligible, twist, physical_gravity):
        vx, vy, wz = twist  # Operator command before automatic heading correction.
        if eligible and .15 <= vx <= .20 and vy == 0. and wz == 0.:
            self.command_acc_x_bias_ms2 = 1.5
            self._release = True
        else:
            self.command_acc_x_bias_ms2 = 0.
            # A straight Run releasing to neutral retains its normal filter decay.
            # Walk, recovery, other gaits, and intentional turns use primary gravity.
            self._release = self._release and eligible and vx == vy == wz == 0.
        return self.gravity if self._release and self.gravity is not None else physical_gravity

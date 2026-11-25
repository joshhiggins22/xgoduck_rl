"""Flat hardwood and a spatially varying compliant residential-carpet proxy."""
import numpy as np


HARDWOOD_IMPEDANCE = (.93, .985, .0008, .5, 2.)
CARPET_IMPEDANCE = (.45, .95, .007, .5, 2.)


class CarpetField:
    """Footprint-scale effective material at world XY, constant throughout an episode."""

    def __init__(self, rng):
        periods = rng.uniform([.16, .32], [.27, .58])
        angles = rng.uniform(0., 2*np.pi, 2)
        self.waves = (2*np.pi/periods)[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        self.phases = rng.uniform(0., 2*np.pi, 2)
        self.tau = rng.uniform(.056, .064)
        self.mu = rng.uniform(.73, .85)

    def at(self, xy):
        a = np.asarray(xy) @ self.waves.T + self.phases
        s = np.sin(a)
        softness = .6*s[..., 0] + .4*s[..., 1]
        grip = .5*np.cos(a[..., 0]) + .5*s[..., 1]
        return self.tau + .012*softness, .007 + .0015*softness, self.mu + .06*grip

    def update_pairs(self, model, pair_ids, xy):
        tau, width, mu = self.at(xy)
        model.pair_solref[pair_ids, 0] = tau
        model.pair_solimp[pair_ids, 2] = width
        model.pair_friction[pair_ids, :2] = mu[:, None]

    def describe(self):
        return dict(tau_s=float(self.tau), friction=float(self.mu), solimp=list(CARPET_IMPEDANCE),
                    wavevectors_rad_per_m=self.waves.tolist(), phases_rad=self.phases.tolist())

"""Hoop and radial stress in the hollow rotating Rotor. SI units; Spin Rate in Hz.

Plane-stress rotating annulus (Timoshenko & Goodier, Theory of Elasticity, sec. 32), free at
both surfaces. The Sample's own centrifugal pressure on the bore is not included.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RotorStress:
    peak_hoop_pa: float     # the largest principal stress, at the bore
    peak_radial_pa: float   # at r = sqrt(a b) for a tube, at the centre for a solid disc


def rotating_tube_stress(outer_radius_m, inner_radius_m, spin_rate_hz, density_kg_m3, poissons_ratio):
    b, a, nu = outer_radius_m, max(inner_radius_m, 0.0), poissons_ratio
    load = density_kg_m3 * (2.0 * math.pi * spin_rate_hz) ** 2
    if a == 0.0:
        centre = (3.0 + nu) / 8.0 * load * b**2
        return RotorStress(peak_hoop_pa=centre, peak_radial_pa=centre)
    return RotorStress(
        peak_hoop_pa=load / 4.0 * ((3.0 + nu) * b**2 + (1.0 - nu) * a**2),
        peak_radial_pa=(3.0 + nu) / 8.0 * load * (b - a) ** 2,
    )

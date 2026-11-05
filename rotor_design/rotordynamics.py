"""Critical speeds of the Rotor on its Bearings. SI units; frequencies in Hz.

The Rotor is a zirconia tube closed by a solid Bottom Cap (z = 0 end) and a solid Turbine Cap.
Its Bearing locates it at two stations, one at each end, sharing the stiffness equally.
"""

import math
from dataclasses import dataclass

import numpy as np

# First free-free beam root, (beta L) for cosh(x) cos(x) = 1.
_FREE_FREE_BETA_L = 4.730040745


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    centre_of_mass_m: float           # from the Bottom Cap face
    polar_inertia_kg_m2: float
    transverse_inertia_kg_m2: float   # about the centre of mass


def rotor_mass_properties(outer_radius_m, inner_radius_m, length_m, bottom_cap_m, turbine_cap_m, density_kg_m3):
    """Tube plus two solid caps; the Sample is not included."""
    ro, ri = outer_radius_m, max(inner_radius_m, 0.0)
    # (mass, centre, polar, transverse about own centre) for each piece, bottom up.
    pieces = []
    for start, end, r_in in ((0.0, bottom_cap_m, 0.0), (bottom_cap_m, length_m - turbine_cap_m, ri),
                             (length_m - turbine_cap_m, length_m, 0.0)):
        h = end - start
        m = density_kg_m3 * math.pi * (ro**2 - r_in**2) * h
        pieces.append((m, (start + end) / 2.0, m * (ro**2 + r_in**2) / 2.0,
                       m * (3.0 * (ro**2 + r_in**2) + h**2) / 12.0))
    mass = sum(p[0] for p in pieces)
    centre = sum(p[0] * p[1] for p in pieces) / mass if mass > 0.0 else length_m / 2.0
    return MassProperties(
        mass_kg=mass,
        centre_of_mass_m=centre,
        polar_inertia_kg_m2=sum(p[2] for p in pieces),
        transverse_inertia_kg_m2=sum(p[3] + p[0] * (p[1] - centre) ** 2 for p in pieces),
    )


def rigid_body_critical_speeds_hz(mass_kg, transverse_inertia_kg_m2, polar_inertia_kg_m2, stiffness_n_m,
                                  lower_arm_m, upper_arm_m):
    """Synchronous forward-whirl critical speeds of a rigid Rotor on two equal stations, ascending.

    Stations sit lower_arm_m below and upper_arm_m above the centre of mass, each with half
    of stiffness_n_m. At synchronous whirl the gyroscopic term turns the tilt inertia into
    I_t - I_p, so a Rotor with I_p >= I_t has no conical critical speed.
    """
    k, a, b = stiffness_n_m / 2.0, lower_arm_m, upper_arm_m
    tilt_inertia = transverse_inertia_kg_m2 - polar_inertia_kg_m2
    # det(K - w^2 M) = 0 for coordinates (translation, tilt), as a polynomial in w^2.
    roots = np.roots([mass_kg * tilt_inertia, -k * (2.0 * tilt_inertia + (a**2 + b**2) * mass_kg),
                      k**2 * (a + b) ** 2])
    squares = sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-12 * abs(r) and r.real > 0.0)
    return tuple(math.sqrt(w2) / (2.0 * math.pi) for w2 in squares)


def first_bending_mode_hz(length_m, outer_radius_m, inner_radius_m, youngs_modulus_pa, density_kg_m3):
    """First free-free bending mode of a uniform tube; the Bearings are far softer than the tube."""
    ro, ri = outer_radius_m, max(inner_radius_m, 0.0)
    second_moment = math.pi * (ro**4 - ri**4) / 4.0
    area = math.pi * (ro**2 - ri**2)
    return (_FREE_FREE_BETA_L**2 / (2.0 * math.pi * length_m**2)
            * math.sqrt(youngs_modulus_pa * second_moment / (density_kg_m3 * area)))


def half_frequency_whirl_threshold_hz(rigid_body_critical_speeds_hz, onset_ratio=2.0):
    """Spin Rate at which a gas bearing's half-frequency whirl sets in: onset_ratio times the
    lowest rigid-body critical speed (the whirl runs near half the spin)."""
    return onset_ratio * min(rigid_body_critical_speeds_hz)

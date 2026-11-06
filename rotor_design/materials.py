"""Zirconia properties and the helium-4 vapour-pressure curve, with their sources. SI units.

Full zirconia citations and the data gaps are in docs/research/zirconia-properties.md.
"""

import math
from dataclasses import dataclass

import numpy as np

# Thermal conductivity, three segments:
#   T <= 1.7 K: k = 1e-3 T^2 W/(m K). Lawless & Gupta, Phys. Rev. B 28, 5507 (1983),
#     tetragonal Y-stabilised ZrO2 measured 1.7-45 K; abstract only, an order-of-magnitude
#     form "at the lowest temperatures".
#   1.7-25.17 K: log-log interpolation between the T^2 law at 1.7 K and the first measured
#     point below. NO MEASURED DATA in this range; the x0.3 to x3 Input Uncertainty covers it.
#   25.17-299.1 K: measured single-crystal Zr0.85Y0.15O1.925 (about 8 mol% Y2O3), data file
#     ysz15.fin posted by the Cahill group (likely Cahill, Watson & Pohl, Phys. Rev. B 46,
#     6131 (1992)); stands in for polycrystalline 3Y-TZP, for which no low-T data were found.
_T2_COEFFICIENT = 1e-3       # W/(m K^3)
_T2_LIMIT_K = 1.7
_MEASURED_T_K = np.array([25.17, 26.42, 29.29, 30.34, 35.55, 40.35, 50.54, 74.39, 99.91, 152.07,
                          202.98, 246.10, 299.10])
_MEASURED_K = np.array([1.344, 1.382, 1.344, 1.358, 1.315, 1.318, 1.327, 1.437, 1.567, 1.863,
                        2.044, 2.136, 2.225])


def _zirconia_conductivity(temperature_k):
    """W/(m K) at a temperature, or elementwise over an array of them."""
    t = np.asarray(temperature_k, dtype=float)
    t0, k0 = _T2_LIMIT_K, _T2_COEFFICIENT * _T2_LIMIT_K**2
    t1, k1 = _MEASURED_T_K[0], _MEASURED_K[0]
    bridge = np.exp(np.interp(np.log(np.maximum(t, t0)), [np.log(t0), np.log(t1)], [np.log(k0), np.log(k1)]))
    k = np.where(t <= _T2_LIMIT_K, _T2_COEFFICIENT * t**2,
                 np.where(t < t1, bridge, np.interp(t, _MEASURED_T_K, _MEASURED_K)))
    return float(k) if k.ndim == 0 else k


@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float
    flexural_strength_pa: float
    tensile_strength_pa: float
    youngs_modulus_pa: float
    poissons_ratio: float
    _conductivity: object

    def conductivity(self, temperature_k, multiplier=1.0):
        """Thermal conductivity (W/(m K)); multiplier carries the Input Uncertainty."""
        return multiplier * self._conductivity(temperature_k)


# Mechanical values: room temperature, 3Y-TZP manufacturer typical values; none measured below 77 K.
ZIRCONIA = Material(
    name="zirconia (3Y-TZP)",
    density_kg_m3=6020.0,               # CoorsTek "Zirconia Materials" brochure (2023), sintered YTZP
    flexural_strength_pa=1240e6,        # same, 4-point; datasheet range 951-1720 MPa
    tensile_strength_pa=550e6,          # Superior Technical Ceramics YTZP chart, lower end of 550-690 MPa
    youngs_modulus_pa=210e9,            # CoorsTek brochure; Superior Technical Ceramics (ASTM C848)
    poissons_ratio=0.30,                # Superior Technical Ceramics (ASTM C848); Morgan datasheet
    _conductivity=_zirconia_conductivity,
)


# Saturated vapour pressure of liquid helium-4, on T90. Donnelly & Barenghi, J. Phys. Chem. Ref.
# Data 27, 1217 (1998), section 17 (https://pages.uoregon.edu/rjd/vapor18.htm):
#   1.25-5.0 K: the ITS-90 helium-4 vapour-pressure equation (Preston-Thomas, Metrologia 27, 3
#     (1990)), T90 = A0 + sum_i Ai ((ln(p / Pa) - B) / C)^i, Table 17.1; inverted here for p.
#   0.65-1.25 K: the low-temperature relation p = (m kT / (2 pi hbar^2))^(3/2) kT exp(-L0 / (R T)),
#     L0 = 59.83 J/mol the latent heat at absolute zero.
# Both reproduce the recommended values of Table 17.3 to better than 0.1 %.
_ITS90_HELIUM4 = (
    # (upper limit K, (A0 .. A8), B, C, pressures (Pa) bracketing the range, where T90 rises with p)
    (2.1768, (1.392408, 0.527153, 0.166756, 0.050988, 0.026514, 0.001975, -0.017976, 0.005409, 0.013259),
     5.6, 2.9, (50.0, 1e4)),
    (5.0, (3.146631, 1.357655, 0.413923, 0.091159, 0.016349, 0.001826, -0.004325, -0.004973), 10.3, 1.9,
     (3e3, 3e5)),
)
HELIUM4_CURVE_K = (0.65, 5.0)
_ITS90_LOWER_K = 1.25
_HELIUM4_LATENT_HEAT_J_MOL = 59.83
_R_DONNELLY = 8.314510        # J/(mol K), as used with L0 by Donnelly & Barenghi
_K_BOLTZMANN = 1.380649e-23   # J/K
_HBAR = 1.054571817e-34       # J s
_HELIUM4_ATOM_KG = 4.002602e-3 / 6.02214076e23


def helium_vapour_pressure(temperature_k):
    """Saturated vapour pressure of liquid helium-4 (Pa), 0.65-5.0 K."""
    t = temperature_k
    lo, hi = HELIUM4_CURVE_K
    if not lo <= t <= hi:
        raise ValueError(f"{t} K is off the helium-4 vapour-pressure curve ({lo}-{hi} K)")
    if t < _ITS90_LOWER_K:
        kt = _K_BOLTZMANN * t
        return float((_HELIUM4_ATOM_KG * kt / (2.0 * math.pi * _HBAR**2)) ** 1.5 * kt
                     * math.exp(-_HELIUM4_LATENT_HEAT_J_MOL / (_R_DONNELLY * t)))
    coefficients, b, c, bracket = next((a, b, c, p) for limit, a, b, c, p in _ITS90_HELIUM4 if t <= limit)
    # Bisect for x = (ln p - B) / C within the bracket.
    x_lo, x_hi = ((math.log(p) - b) / c for p in bracket)
    for _ in range(60):
        x = (x_lo + x_hi) / 2.0
        if sum(a * x**i for i, a in enumerate(coefficients)) < t:
            x_lo = x
        else:
            x_hi = x
    return float(math.exp(b + c * (x_lo + x_hi) / 2.0))

"""Zirconia properties with their sources. SI units.

Full citations and the data gaps are in docs/research/zirconia-properties.md.
"""

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
    t = temperature_k
    if t <= _T2_LIMIT_K:
        return _T2_COEFFICIENT * t**2
    if t < _MEASURED_T_K[0]:
        t0, k0 = _T2_LIMIT_K, _T2_COEFFICIENT * _T2_LIMIT_K**2
        t1, k1 = _MEASURED_T_K[0], _MEASURED_K[0]
        return float(np.exp(np.interp(np.log(t), [np.log(t0), np.log(t1)], [np.log(k0), np.log(k1)])))
    return float(np.interp(t, _MEASURED_T_K, _MEASURED_K))


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

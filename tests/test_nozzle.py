"""Nozzle flow, checked against ideal-gas flow where real-gas effects vanish.

Helium at 300 K and 1 bar is ideal to about 0.05 %. Reference values from the
ideal-gas isentropic nozzle formulas, gamma = 5/3, Cd = 0.8, 0.5 mm throat.
"""

import pytest

from rotor_design.gas import AIR, NITROGEN
from rotor_design.nozzle import ThroatSearchRangeError, nozzle_flow

THROAT_M = 0.5e-3
CD = 0.8


def test_choked_flow_matches_ideal_gas_limit():
    # m = Cd A p0 sqrt(gamma / (R T0)) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1)))
    #   = 1.44497e-5 kg/s;  throat velocity sqrt(2 gamma R T0 / (gamma + 1)) = 882.6 m/s
    flow = nozzle_flow(1e5, 300.0, 1e4, THROAT_M, CD)
    assert flow.choked
    assert flow.mass_flow_kg_s == pytest.approx(1.44497e-5, rel=5e-3)
    assert flow.jet_velocity_m_s == pytest.approx(882.6, rel=5e-3)


def test_unchoked_flow_matches_ideal_gas_limit():
    # Back pressure 0.8 p0, above the critical ratio 0.487:
    #   m = Cd A p0 sqrt(2 gamma / ((gamma - 1) R T0) (r^(2/gamma) - r^((gamma+1)/gamma)))
    #     = 1.13725e-5 kg/s;  v = sqrt(2 gamma R T0 / (gamma - 1) (1 - r^((gamma-1)/gamma))) = 515.8 m/s
    flow = nozzle_flow(1e5, 300.0, 0.8e5, THROAT_M, CD)
    assert not flow.choked
    assert flow.mass_flow_kg_s == pytest.approx(1.13725e-5, rel=5e-3)
    assert flow.jet_velocity_m_s == pytest.approx(515.8, rel=5e-3)


@pytest.mark.parametrize("gas, mass_flow, velocity", [
    # Same formulas, gamma = 1.4, 300 K, 1 bar, Cd = 0.8, 0.5 mm throat:
    #   nitrogen (R = 296.80): m = 3.6045e-5 kg/s, v = 322.31 m/s
    #   air (R = 287.05):      m = 3.6652e-5 kg/s, v = 316.96 m/s
    (NITROGEN, 3.6045e-5, 322.31),
    (AIR, 3.6652e-5, 316.96),
])
def test_diatomic_choked_flow_matches_ideal_gas_limit(gas, mass_flow, velocity):
    flow = nozzle_flow(1e5, 300.0, 1e4, THROAT_M, CD, gas)
    assert flow.choked
    assert flow.mass_flow_kg_s == pytest.approx(mass_flow, rel=5e-3)
    assert flow.jet_velocity_m_s == pytest.approx(velocity, rel=5e-3)
    # Critical pressure ratio (2 / (gamma + 1))^(gamma / (gamma - 1)) = 0.5283
    assert flow.throat_pressure_pa / 1e5 == pytest.approx(0.5283, rel=5e-3)


def test_diatomic_nozzle_chokes_only_below_its_critical_ratio():
    assert not nozzle_flow(1e5, 300.0, 0.55e5, THROAT_M, CD, NITROGEN).choked
    assert nozzle_flow(1e5, 300.0, 0.50e5, THROAT_M, CD, NITROGEN).choked


def test_throat_below_search_range_raises_its_own_error():
    # Helium at 6 K and 10 bar is liquid-like: its isentrope peaks below the throat search range.
    with pytest.raises(ThroatSearchRangeError, match="below the throat search range"):
        nozzle_flow(10e5, 6.0, 26.66, THROAT_M, CD)
    assert issubclass(ThroatSearchRangeError, ValueError)

"""Nozzle flow, checked against ideal-gas flow where real-gas effects vanish.

Helium at 300 K and 1 bar is ideal to about 0.05 %. Reference values from the
ideal-gas isentropic nozzle formulas, gamma = 5/3, Cd = 0.8, 0.5 mm throat.
"""

import pytest

from rotor_design.nozzle import nozzle_flow

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

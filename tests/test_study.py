"""Run the study: assumptions in, results CSV and plots out."""

import csv
from pathlib import Path

import pytest

from rotor_design.study import run_study

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"


def test_study_on_tiny_grid_writes_csv_and_plot(tmp_path):
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[10.0, 20.0])

    with open(tmp_path / "probe_bore_fit.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [float(r["rotor_length_mm"]) for r in rows] == [10.0, 20.0]
    for column in ("max_rotor_diameter_mm", "projected_width_mm", "fits_bore", "feasible", "binding_constraint"):
        assert column in rows[0]
    # 20 mm hand value from test_geometry; a shorter Rotor allows a wider one.
    assert float(rows[1]["max_rotor_diameter_mm"]) == pytest.approx(7.874, abs=1e-3)
    assert float(rows[0]["max_rotor_diameter_mm"]) > float(rows[1]["max_rotor_diameter_mm"])

    assert (tmp_path / "max_rotor_diameter_vs_rotor_length.png").stat().st_size > 0


def test_study_writes_required_flow_versus_nozzle_geometry(tmp_path):
    sweeps = {"nozzle_count": [2, 4], "nozzle_throat_mm": [0.2, 0.3], "nozzle_angle_deg": [10.0, 20.0]}
    run_study(ASSUMPTIONS_FILE, tmp_path, rotor_lengths_mm=[20.0], nozzle_sweeps=sweeps)

    with open(tmp_path / "nozzle_geometry_sweep.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {r["swept_variable"] for r in rows} == set(sweeps)
    for r in rows:
        assert float(r["required_mass_flow_kg_s"]) > 0.0
        assert float(r["spin_rate_hz"]) == 1000.0

    assert (tmp_path / "required_flow_vs_nozzle_geometry.png").stat().st_size > 0

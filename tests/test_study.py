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

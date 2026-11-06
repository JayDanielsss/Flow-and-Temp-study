"""Design Sheet for one chosen Design: selection, rows, feasible ranges, header and plot."""

import csv
import math
from dataclasses import asdict, fields
from pathlib import Path

import pytest

import run_design_sheet
from rotor_design.assumptions import load_assumptions
from rotor_design.design_sheet import (ENGINEERS_CHOICE, MODEL_DERIVED, PLACEHOLDER, RANGE_REL_TOLERANCE,
                                       VALIDATION_STATUS, build_rows, feasible_probability, is_feasible,
                                       make_sheet, read_design_sheet)
from rotor_design.evaluate import Design, evaluate

ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "assumptions.toml"
SOURCES = {MODEL_DERIVED, PLACEHOLDER, ENGINEERS_CHOICE}
# One baseline Design keeps these tests quick; its Sample length has a finite feasible range.
RANGED = "sample_length_mm"


@pytest.fixture(scope="module")
def assumptions():
    return load_assumptions(ASSUMPTIONS_FILE)


@pytest.fixture(scope="module")
def baseline(assumptions):
    return evaluate(Design(rotor_length_mm=20.0), assumptions)


def _write_ranked_csv(path, results):
    """A one-row ranked CSV in the study's format: rank, configuration, then the Results."""
    row = {"rank": 1, "configuration": "test",
           **{k: v for k, v in asdict(results).items() if not isinstance(v, (tuple, dict))}}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


@pytest.fixture(scope="module")
def sheets(tmp_path_factory, baseline):
    """One Design Sheet chosen by rank (with an override) and one from a Design file."""
    tmp = tmp_path_factory.mktemp("sheets")
    ranked = tmp / "ranked_designs.csv"
    _write_ranked_csv(ranked, baseline)
    design_file = tmp / "design.toml"
    design_file.write_text("rotor_length_mm = 20.0\nnozzle_throat_mm = 0.5\n")
    out = {}
    for name, choice in (("rank", ["--ranked-csv", str(ranked), "--rank", "1", "--set", "nozzle_count=8",
                                   "--ranges", RANGED]),
                         ("file", ["--design", str(design_file), "--ranges", "none"])):
        csv_path, plot_path = tmp / f"sheet_{name}.csv", tmp / "Plots" / f"sheet_{name}.png"
        run_design_sheet.main([*choice, "--assumptions", str(ASSUMPTIONS_FILE), "--out", str(csv_path),
                               "--plot", str(plot_path), "--uncertainty-csv", str(tmp / "missing.csv"),
                               "--workers", "1"])
        out[name] = (csv_path, plot_path)
    return out


def _row(rows, part, quantity):
    matches = [r for r in rows if r["part"] == part and r["quantity"] == quantity]
    assert len(matches) == 1, (part, quantity)
    return matches[0]


def test_rank_and_design_file_both_write_a_design_sheet(sheets):
    for csv_path, _ in sheets.values():
        header, rows = read_design_sheet(csv_path)
        assert rows and list(rows[0]) == ["part", "quantity", "operating", "as_machined", "feasible_min",
                                          "feasible_max", "unit", "source", "note"]
    _, from_file = read_design_sheet(sheets["file"][0])
    throat = _row(from_file, "Nozzle", "throat diameter")
    assert float(throat["operating"]) == 0.5 and throat["source"] == ENGINEERS_CHOICE


def test_command_line_override_changes_the_evaluated_design(sheets, baseline):
    _, rows = read_design_sheet(sheets["rank"][0])
    count = _row(rows, "Nozzle", "count")
    assert int(count["operating"]) == 8 != baseline.nozzle_count
    assert count["source"] == ENGINEERS_CHOICE and "override" in count["note"]
    supply = float(_row(rows, "Drive Gas", "required Supply Pressure")["operating"])
    assert not math.isclose(supply, baseline.required_supply_pressure_bar, rel_tol=1e-3)
    # Fields the study set come from the rank; fields it left unset are placeholders.
    assert _row(rows, "Rotor", "Rotor Length")["source"] == MODEL_DERIVED
    assert _row(rows, "Sample", "thermal conductivity")["source"] == PLACEHOLDER


def test_every_design_input_and_geometric_result_has_a_row(assumptions):
    sheet = make_sheet(Design(rotor_length_mm=20.0), assumptions)
    rows = build_rows(sheet, range_keys=())
    keys = {r.key for r in rows}
    not_applicable = {"cold_support_gap_mm", "hub_exit_diameter_mm"}  # vapour only, Fluted Cap
    assert {f.name for f in fields(Design)} - not_applicable <= keys
    quantities = {(r.part, r.quantity) for r in rows}
    for geometric in [("Rotor", "largest outer diameter the Probe Bore allows"), ("Rotor", "bore diameter"),
                      ("Stator", "length"), ("Stator", "outer diameter"),
                      ("Stator", "projected width at the Tilt Angle"), ("Probe Bore", "usable bore"),
                      ("Turbine", "Turbine Cap face area left to the blades or flutes"), ("Sample", "volume")]:
        assert geometric in quantities
    for r in rows:
        assert r.part and r.source in SOURCES and isinstance(r.unit, str)


def test_feasible_range_is_feasible_inside_and_infeasible_just_outside(sheets, assumptions, baseline):
    _, rows = read_design_sheet(sheets["rank"][0])
    row = _row(rows, "Sample", "length")
    lo, hi = float(row["feasible_min"]), float(row["feasible_max"])
    nominal = float(row["operating"])
    assert lo <= nominal <= hi and lo < hi
    # Others held at nominal: the rank's Design with its override, diameter pinned.
    design = Design(rotor_length_mm=20.0, nozzle_count=8, rotor_diameter_mm=baseline.rotor_diameter_mm)
    step = 2.0 * RANGE_REL_TOLERANCE * nominal
    for value in (lo, hi):
        assert is_feasible(Design(**{**asdict(design), RANGED: value}), assumptions)
    edges = [(lo, lo - step), (hi, hi + step)]
    finite = [outside for edge, outside in edges if "at the edge searched" not in row["note"] or edge != hi]
    assert finite
    for outside in finite:
        assert not is_feasible(Design(**{**asdict(design), RANGED: outside}), assumptions)
    # Rows not asked for keep an empty range.
    assert _row(rows, "Nozzle", "throat diameter")["feasible_min"] == ""


def test_header_carries_validation_status_and_ranked_placeholders(sheets):
    header, _ = read_design_sheet(sheets["rank"][0])
    text = "\n".join(header)
    assert VALIDATION_STATUS in text
    ranked = [line for line in header if line.strip()[:1].isdigit()]
    assert any("zirconia conductivity" in line and "bridge" in line for line in ranked)
    assert any("Sample conductivity" in line for line in ranked)
    shifts = [float(line.rsplit("shift ", 1)[1].split(" mK")[0]) for line in ranked if " mK" in line]
    assert shifts == sorted(shifts, reverse=True) and len(shifts) >= 2


def test_results_block_reports_heat_loads_and_binding_constraint(sheets, baseline):
    _, rows = read_design_sheet(sheets["file"][0])
    results = {r["quantity"]: r for r in rows if r["part"] == "Results"}
    for quantity in ("Sample Heat Load", "Sample hottest point", "Bath Heat Load", "P(feasible)",
                     "Binding Constraint"):
        assert quantity in results
    assert any(q.startswith("Sample Heat Load: ") for q in results)
    assert results["P(feasible)"]["operating"] == "" and "not available" in results["P(feasible)"]["note"]


def test_engineers_choice_rows_cover_turbine_flutes_and_closures(sheets):
    _, rows = read_design_sheet(sheets["file"][0])
    chosen = [r for r in rows if r["source"] == ENGINEERS_CHOICE and r["operating"] == ""]
    assert any(r["part"] == "Turbine" and "flute" in r["quantity"] for r in chosen)
    assert any("joint" in r["quantity"] or "closure" in r["quantity"] for r in chosen)
    assert all(r["note"] for r in chosen)


def test_temperature_plot_is_written(sheets):
    for _, plot_path in sheets.values():
        assert plot_path.exists() and plot_path.stat().st_size > 0 and plot_path.parent.name == "Plots"


def test_feasible_probability_read_from_the_uncertainty_csv(tmp_path, baseline):
    row = {"rank": 1, "feasible_probability": 0.9,
           **{k: v for k, v in asdict(baseline).items() if not isinstance(v, (tuple, dict))}}
    path = tmp_path / "uncertainty_top_designs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    assert feasible_probability(baseline, path) == 0.9
    other = evaluate(Design(rotor_length_mm=20.0, nozzle_count=8), load_assumptions(ASSUMPTIONS_FILE))
    assert feasible_probability(other, path) is None


def test_sample_position_placement_window_and_standoffs_are_on_the_sheet(sheets, baseline):
    _, rows = read_design_sheet(sheets["file"][0])
    position = _row(rows, "Sample", "Sample Position")
    assert float(position["operating"]) == pytest.approx(baseline.sample_position_mm, rel=1e-5)
    assert float(position["feasible_min"]) == pytest.approx(baseline.placement_window_low_mm, abs=1e-6)
    assert float(position["feasible_max"]) == pytest.approx(baseline.placement_window_high_mm, rel=1e-5)
    assert position["source"] == MODEL_DERIVED
    assert float(_row(rows, "Sample", "Bottom Standoff")["operating"]) == pytest.approx(
        baseline.bottom_standoff_mm, abs=1e-6)
    assert float(_row(rows, "Sample", "Turbine Cap Standoff")["operating"]) == pytest.approx(
        baseline.turbine_cap_standoff_mm, rel=1e-5)
    window = _row(rows, "Results", "Placement Window")
    assert float(window["operating"]) == pytest.approx(
        baseline.placement_window_high_mm - baseline.placement_window_low_mm, rel=1e-5)
    assert "Sample's boundary" in _row(rows, "Results", "Sample Heat Load")["note"]

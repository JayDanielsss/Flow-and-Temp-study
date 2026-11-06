"""Run the whole study from the assumptions file; outputs go to Plots/."""

import os
import sys
from pathlib import Path

from rotor_design.study import run_study

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    assumptions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "assumptions.toml"
    results = run_study(assumptions_path, HERE / "Plots", workers=os.cpu_count())
    fitting = sum(r.feasible for r in results)
    print(f"Evaluated {len(results)} Designs, {fitting} feasible. Outputs in {HERE / 'Plots'}")

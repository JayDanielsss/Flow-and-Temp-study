"""Worker processes keep numpy's math library single-threaded unless the user says otherwise."""

import os
import subprocess
import sys


def _thread_setting(env):
    code = "import rotor_design, os; print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'])"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          check=True).stdout.split()


def test_importing_the_package_defaults_the_math_library_to_one_thread():
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    env["PYTHONPATH"] = os.getcwd()
    assert _thread_setting(env) == ["1", "1"]


def test_a_thread_count_the_user_set_is_kept():
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", PYTHONPATH=os.getcwd())
    assert _thread_setting(env) == ["4", "2"]

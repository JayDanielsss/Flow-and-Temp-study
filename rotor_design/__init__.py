import os

# The study and Monte Carlo run one worker process per core; numpy's math library starting its
# own threads in each of them swamps the cores. Single-threaded unless the user set otherwise
# (only takes effect if numpy is not yet imported).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

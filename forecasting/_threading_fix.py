import os
import sys

# Force the OS to only use ONE CPU core for this process
if sys.platform == "linux":
    os.sched_setaffinity(0, {0})

# Choke the thread pools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
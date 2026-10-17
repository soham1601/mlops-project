"""Heart Disease MLOps pipeline."""

import os

# Our data sets are tiny (hundreds of rows). Letting NumPy, scikit-learn and XGBoost start one
# worker thread per CPU thread only makes them wait on each other, and on a many-core machine
# it can freeze the whole PC. These limits are read when the libraries are first imported, so
# they must be set here, before anything imports NumPy. `setdefault` never overrides a value
# you have set yourself, e.g. `set OMP_NUM_THREADS=8` to allow more threads for big data.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

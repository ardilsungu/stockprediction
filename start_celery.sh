#!/bin/bash
cd "$(dirname "$0")/backend"
source ../venv/bin/activate
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
BLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
celery -A config worker --loglevel=info

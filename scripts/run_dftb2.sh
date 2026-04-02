#!/bin/bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH
export OMP_NUM_THREADS=2
cd /project/rrg-aspuru/memoozd/diffusion-ts
python scripts/test_parallel_quick.py --n-seeds 10 --n-workers 10 --noise 0.2 --functional DFTB2

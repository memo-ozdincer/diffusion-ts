#!/bin/bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH
export OMP_NUM_THREADS=2
cd /project/rrg-aspuru/memoozd/diffusion-ts
python scripts/test_parallel_quick.py --n-seeds 100 --n-workers 50 --noise 0.2

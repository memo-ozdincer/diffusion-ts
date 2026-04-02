#!/bin/bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH
export OMP_NUM_THREADS=2
cd /project/rrg-aspuru/memoozd/diffusion-ts
python scripts/test_isopropanol_single.py

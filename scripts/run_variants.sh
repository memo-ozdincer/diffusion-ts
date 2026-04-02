#!/bin/bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH
export OMP_NUM_THREADS=2
cd /project/rrg-aspuru/memoozd/diffusion-ts

echo "=== DFTB3, 10 seeds, noise=0.2A ==="
python scripts/test_parallel_quick.py --n-seeds 10 --n-workers 10 --noise 0.2 --functional DFTB3 2>&1 | grep -E "(Summary|CRASH|skip)" || true

echo ""
echo "=== DFTB2, 10 seeds, noise=0.05A (gentle) ==="
python scripts/test_parallel_quick.py --n-seeds 10 --n-workers 10 --noise 0.05 --functional DFTB2 2>&1 | grep -E "(Summary|CRASH|skip)" || true

echo ""
echo "=== DFTB0, 100 seeds, noise=0.5A (aggressive) ==="
python scripts/test_parallel_quick.py --n-seeds 100 --n-workers 50 --noise 0.5 2>&1 | tail -5

#!/bin/bash
#SBATCH --job-name=diffusion-ts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/diffusion-ts/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/diffusion-ts/logs/%x_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Sample SLURM template for diffusion-ts
# =============================================================================
#
# BEFORE FIRST RUN — set up the environment:
#
#   module load StdEnv/2023 python/3.11.5
#   python -m venv /project/rrg-aspuru/memoozd/diffusion-ts/.venv
#   source /project/rrg-aspuru/memoozd/diffusion-ts/.venv/bin/activate
#   pip install uv && uv pip install -r requirements.txt
#
# EXTERNAL DATA / DEPENDENCIES:
#
#   Transition1x HDF5:  /project/rrg-aspuru/memoozd/data/transition1x.h5
#                        (~3 GB, contains TS geometries for 10k reactions)
#                        Source: https://github.com/Shenvi-Group/transition1x
#
#   SCINE (Sparrow):    Loaded via scine-sparrow pip package.
#                        Provides DFTB0/DFTB3 energy+force evaluator.
#                        No additional module loads needed beyond pip install.
#
#   HIP-NN potential:   Checkpoint at /project/rrg-aspuru/memoozd/models/
#                        (only needed if using --calculator hip, not for SCINE runs)
#
# =============================================================================

mkdir -p /scratch/memoozd/diffusion-ts/logs

# ---- Paths (edit these) ----
PROJECT_DIR="/project/rrg-aspuru/memoozd/diffusion-ts"
SCRATCH_DIR="/scratch/memoozd/diffusion-ts"
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"

cd "$SCRATCH_DIR"

# ---- Modules & venv ----
module purge
module load StdEnv/2023
module load python/3.11.5
source "$PROJECT_DIR/.venv/bin/activate"

# ---- Temp dirs ----
TMP_BASE="${SLURM_TMPDIR:-${SCRATCH_DIR}/tmp}"
export TMPDIR="${TMP_BASE}/${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# ---- Threading ----
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-128}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# ---- PYTHONPATH so `from src.…` imports work ----
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# ---- Worker config ----
THREADS_PER_WORKER=4
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# ---- Output ----
OUT_DIR="${SCRATCH_DIR}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "diffusion-ts  |  $(date)"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  CPUs: $TOTAL_CPUS  Workers: $N_WORKERS"
echo "=============================================="

# =============================================================================
# Example: NR minimization at noise=1.0A, 20k steps, RFO+PLS
# =============================================================================
python "$PROJECT_DIR/src/benchmarks/runners/run_minimization_parallel.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR/nr_n1.0_20k" \
    --scine-functional "DFTB0" \
    --n-steps 20000 \
    --max-samples 300 \
    --start-from "midpoint_rt_noise1.0A" \
    --noise-seed "$SLURM_JOB_ID" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --method newton_raphson \
    --max-atom-disp 1.3 \
    --force-converged 1e-4 \
    --min-interatomic-dist 0.5 \
    --project-gradient-and-v \
    --log-spectrum-k 10 \
    --optimizer-mode rfo \
    --polynomial-linesearch \
    --trust-radius-floor 0.01 \
    --relaxed-eval-threshold 0.01 \
    --accept-relaxed

# =============================================================================
# Post-run analysis (optional)
# =============================================================================
ANALYSIS_DIR="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"

python "$PROJECT_DIR/src/benchmarks/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_DIR" \
    --top-k 15 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

echo "Done. Results: $OUT_DIR"

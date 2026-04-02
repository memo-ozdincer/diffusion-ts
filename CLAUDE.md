# diffusion-ts

Transition state search + diffusion-based TS generation via adjoint sampling. Extracted from [ts-tools](../ts-tools). Finds index-1 saddle points on potential energy surfaces, then trains diffusion models to generate them directly.

## Project structure

```
src/
  core_algos/          # Backend-agnostic algorithms
    saddle_optimizer.py  # NR→GAD→P-RFO three-phase TS optimizer (main entry point)
    gad.py               # GAD fundamentals: mode tracking, Euler step, RK45
    types.py             # PredictFn protocol
  dependencies/        # Calculator adapters, Eckart projection, logging, utils
    alignment.py         # Kabsch + Hungarian geometry alignment and clustering
    calculators.py       # LJ / HIP / SCINE predict_fn factories
    differentiable_projection.py  # Eckart projection, reduced-basis Hessian
    common_utils.py      # Transition1x dataset loader, CLI helpers
  benchmarks/          # Baseline optimizers and analysis (from ts-tools)
    baselines/           # NR minimization, PIC-ARC
    runners/             # Parallel experiment runners
    scripts/             # Analysis scripts, SLURM templates
  noisy/               # Production multi-mode GAD with v2 kicking
  parallel/            # Multi-worker execution (LJ, SCINE, Dask)
  runners/             # CLI entrypoints
scripts/
  test_parallel_quick.py    # Parallel NR→GAD→P-RFO TS search (main data collection)
  cluster_ts_geometries.py  # Cluster TS geometries via aligned RMSD
  explore_clustering.py     # Compare clustering algorithms (hierarchical, DBSCAN, etc.)
  *.slurm, *.sh             # SLURM templates and run scripts
```

## Key concepts

- **predict_fn interface**: `predict_fn(coords, atomic_nums, do_hessian, require_grad) -> dict` with keys `energy`, `forces`, `hessian`. All algorithms use this; backends live in `dependencies/calculators.py`.
- **Three backends**: LJ (analytical, for testing), HIP (GPU ML potential), SCINE Sparrow (CPU semi-empirical DFTB0).
- **Hessian projection**: Raw Hessians have 5-6 rigid-body null modes. Always use Eckart projection via `reduced_basis_hessian_torch()` before eigenvalue analysis. This returns a full-rank (3N-k, 3N-k) vibrational Hessian — no threshold filtering needed.
- **TS convergence criterion**: **n_neg == 1** (exactly one negative vibrational eigenvalue after Eckart projection) AND **force_norm < 0.01 eV/A**. This is the only valid criterion. No eigenvalue product gates, no threshold relaxation on the TS check.

## Three-phase saddle optimizer

The main optimizer is `src/core_algos/saddle_optimizer.py`:

```python
from src.core_algos.saddle_optimizer import find_transition_state

result = find_transition_state(predict_fn, coords, atomic_nums, atomsymbols)
# result["converged"] == True when n_neg == 1 (GAD found TS)
# result["refined"] == True when P-RFO pushed force < 1e-4
```

**Phase 1 — NR minimization** (find local minimum, n_neg == 0):
- RFO augmented Hessian (no hyperparams, guaranteed downhill)
- Polynomial line search (cubic interpolation refinement)
- Trust region with floor at 0.01 A
- Relaxed convergence: accept min_eval >= -0.01 as "at minimum"

**Phase 2 — GAD saddle search** (climb to index-1 saddle, n_neg == 1):
- Eigenvalue-clamped adaptive dt: `dt ~ 1/clamp(|λ₀|, 0.01, 100)`
- Mode tracking across eigenvector degeneracies
- Eckart-projected GAD dynamics (no TR drift)

**Phase 3 — P-RFO TS refinement** (push force → 0 at saddle):
- Partitioned Rational Function Optimization
- TS mode (lowest eigenvector): maximize via uphill secular equation shift
- All other modes: standard RFO minimization
- Drives force from ~0.01 to ~1e-5 eV/A in 10-20 steps
- Trust region with adaptive radius

All three phases are **state-based** (no path history), suitable for integration with diffusion models.

## Geometry alignment and clustering

`src/dependencies/alignment.py` — comparing molecular geometries requires handling two symmetries that make naive RMSD meaningless:

### Kabsch algorithm (rigid-body alignment)

Two identical molecules at different positions/orientations have large raw RMSD. Kabsch finds the optimal rotation + translation via SVD:

1. Center both geometries at origin: `A_c = A - centroid(A)`, `B_c = B - centroid(B)`
2. Cross-covariance: `H = A_c^T @ B_c`
3. SVD: `H = U Σ V^T`
4. Optimal rotation: `R = V @ diag(1, 1, sign(det(V U^T))) @ U^T`
   - The sign correction ensures a proper rotation (det = +1), not a reflection
5. Translation: `t = centroid(B) - R @ centroid(A)`
6. RMSD: `sqrt(mean(||R @ A_i + t - B_i||^2))`

### Hungarian algorithm (atom permutation symmetry)

Equivalent atoms (e.g., the 6 methyl H in isopropanol) can be assigned to any position. Naive RMSD penalizes "wrong" assignments. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the optimal atom-to-atom mapping that minimizes total squared distance in O(n^3).

For each equivalence class of k atoms:
1. Build k×k cost matrix: `C[i,j] = ||atom_i(geom1) - atom_j(geom2)||^2`
2. Solve assignment problem: find permutation π minimizing `Σ C[i, π(i)]`
3. Apply permutation to geom1 before Kabsch alignment

### Combined pipeline for isopropanol

Isopropanol (C₃H₈O, 12 atoms) has:
- **Fixed atoms**: central C (idx 0), O (idx 3), H-on-C (idx 4), H-on-O (idx 5)
- **Methyl carbons**: C1 (idx 1) ↔ C2 (idx 2) — can swap, but must co-swap their H groups
- **Methyl H pool**: 6 H atoms (idx 6-11) — all equivalent after carbon assignment

The alignment procedure:
1. Try both methyl carbon assignments: (C1→C1, C2→C2) and (C1→C2, C2→C1)
2. For each, co-swap the attached H groups (6,7,8 ↔ 9,10,11)
3. Apply Hungarian matching on the 6 methyl H atoms within each assignment
4. Kabsch-align the permuted geometry
5. Return the minimum RMSD across all assignments

### Two-level hierarchical clustering

Single RMSD thresholds either over-split or mega-cluster. Use two levels:

**Level 1 (L1=0.15 Å)**: Tight cores — near-duplicate TS from the same basin.
**Level 2 (L2=0.4 Å)**: Group cores into broader structural families using mean inter-core RMSD.

Result (87 TS): 35 families (8 multi-member covering 60 TS, 27 singletons).

### Reaction type classification

Each TS classified by which bonds stretched >30% from equilibrium:
- Angular/torsional (43%): No bonds broken, rotational barriers
- C-C breaking (21%): Methyl detachment
- C-H migration (15%): Central H moving, very flat saddles (eig0 ≈ -0.5)
- Multi-bond (11%): 2+ bonds simultaneously stretched (mostly singletons)
- C-O breaking (3%): Hydroxyl leaving

Scripts: `scripts/cluster_structural_v3.py` (two-level), `scripts/plot_reaction_types.py`

```bash
srun --overlap --jobid=<JOBID> python scripts/cluster_structural_v3.py
# Reads: /scratch/memoozd/diffusion-ts/adaptive_results.json
# Writes: /scratch/memoozd/diffusion-ts/structural_clustering_v3/families_v3.json
```

## Data collection pipeline

`scripts/test_parallel_quick.py` — parallel NR→GAD TS search:

1. Generate noisy isopropanol geometries (Gaussian displacement, configurable σ)
2. Filter: reject if min interatomic distance < 0.4 Å
3. NR minimization (1000 steps), collecting n_neg=1 and n_neg=2 snapshots
4. Multi-candidate GAD from handoff points: last_nneg1, first_nneg1, mid_nneg1, last_nneg2
5. TS convergence: n_neg==1 AND force_norm < 0.01
6. Save results with ts_coords to JSON

```bash
srun --overlap --jobid=<JOBID> python scripts/test_parallel_quick.py \
    --n-seeds 100 --noise 0.3 --n-workers 50
```

### Convergence rates (isopropanol, DFTB0)

| Noise (Å) | Rate (v7) | Rate (v11) | No-PLS | Notes |
|-----------|-----------|------------|--------|-------|
| 0.2 | 86% | — | **98%** | 4 main TS families |
| 0.3 | 74% | **99%** | **100%** | rms_force + NR→candidates→GAD |
| 0.5 | 35% | — | **91%** | Many rare/unique TS |

PLS (polynomial line search) makes negligible difference — the handoff mechanism is what matters.

## Running experiments

```bash
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
export PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH
export OMP_NUM_THREADS=2

# Using the saddle optimizer directly
python -c "from src.core_algos.saddle_optimizer import find_transition_state; ..."

# Using the parallel benchmark runner
python src/benchmarks/runners/run_minimization_parallel.py --h5-path transition1x.h5 [flags...]
```

SLURM templates in `scripts/` and `src/benchmarks/scripts/slurm_templates/`.

## Code conventions

- Python 3.11+, shares venv with ts-tools at `/project/rrg-aspuru/memoozd/ts-tools/.venv/`
- Dataset: Transition1x (H5), loaded via `src/dependencies/common_utils.py`
- Results go to `/scratch/memoozd/ts-tools-scratch/runs/`
- Keep `src/core_algos/` backend-agnostic (no HIP/SCINE imports)
- `src/dependencies/` is the glue layer — backend quirks belong here

## Don't

- Don't use anything other than n_neg == 1 + force < 0.01 as TS convergence
- Don't skip Eckart projection when computing vibrational eigenvalues
- Don't add threshold filtering beyond Eckart projection
- Don't use `require_grad=True` with SCINE (raises NotImplementedError)
- Don't use path-based optimizers for diffusion model integration — keep everything state-based

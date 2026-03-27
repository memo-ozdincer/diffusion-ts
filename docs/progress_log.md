# Progress Log

## 2026-03-27

### Session: Data collection + clustering pipeline

**Goal**: Build the full pipeline for Step 1 (TS data collection) and Step 2 (geometry clustering) of the adjoint sampling TS diffusion project.

#### What was done

1. **Parallel TS search** (`scripts/test_parallel_quick.py`)
   - Parallel NR→GAD saddle search using ProcessPoolExecutor (50 workers)
   - SCINE/DFTB0 backend with analytical Hessians
   - Multi-candidate handoff: last_nneg1, first_nneg1, mid_nneg1, last_nneg2
   - Results saved to `/scratch/memoozd/diffusion-ts/parallel_results.json`

2. **Convergence improvement** (background agent, iterative)
   - Baseline at 0.3A noise: 50/100 (50%)
   - Improvements applied:
     - Lower min_dist filter: 0.5→0.4A (skip rate 28→15)
     - More NR steps: 500→1000 (more handoff candidates)
     - More GAD steps: 3000→5000 (more convergence time)
     - Mid-nneg1 handoff candidate (catches cases others miss)
     - Index-2 recovery: perturb along v₂ when stuck at n_neg=2
   - Improvements rejected:
     - dt_scale_factor: mixed results, ±4 seeds cancel out. Removed.
   - **Final: 74/100 at 0.3A** (from 50/100 baseline)
   - **0.5A noise: 35/100** (from 28%)
   - Bug fix: `out_new` undefined when all trust-region retries hit distance check

3. **Kabsch + Hungarian clustering** (`src/dependencies/alignment.py`, `scripts/cluster_ts_geometries.py`)
   - Kabsch: SVD-based rigid alignment (translation + proper rotation)
   - Hungarian: optimal atom permutation via `linear_sum_assignment`
   - Methyl group enumeration: try both C₁↔C₂ assignments, co-swap H groups
   - Hierarchical clustering: average linkage, 0.2A threshold
   - **67 TS clustered into 27 structural clusters**
   - **4 main families** (49/67 = 73% of all TS):
     - Family A (21): E=-310.36, eig0=-0.57, most common
     - Family B (16): E=-310.03, eig0=-1.87
     - Family C (17): E=-308.37, eig0=-2.32, largest
     - Family D (10): E=-308.22, eig0=-1.22
   - 18 singletons including rare TS at E=-311.94 (deepest)
   - Cross-family RMSD ≈ 0.5A confirms structurally distinct
   - Natural RMSD gap at 0.14→0.26A (0.2A threshold is optimal)

4. **Documentation**
   - `CLAUDE.md`: updated with alignment, clustering, and data collection sections
   - `docs/step1_data_collection.md`: full methodology and results
   - `docs/step2_clustering.md`: Kabsch/Hungarian algorithm docs and clustering results
   - Memory files: convergence experiments, clustering results, feedback

5. **DFTB variant testing** (prior work, documented)
   - DFTB0: works (analytical Hessians), used for all production runs
   - DFTB1: does not exist
   - DFTB2: crashes during GAD (semi-numerical Hessian failure)
   - DFTB3: crashes during GAD (same issue)

#### What didn't work

- **dt_scale_factor**: Slowing GAD step size helped some seeds but hurt others equally. Net zero effect. Removed.
- **DFTB2/3**: Semi-numerical Hessians crash during GAD dynamics. Not fixable without upstream SCINE changes.
- **Normal mode displacement**: Test hung, inconclusive. Uniform Gaussian noise works fine.
- **Complete linkage**: Over-splits clusters. Average linkage is better for this data.

#### What's next

- **Step 2 completion**: Re-cluster with convergence agent's final results (may have more TS)
- **Step 3**: Adjoint sampling — train diffusion model using TS geometries as target distribution
- **Step 4**: Evaluation — generate TS candidates from diffusion model, verify with saddle optimizer

#### Key numbers

| Metric | Value |
|--------|-------|
| TS convergence at 0.2A | 86% (79/92) |
| TS convergence at 0.3A | 74% (74/100) |
| TS convergence at 0.5A | 35% (35/100) |
| Structural clusters (0.2A) | 27 |
| Main TS families | 4 |
| Singleton TS | 18 |
| SLURM job | 1239045 on tri0034 |

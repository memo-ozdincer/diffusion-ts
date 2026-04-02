# Step 1: TS Data Collection — Results

## Overview

Collected transition state geometries for isopropanol (C3H8O, 12 atoms) using the NR→GAD→P-RFO three-phase saddle optimizer with SCINE Sparrow DFTB0.

## Method

1. Start from isopropanol equilibrium geometry
2. Add isotropic Gaussian noise (σ = 0.2–0.5 Å) to all atomic positions
3. Filter: reject geometries with any interatomic distance < 0.4 Å
4. **Phase 1 — NR minimization** (up to 1000 steps):
   - RFO augmented Hessian with polynomial line search
   - Trust region with floor at 0.01 Å
   - Collect snapshots where n_neg ≥ 1 for GAD handoff
5. **Phase 2 — GAD saddle search** (up to 5000 steps):
   - Multi-candidate handoff: try last n_neg=1, first n_neg=1, middle n_neg=1, last n_neg=2
   - Eigenvalue-clamped adaptive dt: `dt = base_dt / clamp(|λ₀|, 0.01, 100)`
   - Mode tracking via eigenvector overlap
   - Index-2 recovery: perturb along v₂ when stuck at n_neg=2 with low force
6. **Convergence**: n_neg == 1 (after Eckart projection) AND force_norm < 0.01 eV/Å
7. **Phase 3 — P-RFO TS refinement** (up to 200 steps):
   - Partitioned Rational Function Optimization: maximize along TS mode, minimize along all others
   - Drives force from ~0.01 to ~1e-5 eV/Å in 10-20 steps
   - Trust region with adaptive radius
   - Tight convergence: force_norm < 1e-4 eV/Å

## Key design decisions

### Early handoff (NR → GAD)

GAD requires at least one negative Hessian eigenvalue to follow uphill. If NR converges fully to a minimum (n_neg=0), GAD has no mode to track. Solution: save geometry snapshots during NR at every step where n_neg ≥ 1, then start GAD from these handoff points.

Multiple candidates increase success: the last n_neg=1 snapshot is closest to the minimum but may have lost the saddle direction, while the first n_neg=1 snapshot retains more saddle character. Trying several maximizes the chance one leads to a TS.

### Eckart projection

Raw 3N×3N Hessians include 5–6 near-zero eigenvalues from translation (3) and rotation (2–3). These rigid-body modes are NOT vibrational and must be projected out before counting negative eigenvalues. The Eckart projection (`reduced_basis_hessian_torch()`) constructs the (3N-k)×(3N-k) vibrational Hessian directly, where k = 5 or 6.

For isopropanol (N=12): 36 DOF → 30 vibrational DOF after Eckart projection.

### DFTB0 only — DFTB2/DFTB3 crash during GAD

We tested all available DFTB variants in SCINE Sparrow. Only DFTB0 is viable for the full NR→GAD pipeline.

**DFTB0 (non-self-consistent tight binding):**
- Analytical Hessians — fast, numerically stable, exact derivatives
- Works reliably through both NR and GAD phases
- All production data collected with DFTB0

**DFTB2 (self-consistent charge, SCC-DFTB):**
- Single-point energy/force/Hessian works on both clean and noisy geometries
- **Crashes during GAD dynamics** with: `RuntimeError: Gradient calculation in semi-numerical Hessian evaluation failed`
- The failure occurs because DFTB2 uses **semi-numerical Hessians** (finite-difference of analytical gradients). During GAD, the geometry moves to regions where the SCF procedure fails to converge for the displaced geometries used in the finite-difference stencil, even though it converges for the central geometry.
- Confirmed NOT a multiprocessing issue — crashes identically in single-process mode (`scripts/test_dftb_single_process.py`)
- Confirmed NOT a noise issue — crashes even from clean equilibrium geometry after a few GAD steps

**DFTB3 (3rd-order expansion, DFTB3-D3BJ):**
- Same behavior as DFTB2: single-point works, GAD crashes
- Same root cause: semi-numerical Hessians fail during dynamics

**DFTB1:**
- Does not exist as a method in SCINE Sparrow. The naming goes DFTB0 → DFTB2 → DFTB3.

**Root cause:** SCINE Sparrow's `HessianCalculator` for SCC methods (DFTB2/3) performs central finite differences of the gradient. Each finite-difference displacement creates a slightly perturbed geometry that requires its own SCF convergence. Near saddle points or in distorted geometries (which GAD deliberately explores), these displaced geometries can land in regions where the SCF procedure diverges. DFTB0 avoids this entirely because its Hessian is analytical — no SCF, no finite differences.

**Implication for Step 3:** The adjoint sampler's `grad_E` calls will also need Hessians (for the GAD vector field). Using SCINE directly limits us to DFTB0. A GPU-based ML surrogate trained on DFTB0 data would bypass this limitation entirely.

### Adaptive NR/GAD optimizer (current)

The consolidated optimizer (`saddle_optimizer.py` + `run_ts_search.py`) uses:
- **rms_force** as convergence metric (stricter than mean per-atom force)
- **NR→candidates→GAD pipeline**: NR collects n_neg=0,1,2 snapshots during minimization, then GAD tries each as handoff point
- **Multi-start**: full noise + 60% noise for σ ≥ 0.3 Å
- **100% convergence on 0.3 Å noise** (46/46 valid seeds), up from 74% with the old pipeline

Key iteration history (adaptive switching attempts):

| Version | Rate | Change |
|---------|------|--------|
| Pure adaptive (per-step NR/GAD) | 0% | NR dominates, GAD never runs |
| Sticky GAD (NR must converge first) | 21% | NR oscillates, misses handoff points |
| + n_neg=2 recovery | 32% | Recovers index-2 stuck cases |
| + multi-start (60% noise) | 47% | More NR starts find more handoff points |
| NR→candidates→GAD pipeline | **100%** | Handoff mechanism is essential |

**P-RFO refinement was attempted and abandoned**: Newton-based methods (P-RFO, eigenvector-following) fail on DFTB0 because the PES is too anharmonic — the quadratic model overshoots even at 0.005 Å displacement. Continued GAD with the same params reduces force from ~0.01 to ~0.001 in ~800 steps.

### State-based optimization

All phases are state-based: the next step depends only on the current geometry and its energy/forces/Hessian, not on path history. This is deliberate — the diffusion model will learn to generate TS geometries from a noise distribution, so the optimization must be compatible with single-point evaluations.

## Convergence improvements (0.3 Å noise)

Starting from 50% baseline, systematically tested improvements:

| Change | Effect | Kept? |
|--------|--------|-------|
| Lower min_dist filter: 0.5 → 0.4 Å | Skip rate 28→15 | Yes |
| More NR steps: 500 → 1000 | More handoff candidates | Yes |
| More GAD steps: 3000 → 5000 | More time to converge | Yes |
| Mid-nneg1 candidate | Catches cases first/last miss | Yes |
| Index-2 recovery kick | Helps stuck n_neg=2 cases | Yes |
| dt_scale_factor = 0.5 | Mixed: ±4 seeds | No |

Final v7 pipeline: **74/100** at 0.3 Å (from 50/100 baseline).

### Consolidated optimizer (v11): 99% convergence

The consolidated `run_ts_search.py` achieves **87/88 = 99%** (100 seeds, 12 skipped by distance filter). Key changes:
- **rms_force** convergence metric (stricter than old mean per-atom force)
- Cleaner NR→candidates→GAD with multi-start (full + 60% noise)
- Single failure: seed 81 (n_neg=3, force=2.88 after 10k GAD steps)

### Why the handoff mechanism is essential

NR from noisy geometries oscillates through Morse indices (n_neg goes 8→3→1→0→2→5→...). The transient low-n_neg moments are ideal GAD launch points but are fleeting — NR overshoots past them in 1-2 steps. Pure adaptive NR/GAD switching (per-step based on n_neg) fails because GAD never gets a sustained run. The fix: let NR run freely, record every n_neg ≤ 2 snapshot, then hand the best to GAD.

### Failure modes

1. **Skipped** (~12%): Noise creates atom clashes (dist < 0.4 Å)
2. **No TS** (~1%): Rare edge cases where no candidate geometry leads to a TS

## Results summary

### Convergence rates by noise level

**v7 (original multi-candidate pipeline):**

| Noise (Å) | TS found | Skip | No handoff | No TS | Rate |
|-----------|----------|------|------------|-------|------|
| 0.2 | 79/92 | 8 | — | 13 | 86% |
| 0.3 | 74/100 | 15 | 3 | 11 | 74% |
| 0.5 | 35/100 | 13 | 8 | 43 | 35% |

**v11 (NR→candidates→GAD + rms_force + multi-start):**

| Noise (Å) | With PLS | No PLS | Notes |
|-----------|----------|--------|-------|
| 0.2 | — | **98%** (49/50) | Near-perfect |
| 0.3 | **99%** (87/88) | **100%** (46/46) | PLS makes no difference |
| 0.5 | — | **91%** (40/44) | Huge improvement over v7's 35% |

Polynomial line search (PLS) has negligible effect — the handoff mechanism is what matters.

### TS diversity increases with noise

Higher noise explores more of the PES, discovering rarer TS:

- 0.2 Å: 4 main TS families
- 0.3 Å: 8 families + many singletons
- 0.5 Å: Even more unique TS (but lower yield)

## Bug fix: `out_new` undefined at high noise

At 0.5 Å noise, the trust-region inner loop in `saddle_optimizer.py` can hit the minimum interatomic distance check on every retry, causing `out_new` to never be assigned. This crashes at the line that references `out_new` after the loop. Fix: initialize `out_new = None` before the loop and skip the step if it remains `None`.

This bug is rare at 0.2–0.3 Å noise (geometries stay reasonable) but triggers ~2% of the time at 0.5 Å.

## Output

Results saved to `/scratch/memoozd/diffusion-ts/parallel_results.json`. Each record:

```json
{
  "seed": 17,
  "status": "ts",
  "final_energy": -310.3619,
  "ts_eig0": -0.5828,
  "final_force_norm": 0.0089,
  "ts_coords": [[x, y, z], ...],
  "handoff_type": "last_nneg1",
  "gad_steps": 819
}
```

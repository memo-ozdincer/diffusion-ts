# Step 2: TS Geometry Clustering — Results

## Overview

Cluster the collected TS geometries to identify structurally distinct transition states. Two molecules may look different due to rigid-body orientation or atom labeling but actually be the same TS. Proper alignment handles both.

## Why naive RMSD fails

Given two TS geometries A and B as (12, 3) coordinate arrays:

**Problem 1 — Orientation**: `RMSD(A, B)` depends on how the molecules are oriented in space. Two identical molecules rotated 90° apart have large RMSD. Solution: **Kabsch alignment**.

**Problem 2 — Atom labeling**: Isopropanol has 6 methyl H atoms. Swapping H₆ and H₇ gives a physically identical molecule but different coordinate array. Naive RMSD treats this as a different geometry. Solution: **Hungarian matching**.

**Problem 3 — Functional group symmetry**: The two methyl groups (CH₃) can swap entirely. Swapping C₁↔C₂ and their H groups gives the same molecule. This is a discrete symmetry that must be enumerated explicitly.

## Algorithm

### Step 1: Kabsch alignment

For geometries A (to align) and B (target):

```
A_c = A - mean(A)       # center at origin
B_c = B - mean(B)
H = A_c^T @ B_c         # cross-covariance matrix (3×3)
U, Σ, V^T = SVD(H)      # singular value decomposition
d = det(V @ U^T)         # check for reflection
R = V @ diag(1,1,sign(d)) @ U^T   # optimal rotation (proper)
t = mean(B) - R @ mean(A)          # translation
RMSD = sqrt(mean(||R @ A_i + t - B_i||^2))
```

The `sign(d)` correction prevents reflections (improper rotations). Without it, the algorithm could "cheat" by reflecting the molecule through a plane.

### Step 2: Hungarian matching

For each equivalence class of k atoms, build a k×k cost matrix where `C[i,j] = ||atom_i(geom1) - atom_j(geom2)||^2` and solve the assignment problem using `scipy.optimize.linear_sum_assignment` (O(k^3) Hungarian algorithm).

This finds the atom permutation π that minimizes `Σ_i C[i, π(i)]` — the total squared distance.

### Step 3: Methyl group enumeration

Isopropanol's two methyl groups can swap. For each of the 2! = 2 permutations of methyl carbons:

1. Swap the carbon atoms (C₁ ↔ C₂)
2. Co-swap their attached H groups (H₆,H₇,H₈ ↔ H₉,H₁₀,H₁₁)
3. Run Hungarian matching on all 6 methyl H atoms
4. Kabsch-align the permuted+matched geometry
5. Keep the minimum RMSD across all permutations

### Step 4: Hierarchical clustering

1. Compute N×N pairwise aligned RMSD matrix (symmetric, O(N² × k³) per pair)
2. Convert to condensed form for scipy
3. Agglomerative clustering with **average linkage** (robust to outliers)
4. Cut dendrogram at RMSD threshold (0.2 Å)

Average linkage uses the mean inter-cluster RMSD as the merge criterion, which is more stable than complete linkage (max) for our data.

## Isopropanol atom structure

```
Index  Atom  Role                 Equivalence class
0      C     Central carbon       Fixed
1      C     Methyl carbon 1      methyl_C (swappable with 2)
2      C     Methyl carbon 2      methyl_C (swappable with 1)
3      O     Hydroxyl oxygen      Fixed
4      H     H on central C       Fixed
5      H     H on O (hydroxyl)    Fixed
6      H     H on methyl 1        H_methyl (all 6 equivalent)
7      H     H on methyl 1        H_methyl
8      H     H on methyl 1        H_methyl
9      H     H on methyl 2        H_methyl
10     H     H on methyl 2        H_methyl
11     H     H on methyl 2        H_methyl
```

When methyl carbons swap (1↔2), their H groups must co-swap (6,7,8 ↔ 9,10,11). Within each methyl group, the 3 H atoms are freely permutable via Hungarian matching.

## Results (67 TS, 0.2 Å threshold, average linkage)

### RMSD distribution

```
All 2211 pairwise RMSD values:
  5th percentile:  0.021 Å    (same TS, slight convergence variation)
  25th percentile: 0.396 Å    (different TS)
  50th percentile: 0.495 Å    (different TS)
  75th percentile: 0.555 Å    (different TS)
  95th percentile: 1.198 Å    (very different TS)

Largest natural gap: 0.144 → 0.256 Å (delta = 0.112 Å)
  → This gap separates "same TS" from "different TS"
  → Threshold of 0.2 Å sits within this gap
```

### Threshold sensitivity

| Threshold | Clusters | Largest cluster | Notes |
|-----------|----------|----------------|-------|
| 0.1 Å | 28-29 | 10-11 | Over-split |
| 0.2 Å | 27 | 11 | Best separation |
| 0.3 Å | 27 | 11 | Same as 0.2 (gap) |
| 0.5 Å | 16-19 | 22-45 | Under-split, merges families |

The 0.2–0.3 Å range gives identical clustering because of the natural gap at 0.14–0.26 Å.

### TS families

Grouping clusters by energy similarity (< 0.1 eV):

| Family | Members | Energy (eV) | eig0 | Description |
|--------|---------|-------------|------|-------------|
| A | 21 | -310.36 | -0.57 | Most common, shallow saddle |
| B | 16 | -310.03 | -1.87 | Mixed eig0, moderate curvature |
| C | 17 | -308.37 | -2.32 | Steep saddle, largest family |
| D | 10 | -308.23 | -1.22 | Smaller group |
| E | 3 | -311.91 | -2.38 | Deepest TS |

Plus several singletons at unique energies (E = -311.35, -310.88, -309.60, -308.90, -306.87, etc.)

### Key findings

1. **Energy groups = structural groups**: Cross-energy RMSD is ~0.5 Å, confirming different energies correspond to genuinely different TS geometries.

2. **Within-family sub-splitting**: Families A and B each split into 2 sub-clusters at 0.2 Å threshold. These are likely mirror-image or H-rotamer variants of the same saddle point.

3. **Higher noise = more diversity**: The 17 additional TS from 0.3 Å noise (convergence improvement run) added new singletons and expanded existing families, confirming that higher noise explores more of the PES.

4. **Borderline TS**: Seeds 45 (eig0 = -0.003) and 83 (eig0 = -0.001) have nearly flat saddle points. These are technically TS by the n_neg=1 criterion but may not be chemically meaningful.

5. **Extreme curvature**: Seed 6 (eig0 = -7.02) has an unusually steep saddle — worth investigating whether this is a real TS or a numerical artifact.

## Output

Clustered results: `/scratch/memoozd/diffusion-ts/clustered_results_67ts.json`

```json
{
  "n_ts": 67,
  "n_clusters": 33,
  "linkage_method": "average",
  "rmsd_threshold": 0.2,
  "clusters": [
    {
      "cluster_id": 1,
      "size": 6,
      "energy_mean": -310.36,
      "representative_seed": 66,
      "representative_coords": [[x, y, z], ...],
      "members": [...]
    }
  ],
  "rmsd_matrix": [[...]]
}
```

## Code

- `src/dependencies/alignment.py` — Kabsch, Hungarian, aligned_rmsd, pairwise_rmsd_matrix
- `scripts/cluster_ts_geometries.py` — Load TS records, cluster, print summary, save results

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

## Clustering method exploration (74 TS)

Tested 131 clustering configurations across 7 algorithm families on the 74×74 aligned RMSD matrix. Compared against energy-based ground truth (16 energy groups at 0.1 eV tolerance).

### Methods tested

| Algorithm | Variants | Best ARI | Best Silhouette |
|-----------|----------|----------|-----------------|
| Hierarchical (complete/average/single) | Thresholds 0.1–0.5 Å, k=3–20 | 0.035 | 0.598 |
| Hierarchical (Ward) | On MDS embedding, thresholds + k | 0.041 | 0.562 |
| DBSCAN | eps=0.05–0.5, min_samples=2,3,5 | 0.039 | **0.940** |
| OPTICS | min_samples=2,3,5 | 0.032 | 0.768 |
| KMeans | k=3–20 on MDS embedding | 0.038 | 0.477 |
| Spectral | σ=median,p25; k=3–10 | **0.050** | 0.365 |
| Agglomerative (sklearn) | complete/average/single, k=3–10 | 0.035 | 0.598 |

### Key finding: RMSD clustering ≠ energy clustering

**All methods have ARI < 0.05** — RMSD-based structural clustering does NOT correlate with energy-based grouping. This means:

1. **Structurally similar TS can have very different energies.** The RMSD vs energy scatter (below) shows no correlation between pairwise RMSD and pairwise energy difference.

2. **Intra-energy-group RMSD overlaps with inter-group RMSD.** The "natural gap" at 0.14–0.26 Å seen with 67 TS disappears with 74 — more data fills in the gap. Energy groups have within-group RMSD spread of 0.3–1.0 Å, comparable to between-group distances.

3. **DBSCAN finds tight structural clusters** (silhouette 0.94 at eps=0.05) that group geometries within 0.05 Å — but these structural clusters span 2+ eV in energy.

4. **Spectral clustering is the least bad** (ARI=0.05) but still essentially random relative to energy.

### Implication for diffusion model training

For the diffusion model, we should cluster by **energy** (or energy + eig0), not by RMSD. The TS families identified in the earlier analysis (A–E) are defined by energy, not structure. Structurally, many distinct TS map to similar RMSD regions.

### Visualizations

- `docs/figures/clustering_exploration/rmsd_vs_energy_scatter.png` — Pairwise RMSD vs energy/curvature difference
- `docs/figures/clustering_exploration/tsne_perplexity_comparison.png` — t-SNE at perplexity 5/10/15/30
- `docs/figures/clustering_exploration/top_methods_mds_grid.png` — Top 6 methods on MDS embedding
- `docs/figures/clustering_exploration/best_method_detail.png` — Best method (spectral k=8) detail
- `docs/figures/clustering_exploration/rmsd_band_analysis.png` — Intra vs inter energy group RMSD
- `docs/figures/clustering_exploration/metrics_comparison.png` — ARI / silhouette / energy coherence bars

### What we tried and dropped

| Method | Why dropped |
|--------|-------------|
| Ward linkage on RMSD | Ward requires Euclidean distance, not precomputed. Used MDS embedding instead — loses information |
| DBSCAN | Finds structurally tight clusters but they don't correspond to chemical TS families |
| KMeans on MDS | Assumes spherical clusters in MDS space, which is not the structure |
| High silhouette methods | High internal quality ≠ high energy coherence |
| Low perplexity t-SNE (5) | Over-fragments into micro-clusters |

### What we kept

| Method | Why kept |
|--------|---------|
| Energy-based clustering (tol=0.1 eV) | Only clustering that captures chemical TS families |
| Hierarchical average linkage + RMSD threshold | Identifies structural sub-variants within energy families |
| MDS embedding | Good for visualization, preserves global RMSD structure |
| t-SNE perplexity=15 | Best balance of local and global structure |

## Two-level structural clustering (87 TS, v3)

With 87 TS from the adaptive optimizer (99% convergence at 0.3 A noise), re-did structural clustering with a two-level hierarchy.

### RMSD distribution (87 TS, 3741 pairs)

```
  min=0.0001, max=2.217 A
  mean=0.718,  median=0.624 A
  std=0.345
  5th percentile:  0.354 A
  25th percentile: 0.497 A
  50th percentile: 0.624 A
  95th percentile: 1.402 A
```

The "natural gap" at 0.14–0.26 A seen with 67 TS still exists but is narrower. k-NN distance plot shows ~50 TS have very close neighbors (NN < 0.1 A), while ~37 are more structurally isolated.

### Method: Two-level hierarchical clustering

**Level 1 (L1=0.15 A)**: Tight cores — near-duplicate TS from the same basin. Produces 41 cores (10 multi-member, 31 singletons).

**Level 2 (L2=0.4 A)**: Group cores into broader structural families using mean inter-core RMSD. Produces 35 families (8 multi-member covering 60 TS, 27 singletons).

Why two levels: A single threshold can't capture both "identical TS" (RMSD < 0.15 A) and "same structural motif" (RMSD < 0.4 A). Two levels avoid the mega-cluster problem (0.5 A threshold merges unrelated families into one 35-member group).

### Structural families (L1=0.15 A, L2=0.4 A)

| Family | Size | Cores | E_mean (eV) | E_std | eig0 | RMSD_max | Structural signature |
|--------|------|-------|-------------|-------|------|----------|---------------------|
| F0 | 22 | 3 | -309.27 | 0.826 | -2.23 | 0.43 | C0-C1=1.67, C0-C2=1.77, angle=110° |
| F1 | 8 | 2 | -310.18 | 0.483 | -1.38 | 0.39 | C0-H=2.0 (H migrating), angle=123° |
| F2 | 8 | 1 | -308.22 | 0.000 | -1.43 | 0.001 | C0-C1=1.69, C0-C2=1.40, angle=75° |
| F3 | 7 | 1 | -308.22 | 0.001 | -1.43 | 0.10 | C0-C1=1.40, C0-C2=1.69, angle=75° (mirror of F2) |
| F4 | 6 | 2 | -309.63 | 0.730 | -3.82 | 0.36 | C0-H=1.64, angle=123° |
| F5 | 5 | 2 | -308.36 | 0.004 | -2.32 | 0.21 | C2-C0-O=73° (bent) |
| F6 | 2 | 2 | -308.51 | 1.529 | -1.52 | 0.31 | C0-C1=2.53 (C-C breaking) |
| F7 | 2 | 1 | -308.21 | 0.000 | -1.44 | 0.004 | Similar to F2/F3 |

Plus 27 singletons reassigned to nearest family for soft labeling.

### Key findings (v3)

1. **F2 and F3 are mirror images**: Same energy (-308.22), same eig0 (-1.43), but C0-C1/C0-C2 distances are swapped (1.69/1.40 vs 1.40/1.69). Cross-RMSD is 0.53 A even with methyl swap — the migrating H positions differ enough to keep them structurally distinct. These represent the same reaction channel with opposite methyl group roles.

2. **F1 features H migration**: C0-H distance of 2.0 A (vs equilibrium ~1.09 A) indicates a hydrogen transfer TS. This family has the lowest energy (-310.18 eV).

3. **F0 is a "catch-all" family**: Contains 3 cores spanning 0.83 eV in energy. At L2=0.4 A it's manageable (22 TS); at L2=0.5 A it absorbs too many cores (35 TS). There may be further sub-structure within F0 that finer analysis could resolve.

4. **Bond fingerprints distinguish families**: C0-H distance (1.09 vs 1.64 vs 2.0 A) and C1-C0-C2 angle (75° vs 110° vs 123°) are the most discriminating features. O-H distance (~0.98 A) is constant across all families.

5. **Correlation**: RMSD vs |dE| has r=0.27; RMSD vs |d(eig0)| has r=0.11. Weak but not zero — large structural changes tend to accompany larger energy differences, but there's enormous scatter.

6. **Singletons are genuinely unique**: Median distance from singletons to nearest core TS is 0.79 A. These are not noise — they represent rare TS geometries that the 0.3 A noise displacement accessed infrequently.

### Visualizations

- `docs/figures/structural_clustering_v1/01_rmsd_distribution.png` — RMSD histogram, k-NN plots
- `docs/figures/structural_clustering_v1/02_embeddings_features.png` — MDS/t-SNE colored by energy, eig0, force
- `docs/figures/structural_clustering_v1/03_dendrograms.png` — 4 linkage methods compared
- `docs/figures/structural_clustering_v3/01_main_overview.png` — 9-panel family overview
- `docs/figures/structural_clustering_v3/02_bond_fingerprints.png` — Bond distance distributions
- `docs/figures/structural_clustering_v3/03_angle_fingerprints.png` — Bond angle distributions
- `docs/figures/structural_clustering_v3/04_family_detail.png` — Per-family t-SNE with seed labels
- `docs/figures/structural_clustering_v3/05_full_dendrogram.png` — Full dendrogram with L1/L2 lines

## Reaction type classification

Beyond RMSD-based structural clustering, each TS can be classified by which covalent bonds are significantly stretched (>30% from equilibrium). This provides a chemical interpretation.

### Reaction types (87 TS)

| Type | Count | % | E_mean (eV) | eig0 | Description |
|------|-------|---|-------------|------|-------------|
| Angular/torsional | 37 | 43% | -308.35 ± 0.20 | -2.43 | No bonds broken, angular distortion |
| C-C breaking | 18 | 21% | -309.37 ± 1.24 | -2.12 | Methyl detachment (C0-C1 or C0-C2 > 2.0 A) |
| C-H migration | 13 | 15% | -310.23 ± 0.32 | -0.48 | Central H moving away (C0-H > 1.4 A) |
| Multi-bond | 10 | 11% | -308.99 ± 1.74 | -1.11 | 2+ bonds simultaneously stretched |
| C-O breaking | 3 | 3% | -310.30 ± 1.92 | -1.75 | Hydroxyl leaving (C0-O > 1.86 A) |
| C-C + C-H | 3 | 3% | -307.76 ± 1.64 | -1.86 | C-C and C-H both stretched |
| C-C + O-H | 3 | 3% | -309.13 ± 1.32 | -1.32 | C-C and O-H both stretched |

### Key insights

1. **Angular/torsional TS dominate** (43%): Most TS involve rotational barriers, not bond breaking. These have the tightest energy distribution (-308.35 ± 0.20) and highest curvature (eig0 = -2.43).

2. **C-H migration TS are distinctive**: They have near-zero eig0 (-0.48), making them very flat saddle points. They're also the lowest energy (-310.23) among common types.

3. **F0 sub-structure explained**: The mega-family F0 (n=22) from RMSD clustering actually contains 3 distinct reaction types: angular (n=10 at E≈-308.36), C-C breaking C1 (n=5 at E≈-310.02), C-C breaking C2 (n=7 at E≈-310.02). These merge because C-C stretched geometries are within 0.4 A RMSD of each other after methyl swap.

4. **Singletons are multi-bond TS**: Most singletons involve 2+ bonds simultaneously stretched — these are rare, high-energy TS that the 0.3 A noise displacement occasionally accesses.

### Visualization

- `docs/figures/structural_clustering_v3/06_reaction_types.png` — 6-panel: t-SNE by type/energy/eig0, boxplots, pie chart

### Data

- RMSD matrix: `/scratch/memoozd/diffusion-ts/structural_clustering/rmsd_matrix_87.npy`
- Family assignments: `/scratch/memoozd/diffusion-ts/structural_clustering_v3/families_v3.json`
- Reaction type classification: `/scratch/memoozd/diffusion-ts/structural_clustering_v3/reaction_types.json`

## HDBSCAN comparison (87 TS)

Tested HDBSCAN (variable-density DBSCAN) alongside hierarchical and DBSCAN methods.

| Method | Clusters | Noise | Silhouette | Coverage |
|--------|----------|-------|------------|----------|
| HDBSCAN mcs=3, ms=2 | 9 | 31 | **0.871** | 64% |
| HDBSCAN mcs=4, ms=2 | 8 | 31 | 0.821 | 64% |
| HDBSCAN mcs=2, ms=1 | 16 | 28 | 0.715 | 68% |
| DBSCAN eps=0.15, ms=2 | 42 | 33 | — | 62% |
| Hierarchical avg t=0.4 | 37 | — | — | 64% |

**Best balanced method**: HDBSCAN mcs=3 produces 9 high-quality structural clusters with 0.871 silhouette, covering 56/87 TS. The remaining 31 are genuinely structurally isolated.

## Bonding pattern analysis (chemical reaction types)

Each TS family corresponds to a specific bond-breaking/forming process. Analysis: compare each TS's bond lengths against the equilibrium isopropanol geometry to identify which bonds are stretched/compressed.

| Family | Size | Reaction Type | Key Bond Changes |
|--------|------|---------------|-----------------|
| F1 | 35 | **Methyl H → O migration** | C1-H₇: 1.09→2.17 A (+1.08), C1-O: 2.38→1.75 A (-0.63) |
| F2 | 9 | **Methyl H migration + C-C ring closure** | C1-H₈: 1.10→2.35 A (+1.26), C1-C2: 2.54→1.90 A (-0.64) |
| F3 | 8 | **Central H migration** | C0-H₄: 1.10→2.12 A (+1.03) |
| F4 | 7 | **Central H migration (steep saddle)** | C0-H₄: 1.10→2.12 A (+1.03), eig0=-3.6 |
| F5 | 3 | **C-C dissociation** | C0-C1: 1.48→2.72 A (+1.24) |

F3 and F4 share the same reaction coordinate (C0-H₄ migration) but differ by a factor of 2.5× in TS curvature (eig0 = -1.4 vs -3.6), suggesting different PES topographies for the same chemical process.

## Code

- `src/dependencies/alignment.py` — Kabsch, Hungarian, aligned_rmsd, pairwise_rmsd_matrix
- `scripts/cluster_ts.py` — Original clustering script (67 TS)
- `scripts/cluster_structural.py` — v1: single-level analysis (87 TS)
- `scripts/cluster_structural_v2.py` — v2: HDBSCAN + two-level + bonding analysis
- `scripts/cluster_structural_v3.py` — v3: two-level with L2=0.4 A + structural fingerprints
- `scripts/explore_clustering.py` — Compare 131 clustering methods, generate visualizations

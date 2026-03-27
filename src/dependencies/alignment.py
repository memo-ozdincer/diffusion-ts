"""Kabsch alignment + Hungarian atom matching for molecular geometry comparison.

Provides RMSD-based comparison of molecular geometries accounting for:
- Rigid-body alignment (Kabsch algorithm via SVD)
- Atom permutation symmetry (Hungarian algorithm via linear_sum_assignment)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations


def kabsch_align(
    A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align geometry A onto geometry B using the Kabsch algorithm.

    Args:
        A: (N, 3) coordinates to be rotated/translated.
        B: (N, 3) target coordinates.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation (centroid_B - R @ centroid_A).
        rmsd: RMSD between aligned A and B.
    """
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A_c = A - centroid_A
    B_c = B - centroid_B

    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(V @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = V @ sign_matrix @ U.T

    t = centroid_B - R @ centroid_A

    A_aligned = (R @ A.T).T + t
    diff = A_aligned - B
    rmsd = np.sqrt((diff**2).sum() / len(A))
    return R, t, rmsd


def hungarian_match(
    coords1: np.ndarray,
    coords2: np.ndarray,
    equiv_classes: dict[str, list[int]],
) -> np.ndarray:
    """Find optimal atom permutation matching coords1 to coords2.

    For each equivalence class with >1 atom, uses the Hungarian algorithm
    to find the assignment minimizing total squared distance.

    Args:
        coords1: (N, 3) first geometry.
        coords2: (N, 3) second geometry.
        equiv_classes: mapping of class name to list of atom indices.

    Returns:
        perm: (N,) permutation array such that coords1[perm] best matches coords2.
    """
    n = len(coords1)
    perm = np.arange(n)

    for indices in equiv_classes.values():
        if len(indices) <= 1:
            continue
        idx = np.array(indices)
        # Cost matrix: squared distances between all pairs
        cost = np.sum(
            (coords1[idx][:, None, :] - coords2[None, idx, :]) ** 2, axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[idx[row_ind]] = idx[col_ind]

    return perm


def aligned_rmsd(
    geom1: np.ndarray,
    geom2: np.ndarray,
    equiv_classes: dict[str, list[int]],
    methyl_carbons: list[int] | None = None,
    methyl_hydrogens: list[list[int]] | None = None,
) -> float:
    """Compute minimum aligned RMSD between two geometries.

    Tries all methyl carbon permutations (if provided), applies Hungarian
    matching for equivalent atoms, then Kabsch alignment.

    Args:
        geom1: (N, 3) first geometry.
        geom2: (N, 3) second geometry.
        equiv_classes: base equivalence classes (for non-methyl atoms).
        methyl_carbons: indices of methyl carbons that can swap.
        methyl_hydrogens: list of H-index lists, one per methyl carbon
            (same order as methyl_carbons).

    Returns:
        Minimum RMSD over all valid permutations.
    """
    if methyl_carbons is None or methyl_hydrogens is None:
        # Simple case: just Hungarian + Kabsch
        perm = hungarian_match(geom1, geom2, equiv_classes)
        _, _, rmsd = kabsch_align(geom1[perm], geom2)
        return rmsd

    best_rmsd = float("inf")

    # Precompute invariants outside the permutation loop
    carbon_to_idx = {c: i for i, c in enumerate(methyl_carbons)}
    all_methyl_h = [h for group in methyl_hydrogens for h in group]

    # Equiv classes for Hungarian: fixed atoms + all methyl H as one group
    base_classes = {
        name: indices
        for name, indices in equiv_classes.items()
        if name not in ("methyl_C", "H_methyl")
    }
    base_classes["H_methyl"] = all_methyl_h

    for carbon_perm in permutations(methyl_carbons):
        coord_perm = np.arange(len(geom1))

        for orig, new in zip(methyl_carbons, carbon_perm):
            coord_perm[orig] = new

        for orig_c, new_c in zip(methyl_carbons, carbon_perm):
            orig_hs = methyl_hydrogens[carbon_to_idx[orig_c]]
            new_hs = methyl_hydrogens[carbon_to_idx[new_c]]
            for h_orig, h_new in zip(orig_hs, new_hs):
                coord_perm[h_orig] = h_new

        geom1_permuted = geom1[coord_perm]
        perm = hungarian_match(geom1_permuted, geom2, base_classes)
        _, _, rmsd = kabsch_align(geom1_permuted[perm], geom2)
        if rmsd < best_rmsd:
            best_rmsd = rmsd

    return best_rmsd


def pairwise_rmsd_matrix(
    geometries: list[np.ndarray],
    equiv_classes: dict[str, list[int]],
    methyl_carbons: list[int] | None = None,
    methyl_hydrogens: list[list[int]] | None = None,
) -> np.ndarray:
    """Compute pairwise aligned RMSD matrix for a list of geometries.

    Args:
        geometries: list of (N, 3) coordinate arrays.
        equiv_classes: atom equivalence classes.
        methyl_carbons: indices of swappable methyl carbons.
        methyl_hydrogens: H-index lists per methyl carbon.

    Returns:
        (M, M) symmetric distance matrix where M = len(geometries).
    """
    n = len(geometries)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = aligned_rmsd(
                geometries[i],
                geometries[j],
                equiv_classes,
                methyl_carbons,
                methyl_hydrogens,
            )
            D[i, j] = rmsd
            D[j, i] = rmsd
    return D

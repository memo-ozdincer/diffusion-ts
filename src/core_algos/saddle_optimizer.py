"""NR→GAD transition state optimizer.

Two-phase saddle point search:
  Phase 1 — RFO Newton-Raphson minimization (find local minimum)
  Phase 2 — GAD with eigenvalue-clamped adaptive dt (climb to index-1 saddle)

Convergence criterion: n_neg == 1 (exactly one negative vibrational eigenvalue
after Eckart projection). No additional filtering.

Core features (general-purpose, proven essential):
  - RFO (augmented Hessian): adaptive shift, no hyperparameters, guaranteed downhill
  - Polynomial line search (PLS): cubic interpolation refinement on accepted TR steps
  - Trust region with floor: prevents collapse into micro-steps
  - Relaxed eigenvalue convergence: catches near-minima with tiny ghost negatives
  - GAD mode tracking: consistent eigenvector across steps despite degeneracies
  - Eigenvalue-clamped adaptive dt: self-regulating step size for GAD phase

Optional molecular-specific features (off by default):
  - Oscillation kick: perturb along negative eigenvector when energy oscillates
  - Blind-mode kick: perturb when gradient is orthogonal to negative modes
  - Late escape: aggressive displacement after many stagnant steps
  - Min interatomic distance check: prevent unphysical atomic clashes

Usage:
    result = find_transition_state(
        predict_fn, coords0, atomic_nums, atomsymbols,
        nr_steps=50000, gad_steps=500,
    )
    if result["converged"]:
        ts_coords = result["final_coords"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import torch

from src.dependencies.differentiable_projection import (
    reduced_basis_hessian_torch,
    project_vector_to_vibrational_torch,
    gad_dynamics_projected_torch,
)
from src.core_algos.gad import pick_tracked_mode


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NRConfig:
    """Newton-Raphson (Phase 1) configuration.

    Defaults are the v13 "good general" config — RFO + PLS + trust region.
    """
    n_steps: int = 50_000
    max_atom_disp: float = 1.3          # max per-atom displacement (Angstrom)
    force_converged: float = 1e-4       # force convergence threshold (eV/A)
    trust_radius_floor: float = 0.01    # minimum trust radius (Angstrom)
    polynomial_linesearch: bool = True   # cubic interpolation refinement
    project_gradient: bool = True        # Eckart-project gradient and eigenvectors
    # Relaxed convergence: accept min_eval >= -threshold as "converged"
    relaxed_eval_threshold: float = 0.01
    accept_relaxed: bool = True
    # Molecular safety (set to 0.0 to disable)
    min_interatomic_dist: float = 0.5
    # Logging
    log_spectrum_k: int = 10


@dataclass
class GADConfig:
    """GAD (Phase 2) configuration.

    Eigenvalue-clamped adaptive dt is the default (93-100% TS rate in benchmarks).
    """
    n_steps: int = 500
    dt: float = 0.003                   # base timestep
    dt_min: float = 1e-5
    dt_max: float = 0.1
    dt_adaptation: str = "eigenvalue_clamped"  # best general strategy
    dt_scale_factor: float = 1.0
    max_atom_disp: float = 0.35         # max per-atom displacement per step
    track_mode: bool = True             # eigenvector continuity tracking
    project_gradient: bool = True        # Eckart-project gradient and v
    # Molecular safety
    min_interatomic_dist: float = 0.5
    # Index-2 recovery: when stuck at n_neg=2 with low force, perturb along
    # 2nd negative eigenvector. This can recover 5-10% of failures at high noise.
    index2_recovery: bool = False
    index2_patience: int = 200          # consecutive n_neg=2 steps before kick
    index2_kick_scale: float = 0.3      # displacement magnitude (Angstrom)
    index2_max_kicks: int = 3           # max number of kicks per run


@dataclass
class SaddleOptimizerConfig:
    """Full NR→GAD optimizer config."""
    nr: NRConfig = field(default_factory=NRConfig)
    gad: GADConfig = field(default_factory=GADConfig)
    gad_on_nr_failure: bool = True      # run GAD even if NR didn't converge


# ============================================================================
# Vibrational eigendecomposition (Eckart-projected reduced basis)
# ============================================================================

def vib_eig(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: list[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vibrational eigenvalues/vectors via reduced basis (no threshold filtering).

    Returns:
        evals_vib:    (3N-k,)    vibrational eigenvalues, ascending
        evecs_vib_3N: (3N, 3N-k) eigenvectors in full Cartesian space
        Q_vib:        (3N, 3N-k) orthonormal vibrational basis
    """
    n_atoms = coords.reshape(-1, 3).shape[0]
    hess = hessian.reshape(3 * n_atoms, 3 * n_atoms)
    rb = reduced_basis_hessian_torch(hess, coords.reshape(-1, 3), atomsymbols)
    evals_vib, evecs_red = torch.linalg.eigh(rb["H_red"])
    evecs_vib_3N = rb["Q_vib"] @ evecs_red
    return evals_vib, evecs_vib_3N, rb["Q_vib"]


# ============================================================================
# RFO secular equation solver
# ============================================================================

def _solve_rfo_secular(
    evals: torch.Tensor,
    g_proj: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> Tuple[float, int]:
    """Solve the RFO secular equation for the lowest eigenvalue of the
    augmented Hessian (Schlegel 2011, Eq. 20).

    Finds mu < lam_min such that:  sum c_i^2 / (lam_i - mu) + mu = 0
    """
    c2 = g_proj ** 2
    lam_min = float(evals.min().item())

    g_norm_sq = float(c2.sum().item())
    if g_norm_sq < 1e-30:
        return 0.0, 0

    mu = lam_min - (g_norm_sq ** 0.5) - 0.1

    for n_iter in range(1, max_iter + 1):
        denom = evals - mu
        denom_safe = torch.where(
            denom.abs() < 1e-15, torch.ones_like(denom) * 1e-15, denom
        )
        f_val = float((c2 / denom_safe).sum().item()) + mu
        f_deriv = float((c2 / (denom_safe ** 2)).sum().item()) + 1.0

        if abs(f_val) < tol * max(abs(mu), 1.0):
            break
        if abs(f_deriv) < 1e-30:
            break

        mu_new = mu - f_val / f_deriv
        mu = min(mu_new, lam_min - 1e-8)

    return mu, n_iter


def _rfo_step(
    grad: torch.Tensor,
    V: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """RFO step: augmented Hessian with adaptive shift, guaranteed downhill."""
    coeffs = V.T @ grad
    mu_rfo, secular_iters = _solve_rfo_secular(lam, coeffs)

    shifted = torch.clamp(lam - mu_rfo, min=1e-15)
    step_coeffs = -coeffs / shifted
    delta_x = V @ step_coeffs

    info = {
        "mu_rfo": mu_rfo,
        "secular_iterations": secular_iters,
        "step_norm": float(delta_x.norm().item()),
    }
    return delta_x, info


# ============================================================================
# Polynomial line search (cubic interpolation)
# ============================================================================

def _cubic_interval_minimum(
    energy_prev: float,
    energy_curr: float,
    deriv_prev: float,
    deriv_curr: float,
    t_lo: float = 0.1,
    t_hi: float = 0.9,
) -> Optional[Dict[str, float]]:
    """Fit cubic p(t) on [0,1] matching E and dE/dt at endpoints.
    Return interior minimum if it exists and is lower than E_curr."""
    d = energy_prev
    c = deriv_prev
    y1 = energy_curr - energy_prev - deriv_prev
    y2 = deriv_curr - deriv_prev
    a = y2 - 2.0 * y1
    b = 3.0 * y1 - y2

    candidates: List[float] = []
    eps = 1e-14

    if abs(a) < eps:
        if abs(b) > eps:
            candidates.append(-c / (2.0 * b))
    else:
        disc = 4.0 * b * b - 12.0 * a * c
        if disc >= 0.0:
            sqrt_disc = disc ** 0.5
            denom = 6.0 * a
            if abs(denom) > eps:
                candidates.append((-2.0 * b - sqrt_disc) / denom)
                candidates.append((-2.0 * b + sqrt_disc) / denom)

    best_t = None
    best_e = float("inf")
    for t in candidates:
        if not (t_lo < t < t_hi):
            continue
        if 6.0 * a * t + 2.0 * b <= 0.0:
            continue
        e_t = ((a * t + b) * t + c) * t + d
        if e_t < best_e:
            best_t, best_e = t, e_t

    if best_t is None or best_e >= energy_curr:
        return None
    return {"t_star": float(best_t), "predicted_energy": float(best_e)}


# ============================================================================
# Utility helpers
# ============================================================================

def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _cap_displacement(step_disp: torch.Tensor, max_disp: float) -> torch.Tensor:
    disp_3d = step_disp.reshape(-1, 3)
    max_actual = float(disp_3d.norm(dim=1).max().item())
    if max_actual > max_disp and max_actual > 0:
        disp_3d = disp_3d * (max_disp / max_actual)
    return disp_3d.reshape(step_disp.shape)


def _min_interatomic_distance(coords: torch.Tensor) -> float:
    c = coords.reshape(-1, 3)
    n = c.shape[0]
    if n < 2:
        return float("inf")
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(n, device=c.device, dtype=c.dtype) * 1e10
    return float(dist.min().item())


# ============================================================================
# Phase 1: Newton-Raphson minimization with RFO + PLS + trust region
# ============================================================================

def run_nr_minimization(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: NRConfig,
) -> Dict[str, Any]:
    """RFO Newton-Raphson minimization to local minimum.

    Converges when n_neg == 0 (or relaxed: min_eval >= -threshold) AND force < threshold.

    Returns dict with keys: converged, final_coords, final_energy, final_force_norm,
    converged_step, total_steps, final_n_neg, final_min_eval, trajectory.
    """
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    trust_radius = cfg.max_atom_disp
    trajectory: List[Dict[str, Any]] = []

    # Initial evaluation
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    for step in range(cfg.n_steps):
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        force_norm = _force_mean(forces)

        # Vibrational eigendecomposition
        evals_vib, evecs_vib_3N, _ = vib_eig(hessian, coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item())
        min_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")

        # Log
        record = {
            "step": step, "energy": energy, "force_norm": force_norm,
            "n_neg": n_neg, "min_eval": min_eval, "trust_radius": trust_radius,
            "coords": coords.detach().cpu().clone(),
        }
        if cfg.log_spectrum_k > 0 and evals_vib.numel() > 0:
            k = min(cfg.log_spectrum_k, evals_vib.numel())
            record["bottom_spectrum"] = evals_vib[:k].tolist()
        trajectory.append(record)

        # ---- Convergence check ----
        strict = n_neg == 0 and force_norm < cfg.force_converged
        relaxed = (
            not strict
            and cfg.relaxed_eval_threshold > 0
            and force_norm < cfg.force_converged
            and min_eval >= -cfg.relaxed_eval_threshold
        )
        if strict or (cfg.accept_relaxed and relaxed):
            return {
                "converged": True,
                "convergence_class": "STRICT" if strict else "RELAXED",
                "converged_step": step,
                "final_coords": coords.detach().cpu(),
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_n_neg": n_neg,
                "final_min_eval": min_eval,
                "total_steps": step + 1,
                "trajectory": trajectory,
            }

        # ---- Build RFO step ----
        grad = -forces.reshape(-1)
        if cfg.project_gradient:
            grad = -project_vector_to_vibrational_torch(
                forces.reshape(-1), coords, atomsymbols,
            )

        work_dtype = grad.dtype
        V = evecs_vib_3N.to(device=grad.device, dtype=work_dtype)
        lam = evals_vib.to(device=grad.device, dtype=work_dtype)

        delta_x, rfo_info = _rfo_step(grad, V, lam)
        record["rfo_info"] = rfo_info

        # ---- Trust region with PLS ----
        accepted = False
        retries = 0
        max_retries = 10
        out_new = None
        new_coords = None

        while not accepted and retries < max_retries:
            capped_disp = _cap_displacement(delta_x.reshape(-1, 3), trust_radius)
            dx_flat = capped_disp.reshape(-1).to(work_dtype)
            dx_red = V.T @ dx_flat
            pred_dE = float((grad.dot(dx_flat) + 0.5 * (lam * dx_red * dx_red).sum()).item())

            new_coords = coords + capped_disp

            # Interatomic distance check (molecular safety)
            if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
                trust_radius = max(trust_radius * 0.5, cfg.trust_radius_floor)
                retries += 1
                continue

            out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            energy_new = _to_float(out_new["energy"])
            actual_dE = energy_new - energy

            if actual_dE <= 1e-5:
                accepted = True
                accepted_disp = capped_disp
                accepted_coords = new_coords.detach()
                accepted_out = out_new
                accepted_energy = energy_new
                pred_dE_used = pred_dE

                # Polynomial line search refinement
                if cfg.polynomial_linesearch and step > 0:
                    step_vec = capped_disp.reshape(-1).to(work_dtype)
                    step_norm = float(step_vec.norm().item())
                    if step_norm > 1e-12:
                        grad_prev = -forces.reshape(-1).to(work_dtype)
                        forces_new = out_new["forces"]
                        if forces_new.dim() == 3 and forces_new.shape[0] == 1:
                            forces_new = forces_new[0]
                        grad_curr = -forces_new.reshape(-1).to(work_dtype)

                        deriv_prev = float(grad_prev.dot(step_vec).item())
                        deriv_curr = float(grad_curr.dot(step_vec).item())

                        cubic = _cubic_interval_minimum(
                            energy, energy_new, deriv_prev, deriv_curr,
                        )
                        if cubic is not None:
                            refined_disp = capped_disp * cubic["t_star"]
                            refined_coords = coords + refined_disp

                            valid = (
                                cfg.min_interatomic_dist <= 0
                                or _min_interatomic_distance(refined_coords) >= cfg.min_interatomic_dist
                            )
                            if valid:
                                out_ref = predict_fn(
                                    refined_coords, atomic_nums,
                                    do_hessian=True, require_grad=False,
                                )
                                ref_energy = _to_float(out_ref["energy"])
                                if ref_energy < accepted_energy - 1e-8:
                                    accepted_disp = refined_disp
                                    accepted_coords = refined_coords.detach()
                                    accepted_out = out_ref
                                    accepted_energy = ref_energy
                                    dx_ref = accepted_disp.reshape(-1).to(work_dtype)
                                    dx_ref_red = V.T @ dx_ref
                                    pred_dE_used = float(
                                        (grad.dot(dx_ref) + 0.5 * (lam * dx_ref_red * dx_ref_red).sum()).item()
                                    )
                                    record["pls_applied"] = True

                # Trust radius update
                actual_dE_final = accepted_energy - energy
                rho = actual_dE_final / pred_dE_used if pred_dE_used < -1e-8 else 0.0

                if rho > 0.75:
                    trust_radius = min(trust_radius * 1.5, cfg.max_atom_disp)
                elif rho < 0.25:
                    trust_radius = max(trust_radius * 0.5, cfg.trust_radius_floor)

                coords = accepted_coords
                out = accepted_out
            else:
                trust_radius = max(trust_radius * 0.25, cfg.trust_radius_floor)
                retries += 1

        if not accepted:
            # All retries failed — take last tried step if we evaluated it,
            # otherwise stay at current position (all retries hit distance check)
            if out_new is not None and new_coords is not None:
                coords = new_coords.detach()
                out = out_new

    # Did not converge within budget
    return {
        "converged": False,
        "convergence_class": "NONE",
        "converged_step": None,
        "final_coords": coords.detach().cpu(),
        "final_energy": _to_float(out["energy"]),
        "final_force_norm": force_norm,
        "final_n_neg": n_neg,
        "final_min_eval": min_eval,
        "total_steps": cfg.n_steps,
        "trajectory": trajectory,
    }


# ============================================================================
# Phase 2: GAD saddle search with eigenvalue-clamped adaptive dt
# ============================================================================

def _compute_adaptive_dt(
    dt_base: float,
    dt_min: float,
    dt_max: float,
    method: str,
    eig_0: float,
    eps: float = 1e-8,
    dt_scale_factor: float = 1.0,
) -> float:
    """State-based adaptive timestep. No path history needed.

    Args:
        dt_scale_factor: Global multiplier on the effective dt. Use < 1.0
            (e.g. 0.5) at high noise for stability.
    """
    if method == "none":
        return dt_base * dt_scale_factor

    if method == "eigenvalue_clamped":
        # Clamp |lambda_0| to [1e-2, 1e2] to prevent extreme dt.
        # Small |lambda| (flat) → large dt. Large |lambda| (curved) → small dt.
        lam = min(max(abs(eig_0), 1e-2), 1e2)
        dt_eff = dt_base / (lam + eps)
    elif method == "harmonic":
        omega = math.sqrt(abs(eig_0) + eps)
        dt_eff = dt_base / omega
    else:
        dt_eff = dt_base

    dt_eff *= dt_scale_factor
    return float(max(dt_min, min(dt_eff, dt_max)))


def run_gad_saddle_search(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: GADConfig,
    force_converged: float = 0.01,
) -> Dict[str, Any]:
    """GAD dynamics to find index-1 saddle point.

    Convergence: n_neg == 1 AND force_norm < force_converged on
    Eckart-projected vibrational Hessian.

    Returns dict with keys: converged, final_coords, converged_step, total_steps,
    final_morse_index, trajectory.
    """
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    v_prev: Optional[torch.Tensor] = None
    trajectory: List[Dict[str, Any]] = []

    # Index-2 recovery state
    consec_nneg2: int = 0
    n_kicks_used: int = 0

    for step in range(cfg.n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        num_atoms = int(forces.shape[0])

        # Vibrational eigendecomposition
        evals_vib, evecs_vib_3N, _ = vib_eig(hessian, coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else 0
        eig_0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0

        record = {
            "step": step,
            "energy": _to_float(out["energy"]),
            "force_norm": _force_mean(forces),
            "n_neg": n_neg,
            "eig_0": eig_0,
        }
        trajectory.append(record)

        # ---- Convergence: exactly one negative eigenvalue AND force converged ----
        force_norm = _force_mean(forces)
        if n_neg == 1 and force_norm < force_converged:
            return {
                "converged": True,
                "converged_step": step,
                "final_coords": coords.detach().cpu(),
                "final_energy": _to_float(out["energy"]),
                "final_force_norm": force_norm,
                "final_morse_index": 1,
                "total_steps": step + 1,
                "trajectory": trajectory,
            }

        # ---- Index-2 recovery: perturb along 2nd negative eigenvector ----
        if cfg.index2_recovery:
            if n_neg >= 2:
                consec_nneg2 += 1
            else:
                consec_nneg2 = 0

            if (consec_nneg2 >= cfg.index2_patience
                    and n_kicks_used < cfg.index2_max_kicks
                    and evecs_vib_3N.shape[1] >= 2):
                # Perturb along the 2nd negative eigenvector (index 1)
                v2 = evecs_vib_3N[:, 1].to(device=coords.device, dtype=coords.dtype)
                v2 = v2 / (v2.norm() + 1e-12)
                kick = cfg.index2_kick_scale * v2.reshape(-1, 3)
                coords = coords + kick
                consec_nneg2 = 0
                n_kicks_used += 1
                v_prev = None  # reset mode tracking after kick
                record["index2_kick"] = True
                continue  # re-evaluate at new position

        # ---- Mode tracking: pick guide vector ----
        # Use bottom-k eigenvectors from the reduced-basis Hessian
        k_track = min(8, evecs_vib_3N.shape[1])
        V_cand = evecs_vib_3N[:, :k_track].to(device=forces.device, dtype=forces.dtype)
        v_prev_local = (
            v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
            if (cfg.track_mode and v_prev is not None) else None
        )
        v, _mode_idx, _overlap = pick_tracked_mode(V_cand, v_prev_local, k=k_track)

        # ---- Compute GAD direction ----
        if cfg.project_gradient:
            gad_vec, v_proj, _info = gad_dynamics_projected_torch(
                coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
            )
            v = v_proj.reshape(-1)
        else:
            f_flat = forces.reshape(-1)
            gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
            gad_vec = gad_flat.view(num_atoms, 3)

        if cfg.track_mode:
            v_prev = v.detach().clone().reshape(-1)

        # ---- Adaptive timestep ----
        dt_eff = _compute_adaptive_dt(
            cfg.dt, cfg.dt_min, cfg.dt_max,
            cfg.dt_adaptation, eig_0,
            dt_scale_factor=cfg.dt_scale_factor,
        )
        record["dt_eff"] = dt_eff

        # ---- Take step with displacement capping ----
        step_disp = dt_eff * gad_vec
        max_disp = float(step_disp.norm(dim=1).max().item())
        if max_disp > cfg.max_atom_disp and max_disp > 0:
            step_disp = step_disp * (cfg.max_atom_disp / max_disp)

        new_coords = coords + step_disp

        # Interatomic distance safety
        if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    # Did not converge
    final_morse = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else -1
    return {
        "converged": False,
        "converged_step": None,
        "final_coords": coords.detach().cpu(),
        "final_energy": _to_float(out["energy"]),
        "final_force_norm": float(_force_mean(forces)),
        "final_morse_index": final_morse,
        "total_steps": cfg.n_steps,
        "trajectory": trajectory,
    }


# ============================================================================
# Top-level: NR → GAD transition state search
# ============================================================================

def find_transition_state(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: Optional[SaddleOptimizerConfig] = None,
) -> Dict[str, Any]:
    """Find an index-1 saddle point (transition state) from a noisy starting geometry.

    Phase 1: RFO Newton-Raphson minimizes to a local minimum.
    Phase 2: GAD dynamics climbs from the minimum to an index-1 saddle.

    Args:
        predict_fn: Callable(coords, atomic_nums, do_hessian, require_grad) -> dict
        coords0: (N, 3) or (1, N, 3) starting coordinates
        atomic_nums: (N,) atomic numbers
        atomsymbols: ['C', 'H', ...] element symbols
        cfg: Optimizer config (uses defaults if None)

    Returns:
        dict with keys:
            converged: bool — True if n_neg == 1 at final geometry
            final_coords: (N, 3) tensor
            phase1_result: full NR result dict
            phase2_result: full GAD result dict (or None if phase 2 skipped)
            total_steps: combined step count
    """
    if cfg is None:
        cfg = SaddleOptimizerConfig()

    # ====== Phase 1: NR minimization ======
    nr_result = run_nr_minimization(
        predict_fn, coords0, atomic_nums, atomsymbols, cfg.nr,
    )

    nr_converged = nr_result["converged"]
    nr_coords = nr_result["final_coords"]

    # ====== Phase 2: GAD saddle search ======
    gad_result = None
    if nr_converged or cfg.gad_on_nr_failure:
        gad_result = run_gad_saddle_search(
            predict_fn, nr_coords, atomic_nums, atomsymbols, cfg.gad,
        )

    # ====== Assemble result ======
    ts_converged = gad_result is not None and gad_result["converged"]
    final_coords = gad_result["final_coords"] if gad_result is not None else nr_coords
    total_steps = nr_result["total_steps"] + (gad_result["total_steps"] if gad_result else 0)

    return {
        "converged": ts_converged,
        "final_coords": final_coords,
        "final_energy": gad_result["final_energy"] if gad_result else nr_result["final_energy"],
        "total_steps": total_steps,
        "phase1_result": nr_result,
        "phase2_result": gad_result,
    }

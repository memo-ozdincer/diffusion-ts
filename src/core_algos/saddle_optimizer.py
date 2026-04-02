"""Adaptive NR/GAD transition state optimizer.

Single-loop saddle point search that switches between NR and GAD based on
the Morse index (number of negative vibrational eigenvalues):

    n_neg >= 2  →  NR step (minimize to reduce index)
    n_neg < 2   →  GAD step (climb toward index-1 saddle)
    n_neg == 1 AND rms_force < threshold  →  CONVERGED

This replaces the old two-phase NR→GAD pipeline with a unified loop that
adapts automatically. No multi-candidate handoff, no rescue logic needed.

Core features:
  - RFO (augmented Hessian): adaptive shift, guaranteed downhill for NR
  - Polynomial line search (PLS): cubic interpolation refinement on NR steps
  - Trust region with floor: prevents collapse into micro-steps
  - GAD mode tracking: consistent eigenvector across steps
  - Eigenvalue-clamped adaptive dt: self-regulating step size for GAD
  - Three force metrics: rms_force, max_atomic_force, max_force_component
  - Energy divergence guard: abort if energy rises too far

Usage:
    result = find_transition_state(predict_fn, coords, atomic_nums, atomsymbols)
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
# Force metrics
# ============================================================================

def rms_force(forces: torch.Tensor) -> float:
    """RMS of all force components: sqrt(mean(F_i^2))."""
    return float(forces.reshape(-1).square().mean().sqrt().item())


def max_atomic_force(forces: torch.Tensor) -> float:
    """Max per-atom force magnitude: max_i ||F_i||."""
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).max().item())


def max_force_component(forces: torch.Tensor) -> float:
    """Max absolute force component: max |F_ij|."""
    return float(forces.abs().max().item())


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TSOptimizerConfig:
    """Unified NR/GAD transition state optimizer configuration."""
    # Step budget
    max_steps: int = 10_000

    # Force convergence (applied to rms_force by default)
    force_converged: float = 0.01       # eV/A

    # NR parameters (used when n_neg >= 2)
    nr_max_atom_disp: float = 1.3       # trust region max (A)
    nr_trust_floor: float = 0.01        # trust region min (A)
    nr_polynomial_linesearch: bool = True
    nr_project_gradient: bool = True

    # GAD parameters (used when n_neg < 2)
    gad_dt: float = 0.01               # base timestep
    gad_dt_min: float = 1e-5
    gad_dt_max: float = 0.3
    gad_max_atom_disp: float = 0.5     # max per-atom displacement per step
    gad_track_mode: bool = True
    gad_project_gradient: bool = True

    # Molecular safety
    min_interatomic_dist: float = 0.4   # reject steps below this (A), 0 to disable

    # Energy divergence guard
    max_energy_rise: float = 20.0       # abort if E rises more than this from initial

    # Logging
    log_every: int = 100                # print progress every N steps
    log_spectrum_k: int = 5


# Legacy configs for backward compatibility
@dataclass
class NRConfig:
    """Newton-Raphson configuration (legacy, used by run_nr_minimization)."""
    n_steps: int = 50_000
    max_atom_disp: float = 1.3
    force_converged: float = 1e-4
    trust_radius_floor: float = 0.01
    polynomial_linesearch: bool = True
    project_gradient: bool = True
    relaxed_eval_threshold: float = 0.01
    accept_relaxed: bool = True
    min_interatomic_dist: float = 0.5
    log_spectrum_k: int = 10


@dataclass
class GADConfig:
    """GAD configuration (legacy, used by run_gad_saddle_search)."""
    n_steps: int = 500
    dt: float = 0.003
    dt_min: float = 1e-5
    dt_max: float = 0.1
    dt_adaptation: str = "eigenvalue_clamped"
    dt_scale_factor: float = 1.0
    max_atom_disp: float = 0.35
    track_mode: bool = True
    project_gradient: bool = True
    min_interatomic_dist: float = 0.5
    index2_recovery: bool = False
    index2_patience: int = 200
    index2_kick_scale: float = 0.3
    index2_max_kicks: int = 3


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


def _compute_adaptive_dt(
    dt_base: float,
    dt_min: float,
    dt_max: float,
    eig_0: float,
    eps: float = 1e-8,
) -> float:
    """Eigenvalue-clamped adaptive timestep."""
    lam = min(max(abs(eig_0), 1e-2), 1e2)
    dt_eff = dt_base / (lam + eps)
    return float(max(dt_min, min(dt_eff, dt_max)))


# ============================================================================
# Core: Adaptive NR/GAD transition state search
# ============================================================================

def find_transition_state(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: Optional[TSOptimizerConfig] = None,
) -> Dict[str, Any]:
    """Find an index-1 saddle point via adaptive NR/GAD switching.

    Each step:
      1. Compute energy, forces, Hessian → vibrational eigendecomposition
      2. If n_neg >= 2: take RFO Newton-Raphson step (minimize, reduce index)
      3. If n_neg < 2:  take GAD step (climb toward saddle)
      4. If n_neg == 1 and rms_force < threshold: CONVERGED

    Args:
        predict_fn: Callable(coords, atomic_nums, do_hessian, require_grad) -> dict
        coords0: (N, 3) starting coordinates
        atomic_nums: (N,) atomic numbers
        atomsymbols: ['C', 'H', ...] element symbols
        cfg: Optimizer config (uses defaults if None)

    Returns:
        dict with: converged, final_coords, final_energy, final_force_norm,
        final_n_neg, converged_step, total_steps, n_nr_steps, n_gad_steps,
        trajectory.
    """
    if cfg is None:
        cfg = TSOptimizerConfig()

    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    trajectory: List[Dict[str, Any]] = []

    # GAD mode tracking state
    v_prev: Optional[torch.Tensor] = None

    # NR trust radius state
    trust_radius = cfg.nr_max_atom_disp

    # Counters
    n_nr_steps = 0
    n_gad_steps = 0

    # Phase tracking: once we enter GAD (NR converged), stay in GAD
    gad_activated = False

    # Snapshot collection: save geometries where n_neg <= 1 during NR
    # for handoff to GAD if NR never fully converges
    best_snapshot: Optional[Dict[str, Any]] = None  # lowest force with n_neg <= 1

    # Initial energy for divergence guard
    out_init = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    energy_init = _to_float(out_init["energy"])
    out = out_init

    prev_energy = energy_init
    prev_forces = None

    for step in range(cfg.max_steps):
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        # Force metrics
        f_rms = rms_force(forces)
        f_max = max_atomic_force(forces)

        # Vibrational eigendecomposition
        evals_vib, evecs_vib_3N, _ = vib_eig(hessian, coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else 0
        eig_0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0

        # Phase transition: activate GAD once NR has properly converged.
        # NR must reach a genuine minimum (n_neg == 0, force converged) or
        # be very close to a TS (n_neg == 1, force small).
        # "Relaxed" convergence also counts: min_eval >= -0.01 with low force.
        if not gad_activated:
            min_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else 0.0
            nr_converged = (
                (n_neg == 0 and f_rms < 0.01)
                or (n_neg <= 1 and min_eval >= -0.01 and f_rms < 0.01)
            )
            if nr_converged:
                gad_activated = True

        # Collect snapshots: save best low-n_neg geometry during NR for handoff.
        # Priority: n_neg=1 best (already at saddle), then n_neg=0, then n_neg=2.
        if not gad_activated and n_neg <= 2:
            snap_score = {1: 0, 0: 1, 2: 2}.get(n_neg, 3)
            if best_snapshot is None:
                is_better = True
            else:
                old_score = {1: 0, 0: 1, 2: 2}.get(best_snapshot["n_neg"], 3)
                is_better = snap_score < old_score or (snap_score == old_score and f_rms < best_snapshot["force"])
            if is_better:
                best_snapshot = {
                    "coords": coords.detach().cpu().clone(),
                    "force": f_rms,
                    "n_neg": n_neg,
                    "step": step,
                }

        # Decide mode: NR to minimize until convergence, then GAD.
        # Exception: if GAD is stuck at n_neg >= 2 with low force, switch
        # back to NR to re-minimize (the user's core insight).
        if gad_activated and n_neg >= 2 and f_rms < 0.5:
            use_nr = True
            gad_activated = False
            v_prev = None
        else:
            use_nr = not gad_activated

        # Log
        record = {
            "step": step,
            "energy": energy,
            "rms_force": f_rms,
            "max_force": f_max,
            "n_neg": n_neg,
            "eig_0": eig_0,
            "mode": "NR" if use_nr else "GAD",
        }
        trajectory.append(record)

        if cfg.log_every > 0 and step % cfg.log_every == 0:
            mode = "NR" if use_nr else "GAD"
            print(f"  step={step:5d} {mode} n_neg={n_neg} E={energy:.4f} "
                  f"rms_F={f_rms:.6f} max_F={f_max:.6f} eig0={eig_0:.4f}")

        # ---- Convergence check ----
        if n_neg == 1 and f_rms < cfg.force_converged:
            return {
                "converged": True,
                "converged_step": step,
                "final_coords": coords.detach().cpu(),
                "final_energy": energy,
                "final_force_norm": f_rms,
                "final_max_force": f_max,
                "final_n_neg": n_neg,
                "total_steps": step + 1,
                "n_nr_steps": n_nr_steps,
                "n_gad_steps": n_gad_steps,
                "trajectory": trajectory,
            }

        # ---- Energy divergence guard (only during GAD phase) ----
        # NR is allowed to explore freely; GAD should not diverge
        if gad_activated and energy > energy_init + cfg.max_energy_rise:
            record["abort"] = "energy_diverged"
            break

        # ---- Take step: NR or GAD ----
        if use_nr:
            # ============ NR STEP (minimize to reduce Morse index) ============
            record["mode"] = "NR"
            n_nr_steps += 1

            grad = -forces.reshape(-1)
            if cfg.nr_project_gradient:
                grad = -project_vector_to_vibrational_torch(
                    forces.reshape(-1), coords, atomsymbols,
                )

            work_dtype = grad.dtype
            V = evecs_vib_3N.to(device=grad.device, dtype=work_dtype)
            lam = evals_vib.to(device=grad.device, dtype=work_dtype)

            delta_x, rfo_info = _rfo_step(grad, V, lam)
            record["step_norm"] = rfo_info["step_norm"]

            # Trust region
            accepted = False
            retries = 0
            out_new = None
            new_coords = None

            while not accepted and retries < 10:
                capped_disp = _cap_displacement(delta_x.reshape(-1, 3), trust_radius)
                new_coords = coords + capped_disp

                # Distance safety
                if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
                    trust_radius = max(trust_radius * 0.5, cfg.nr_trust_floor)
                    retries += 1
                    continue

                out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
                energy_new = _to_float(out_new["energy"])

                if energy_new <= energy + 1e-5:
                    accepted = True

                    # PLS refinement
                    if cfg.nr_polynomial_linesearch and step > 0 and prev_forces is not None:
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

                            cubic = _cubic_interval_minimum(energy, energy_new, deriv_prev, deriv_curr)
                            if cubic is not None:
                                refined_coords = coords + capped_disp * cubic["t_star"]
                                valid = cfg.min_interatomic_dist <= 0 or _min_interatomic_distance(refined_coords) >= cfg.min_interatomic_dist
                                if valid:
                                    out_ref = predict_fn(refined_coords, atomic_nums, do_hessian=True, require_grad=False)
                                    if _to_float(out_ref["energy"]) < energy_new - 1e-8:
                                        new_coords = refined_coords
                                        out_new = out_ref
                                        energy_new = _to_float(out_ref["energy"])

                    # Trust radius update
                    dx_flat = capped_disp.reshape(-1).to(work_dtype)
                    dx_red = V.T @ dx_flat
                    pred_dE = float((grad.dot(dx_flat) + 0.5 * (lam * dx_red * dx_red).sum()).item())
                    actual_dE = energy_new - energy
                    rho = actual_dE / pred_dE if pred_dE < -1e-8 else 0.0

                    if rho > 0.75:
                        trust_radius = min(trust_radius * 1.5, cfg.nr_max_atom_disp)
                    elif rho < 0.25:
                        trust_radius = max(trust_radius * 0.5, cfg.nr_trust_floor)

                    prev_forces = forces.clone()
                    coords = new_coords.detach()
                    out = out_new
                else:
                    trust_radius = max(trust_radius * 0.25, cfg.nr_trust_floor)
                    retries += 1

            if not accepted:
                if out_new is not None and new_coords is not None:
                    prev_forces = forces.clone()
                    coords = new_coords.detach()
                    out = out_new

        else:
            # ============ GAD STEP (climb toward index-1 saddle) ============
            record["mode"] = "GAD"
            n_gad_steps += 1

            num_atoms = int(forces.shape[0])

            # Mode tracking
            k_track = min(8, evecs_vib_3N.shape[1])
            V_cand = evecs_vib_3N[:, :k_track].to(device=forces.device, dtype=forces.dtype)
            v_prev_local = (
                v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
                if (cfg.gad_track_mode and v_prev is not None) else None
            )
            v, _mode_idx, _overlap = pick_tracked_mode(V_cand, v_prev_local, k=k_track)

            # GAD direction
            if cfg.gad_project_gradient:
                gad_vec, v_proj, _info = gad_dynamics_projected_torch(
                    coords=coords, forces=forces, v=v, atomsymbols=atomsymbols,
                )
                v = v_proj.reshape(-1)
            else:
                f_flat = forces.reshape(-1)
                gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
                gad_vec = gad_flat.view(num_atoms, 3)

            if cfg.gad_track_mode:
                v_prev = v.detach().clone().reshape(-1)

            # Adaptive timestep
            dt_eff = _compute_adaptive_dt(cfg.gad_dt, cfg.gad_dt_min, cfg.gad_dt_max, eig_0)
            record["dt_eff"] = dt_eff

            # Take step
            step_disp = dt_eff * gad_vec
            max_disp = float(step_disp.norm(dim=1).max().item())
            if max_disp > cfg.gad_max_atom_disp and max_disp > 0:
                step_disp = step_disp * (cfg.gad_max_atom_disp / max_disp)

            new_coords = coords + step_disp

            # Distance safety
            if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
                step_disp = step_disp * 0.5
                new_coords = coords + step_disp

            prev_forces = forces.clone()
            coords = new_coords.detach()
            # Reset mode tracking when switching from NR to GAD
            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    # If we exhausted the budget without converging, but collected a good
    # snapshot during NR, try a last-ditch GAD run from that snapshot.
    if not gad_activated and best_snapshot is not None:
        remaining = cfg.max_steps - (step + 1)
        if remaining > 100:
            # Run GAD from the best snapshot
            snap_coords = best_snapshot["coords"].to(coords.device)
            v_prev = None
            gad_activated = True
            coords = snap_coords.reshape(-1, 3)
            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

            for gad_step in range(min(remaining, cfg.max_steps)):
                forces_g = out["forces"]
                hessian_g = out["hessian"]
                if forces_g.dim() == 3 and forces_g.shape[0] == 1:
                    forces_g = forces_g[0]
                forces_g = forces_g.reshape(-1, 3)

                evals_g, evecs_g, _ = vib_eig(hessian_g, coords, atomsymbols)
                n_neg_g = int((evals_g < 0).sum().item())
                f_rms_g = rms_force(forces_g)
                eig_0_g = float(evals_g[0].item())
                n_gad_steps += 1

                if n_neg_g == 1 and f_rms_g < cfg.force_converged:
                    return {
                        "converged": True,
                        "converged_step": step + 1 + gad_step,
                        "final_coords": coords.detach().cpu(),
                        "final_energy": _to_float(out["energy"]),
                        "final_force_norm": f_rms_g,
                        "final_max_force": max_atomic_force(forces_g),
                        "final_n_neg": 1,
                        "total_steps": step + 1 + gad_step + 1,
                        "n_nr_steps": n_nr_steps,
                        "n_gad_steps": n_gad_steps,
                        "trajectory": trajectory,
                        "handoff_snapshot": best_snapshot["step"],
                    }

                # GAD step
                k_track = min(8, evecs_g.shape[1])
                V_cand = evecs_g[:, :k_track].to(device=forces_g.device, dtype=forces_g.dtype)
                v_prev_local = v_prev.to(device=forces_g.device, dtype=forces_g.dtype).reshape(-1) if v_prev is not None else None
                v, _, _ = pick_tracked_mode(V_cand, v_prev_local, k=k_track)

                gad_vec, v_proj, _ = gad_dynamics_projected_torch(
                    coords=coords, forces=forces_g, v=v, atomsymbols=atomsymbols)
                v = v_proj.reshape(-1)
                v_prev = v.detach().clone().reshape(-1)

                dt_eff = _compute_adaptive_dt(cfg.gad_dt, cfg.gad_dt_min, cfg.gad_dt_max, eig_0_g)
                step_disp = dt_eff * gad_vec
                max_d = float(step_disp.norm(dim=1).max().item())
                if max_d > cfg.gad_max_atom_disp and max_d > 0:
                    step_disp = step_disp * (cfg.gad_max_atom_disp / max_d)
                new_coords = coords + step_disp
                if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
                    step_disp = step_disp * 0.5
                    new_coords = coords + step_disp
                coords = new_coords.detach()
                out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

            # Update final state from snapshot GAD
            forces = forces_g
            evals_vib = evals_g

    # Did not converge
    force_final = rms_force(forces)
    n_neg_final = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else -1
    return {
        "converged": False,
        "converged_step": None,
        "final_coords": coords.detach().cpu(),
        "final_energy": _to_float(out["energy"]),
        "final_force_norm": force_final,
        "final_max_force": max_atomic_force(forces),
        "final_n_neg": n_neg_final,
        "total_steps": step + 1 if step > 0 else 0,
        "n_nr_steps": n_nr_steps,
        "n_gad_steps": n_gad_steps,
        "trajectory": trajectory,
    }


# ============================================================================
# Legacy API: run_nr_minimization (unchanged for backward compatibility)
# ============================================================================

def run_nr_minimization(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: NRConfig,
) -> Dict[str, Any]:
    """RFO Newton-Raphson minimization to local minimum (legacy API)."""
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    trust_radius = cfg.max_atom_disp
    trajectory: List[Dict[str, Any]] = []

    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    for step in range(cfg.n_steps):
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        force_norm = rms_force(forces)

        evals_vib, evecs_vib_3N, _ = vib_eig(hessian, coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item())
        min_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")

        record = {
            "step": step, "energy": energy, "force_norm": force_norm,
            "n_neg": n_neg, "min_eval": min_eval, "trust_radius": trust_radius,
            "coords": coords.detach().cpu().clone(),
        }
        if cfg.log_spectrum_k > 0 and evals_vib.numel() > 0:
            k = min(cfg.log_spectrum_k, evals_vib.numel())
            record["bottom_spectrum"] = evals_vib[:k].tolist()
        trajectory.append(record)

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

        grad = -forces.reshape(-1)
        if cfg.project_gradient:
            grad = -project_vector_to_vibrational_torch(
                forces.reshape(-1), coords, atomsymbols,
            )

        work_dtype = grad.dtype
        V = evecs_vib_3N.to(device=grad.device, dtype=work_dtype)
        lam = evals_vib.to(device=grad.device, dtype=work_dtype)

        delta_x, rfo_info = _rfo_step(grad, V, lam)

        accepted = False
        retries = 0
        out_new = None
        new_coords = None

        while not accepted and retries < 10:
            capped_disp = _cap_displacement(delta_x.reshape(-1, 3), trust_radius)
            dx_flat = capped_disp.reshape(-1).to(work_dtype)
            dx_red = V.T @ dx_flat
            pred_dE = float((grad.dot(dx_flat) + 0.5 * (lam * dx_red * dx_red).sum()).item())

            new_coords = coords + capped_disp

            if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
                trust_radius = max(trust_radius * 0.5, cfg.trust_radius_floor)
                retries += 1
                continue

            out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            energy_new = _to_float(out_new["energy"])

            if energy_new <= energy + 1e-5:
                accepted = True
                accepted_coords = new_coords.detach()
                accepted_out = out_new
                accepted_energy = energy_new
                pred_dE_used = pred_dE

                if cfg.polynomial_linesearch and step > 0:
                    step_vec = capped_disp.reshape(-1).to(work_dtype)
                    if float(step_vec.norm().item()) > 1e-12:
                        forces_new = out_new["forces"]
                        if forces_new.dim() == 3 and forces_new.shape[0] == 1:
                            forces_new = forces_new[0]
                        cubic = _cubic_interval_minimum(
                            energy, energy_new,
                            float((-forces.reshape(-1).to(work_dtype)).dot(step_vec).item()),
                            float((-forces_new.reshape(-1).to(work_dtype)).dot(step_vec).item()),
                        )
                        if cubic is not None:
                            refined_coords = coords + capped_disp * cubic["t_star"]
                            valid = cfg.min_interatomic_dist <= 0 or _min_interatomic_distance(refined_coords) >= cfg.min_interatomic_dist
                            if valid:
                                out_ref = predict_fn(refined_coords, atomic_nums, do_hessian=True, require_grad=False)
                                if _to_float(out_ref["energy"]) < accepted_energy - 1e-8:
                                    accepted_coords = refined_coords.detach()
                                    accepted_out = out_ref
                                    accepted_energy = _to_float(out_ref["energy"])

                actual_dE = accepted_energy - energy
                rho = actual_dE / pred_dE_used if pred_dE_used < -1e-8 else 0.0
                if rho > 0.75:
                    trust_radius = min(trust_radius * 1.5, cfg.max_atom_disp)
                elif rho < 0.25:
                    trust_radius = max(trust_radius * 0.5, cfg.trust_radius_floor)

                coords = accepted_coords
                out = accepted_out
            else:
                trust_radius = max(trust_radius * 0.25, cfg.trust_radius_floor)
                retries += 1

        if not accepted and out_new is not None and new_coords is not None:
            coords = new_coords.detach()
            out = out_new

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
# Legacy API: run_gad_saddle_search (unchanged for backward compatibility)
# ============================================================================

def run_gad_saddle_search(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list[str],
    cfg: GADConfig,
    force_converged: float = 0.01,
) -> Dict[str, Any]:
    """GAD dynamics to find index-1 saddle point (legacy API)."""
    coords = coords0.detach().clone().to(torch.float32).reshape(-1, 3)
    v_prev: Optional[torch.Tensor] = None
    trajectory: List[Dict[str, Any]] = []

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

        evals_vib, evecs_vib_3N, _ = vib_eig(hessian, coords, atomsymbols)
        n_neg = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else 0
        eig_0 = float(evals_vib[0].item()) if evals_vib.numel() > 0 else 0.0

        force_norm = rms_force(forces)
        record = {
            "step": step,
            "energy": _to_float(out["energy"]),
            "force_norm": force_norm,
            "n_neg": n_neg,
            "eig_0": eig_0,
        }
        trajectory.append(record)

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

        if cfg.index2_recovery:
            if n_neg >= 2:
                consec_nneg2 += 1
            else:
                consec_nneg2 = 0
            if (consec_nneg2 >= cfg.index2_patience
                    and n_kicks_used < cfg.index2_max_kicks
                    and evecs_vib_3N.shape[1] >= 2):
                v2 = evecs_vib_3N[:, 1].to(device=coords.device, dtype=coords.dtype)
                v2 = v2 / (v2.norm() + 1e-12)
                coords = coords + cfg.index2_kick_scale * v2.reshape(-1, 3)
                consec_nneg2 = 0
                n_kicks_used += 1
                v_prev = None
                continue

        k_track = min(8, evecs_vib_3N.shape[1])
        V_cand = evecs_vib_3N[:, :k_track].to(device=forces.device, dtype=forces.dtype)
        v_prev_local = (
            v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1)
            if (cfg.track_mode and v_prev is not None) else None
        )
        v, _mode_idx, _overlap = pick_tracked_mode(V_cand, v_prev_local, k=k_track)

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

        dt_eff = _compute_adaptive_dt(
            cfg.dt, cfg.dt_min, cfg.dt_max, eig_0,
        )

        step_disp = dt_eff * gad_vec
        max_disp = float(step_disp.norm(dim=1).max().item())
        if max_disp > cfg.max_atom_disp and max_disp > 0:
            step_disp = step_disp * (cfg.max_atom_disp / max_disp)

        new_coords = coords + step_disp
        if cfg.min_interatomic_dist > 0 and _min_interatomic_distance(new_coords) < cfg.min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    final_morse = int((evals_vib < 0).sum().item()) if evals_vib.numel() > 0 else -1
    return {
        "converged": False,
        "converged_step": None,
        "final_coords": coords.detach().cpu(),
        "final_energy": _to_float(out["energy"]),
        "final_force_norm": rms_force(forces),
        "final_morse_index": final_morse,
        "total_steps": cfg.n_steps,
        "trajectory": trajectory,
    }

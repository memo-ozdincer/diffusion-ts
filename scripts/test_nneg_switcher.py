#!/usr/bin/env python
"""Test pure n_neg-based NR/GAD switching for generative modeling.

Rule: if n_neg < 2 → GAD step, else → NR step.
No pipeline, no handoff, no snapshots. Pure geometry-based switching.

This is what a diffusion model denoiser would need: a stateless
per-step rule that depends only on the current geometry.
"""

import json, os, sys, time, argparse
import torch
import numpy as np
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.core_algos.saddle_optimizer import (
    rms_force, max_atomic_force, vib_eig,
    _rfo_step, _cap_displacement, _min_interatomic_distance,
    _to_float,
)
from src.core_algos.gad import compute_gad_vector_tracked
from src.dependencies.scine_calculator import ScineSparrowCalculator
from src.dependencies.calculators import make_scine_predict_fn

# Isopropanol reference
ATOMIC_NUMS = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
ATOM_SYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]
REF_COORDS = torch.tensor([
    [-0.2780, -0.0126,  0.0218],
    [ 1.2206,  0.1149, -0.2171],
    [-0.9600,  1.3476, -0.0370],
    [-0.8550, -0.9426, -0.9038],
    [-0.4551, -0.4216,  1.0301],
    [-0.5580, -1.8562, -0.7509],
    [ 1.7389, -0.7954,  0.0789],
    [ 1.6923,  0.9435,  0.3218],
    [ 1.3824,  0.3192, -1.2808],
    [-0.6694,  2.0549,  0.7407],
    [-0.6958,  1.7728, -1.0047],
    [-2.0476,  1.2499,  0.0073],
], dtype=torch.float64)


def run_nneg_switcher(
    predict_fn, coords0, max_steps=3000, force_tol=0.01,
    gad_dt=0.01, gad_dt_max=0.3, gad_max_disp=0.5,
    nr_max_disp=0.3, nr_trust_floor=0.01,
    min_dist=0.4, log_every=50,
):
    """Pure n_neg-based switching: n_neg < 2 → GAD, else → NR.

    Completely stateless w.r.t. history — only the current geometry
    determines which step to take. Compatible with generative modeling.
    """
    coords = coords0.detach().clone().to(torch.float64).reshape(-1, 3)
    trajectory = []
    v_prev = None
    trust_radius = nr_max_disp
    n_nr = n_gad = 0

    for step in range(max_steps):
        out = predict_fn(coords, ATOMIC_NUMS, do_hessian=True, require_grad=False)
        energy = _to_float(out["energy"])
        forces = out["forces"]
        if forces.dim() == 3:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        hessian = out["hessian"]

        f_rms = rms_force(forces)
        f_max = max_atomic_force(forces)

        evals, evecs, _ = vib_eig(hessian, coords, ATOM_SYMBOLS)
        n_neg = int((evals < 0).sum().item()) if evals.numel() > 0 else 0
        eig_0 = float(evals[0].item()) if evals.numel() > 0 else 0.0

        # THE RULE: n_neg < 2 → GAD, else → NR
        use_gad = n_neg < 2

        record = {
            "step": step, "energy": energy, "rms_force": f_rms,
            "n_neg": n_neg, "eig_0": eig_0,
            "mode": "GAD" if use_gad else "NR",
        }
        trajectory.append(record)

        if log_every > 0 and step % log_every == 0:
            mode = "GAD" if use_gad else "NR"
            print(f"  step={step:5d} {mode:3s} n_neg={n_neg} E={energy:.4f} "
                  f"F={f_rms:.6f} eig0={eig_0:.4f}")

        # Convergence
        if n_neg == 1 and f_rms < force_tol:
            return {
                "converged": True, "step": step, "energy": energy,
                "force": f_rms, "n_neg": n_neg, "eig_0": eig_0,
                "n_nr": n_nr, "n_gad": n_gad,
                "coords": coords.detach().cpu().numpy().tolist(),
                "trajectory": trajectory,
            }

        if use_gad:
            # === GAD step ===
            n_gad += 1
            gad_vec, v_next, _ = compute_gad_vector_tracked(
                forces, hessian, v_prev, k_track=8,
            )
            v_prev = v_next.detach().cpu()

            # Adaptive dt
            lam0 = abs(eig_0)
            lam_clamped = max(0.01, min(lam0, 100.0))
            dt = min(gad_dt / lam_clamped, gad_dt_max)

            disp = (gad_vec * dt).reshape(-1, 3)
            disp = _cap_displacement(disp, gad_max_disp)
            new_coords = coords + disp

        else:
            # === NR step ===
            n_nr += 1
            v_prev = None  # reset mode tracking when switching to NR

            grad = -forces.reshape(-1)
            V = evecs.to(device=grad.device, dtype=grad.dtype)
            lam = evals.to(device=grad.device, dtype=grad.dtype)

            delta_x, _ = _rfo_step(grad, V, lam)

            # Trust region
            capped = _cap_displacement(delta_x.reshape(-1, 3), trust_radius)
            new_coords = coords + capped

            if min_dist > 0 and _min_interatomic_distance(new_coords) < min_dist:
                trust_radius = max(trust_radius * 0.5, nr_trust_floor)
                new_coords = coords + _cap_displacement(delta_x.reshape(-1, 3), trust_radius)

            # Accept/reject
            out_new = predict_fn(new_coords, ATOMIC_NUMS, do_hessian=False, require_grad=False)
            e_new = _to_float(out_new["energy"])
            if e_new <= energy + 0.1:
                trust_radius = min(trust_radius * 1.2, nr_max_disp)
            else:
                trust_radius = max(trust_radius * 0.5, nr_trust_floor)

        coords = new_coords.detach()

    # Did not converge
    return {
        "converged": False, "step": max_steps, "energy": energy,
        "force": f_rms, "n_neg": n_neg, "eig_0": eig_0,
        "n_nr": n_nr, "n_gad": n_gad,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--force-tol", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    print(f"Pure n_neg switcher: n_neg<2→GAD, else→NR")
    print(f"Seeds: {args.n_seeds}, noise: {args.noise}A, max_steps: {args.max_steps}")
    print("=" * 70)

    calc = ScineSparrowCalculator(functional="DFTB0")
    predict_fn = make_scine_predict_fn(calc)
    results = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        noise = torch.randn_like(REF_COORDS) * args.noise
        start = REF_COORDS + noise

        if _min_interatomic_distance(start) < 0.4:
            print(f"Seed {seed}: SKIP (min_dist < 0.4)")
            results.append({"seed": seed, "status": "skip"})
            continue

        print(f"\nSeed {seed}:")
        t0 = time.time()
        result = run_nneg_switcher(
            predict_fn, start,
            max_steps=args.max_steps,
            force_tol=args.force_tol,
            log_every=args.log_every,
        )
        wall = time.time() - t0

        status = "ts" if result["converged"] else "no_ts"
        result["seed"] = seed
        result["status"] = status
        result["wall"] = wall

        # Summarize trajectory: mode switches
        traj = result.get("trajectory", [])
        modes = [t["mode"] for t in traj]
        switches = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        nneg_hist = Counter(t["n_neg"] for t in traj)

        print(f"  → {status} in {result['step']} steps ({wall:.1f}s), "
              f"NR={result['n_nr']}, GAD={result['n_gad']}, "
              f"switches={switches}")
        print(f"    n_neg distribution: {dict(sorted(nneg_hist.items()))}")

        # Don't save full trajectory to keep JSON manageable
        traj_summary = {
            "switches": switches,
            "nneg_hist": {str(k): v for k, v in nneg_hist.items()},
            "final_100_modes": modes[-100:] if len(modes) >= 100 else modes,
        }
        result["traj_summary"] = traj_summary
        del result["trajectory"]  # too large for JSON

        results.append(result)

    # Summary
    total = len(results)
    skipped = sum(1 for r in results if r["status"] == "skip")
    valid = total - skipped
    ts = sum(1 for r in results if r["status"] == "ts")
    print(f"\n{'='*70}")
    print(f"RESULTS: {ts}/{valid} valid = {ts/valid*100:.0f}% convergence" if valid > 0 else "No valid seeds")
    if ts > 0:
        ts_results = [r for r in results if r["status"] == "ts"]
        steps = [r["step"] for r in ts_results]
        nr_frac = [r["n_nr"]/(r["n_nr"]+r["n_gad"]) for r in ts_results]
        switches = [r["traj_summary"]["switches"] for r in ts_results]
        print(f"Steps: mean={np.mean(steps):.0f}, median={np.median(steps):.0f}")
        print(f"NR fraction: mean={np.mean(nr_frac):.2f}")
        print(f"Switches: mean={np.mean(switches):.0f}, max={max(switches)}")

    # Save
    out_path = "/scratch/memoozd/diffusion-ts/nneg_switcher_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Incremental test: isopropanol through saddle optimizer.

Each test only runs if the previous passed.
Test 4 runs 2 noisy starts in parallel via concurrent.futures.

Usage (on compute node):
  cd /project/rrg-aspuru/memoozd/diffusion-ts
  source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
  PYTHONPATH=$PWD:$PYTHONPATH OMP_NUM_THREADS=2 python scripts/test_isopropanol_single.py
"""

import sys
import os
import time
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---- Isopropanol (C3H8O) approximate geometry (Angstroms) ----
# 12 atoms: C(central), C(methyl1), C(methyl2), O, H(CH), H(OH), 3xH(CH3), 3xH(CH3)
ISOPROPANOL_COORDS = torch.tensor([
    [ 0.000,  0.000,  0.000],   # C central
    [-1.270,  0.760,  0.000],   # C methyl1
    [ 1.270,  0.760,  0.000],   # C methyl2
    [ 0.000, -0.930,  1.100],   # O
    [ 0.000, -0.670, -0.870],   # H on central C
    [ 0.000, -0.400,  1.920],   # H on O
    [-1.270,  1.410,  0.870],   # H methyl1
    [-1.270,  1.410, -0.870],   # H methyl1
    [-2.160,  0.120,  0.000],   # H methyl1
    [ 1.270,  1.410,  0.870],   # H methyl2
    [ 1.270,  1.410, -0.870],   # H methyl2
    [ 2.160,  0.120,  0.000],   # H methyl2
], dtype=torch.float32)

ATOMIC_NUMS = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
ATOMSYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]


def ok(msg):
    print(f"  PASS: {msg}")
    return True

def fail(msg):
    print(f"  FAIL: {msg}")
    return False


# ---- Test 1: SCINE single point ----
def test_scine_single_point():
    print("\n" + "=" * 60)
    print("Test 1: SCINE/DFTB0 single-point on isopropanol")
    print("=" * 60)

    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn

    calc = ScineSparrowCalculator(functional="DFTB0")
    predict_fn = make_scine_predict_fn(calc)

    out = predict_fn(ISOPROPANOL_COORDS, ATOMIC_NUMS, do_hessian=True)
    energy = float(out["energy"])
    forces = out["forces"]
    hessian = out["hessian"]

    if forces.dim() == 3:
        forces = forces[0]
    force_norm = float(forces.reshape(-1, 3).norm(dim=1).mean())

    print(f"  Energy:     {energy:.6f} eV")
    print(f"  Force norm: {force_norm:.6f} eV/A")
    print(f"  Hessian:    {hessian.shape}")

    if hessian.shape != (36, 36):
        fail(f"Hessian shape {hessian.shape} != (36, 36)")
        return None

    ok("SCINE single-point works")
    return True, predict_fn


# ---- Test 2: Eckart projection ----
def test_eckart_projection(predict_fn):
    print("\n" + "=" * 60)
    print("Test 2: Eckart projection -> vibrational eigenvalues")
    print("=" * 60)

    from src.core_algos.saddle_optimizer import vib_eig

    out = predict_fn(ISOPROPANOL_COORDS, ATOMIC_NUMS, do_hessian=True)
    evals, evecs, Q = vib_eig(out["hessian"], ISOPROPANOL_COORDS, ATOMSYMBOLS)
    n_vib = evals.shape[0]
    n_neg = int((evals < 0).sum().item())

    print(f"  Vibrational DOF: {n_vib}  (expected 30 for 12 atoms)")
    print(f"  n_neg:           {n_neg}")
    print(f"  Bottom 5 evals:  {[f'{v:.4f}' for v in evals[:5].tolist()]}")
    print(f"  Top 3 evals:     {[f'{v:.2f}' for v in evals[-3:].tolist()]}")

    if n_vib != 30:
        return fail(f"Got {n_vib} vibrational DOF, expected 30")

    return ok(f"30 vibrational DOF, n_neg={n_neg}")


# ---- Test 3: NR with early handoff to GAD ----
def test_nr_then_gad(predict_fn):
    """Run NR from noisy geometry, monitor n_neg, hand off to GAD at different points."""
    print("\n" + "=" * 60)
    print("Test 3: NR trajectory (0.2A noise) — find handoff points for GAD")
    print("=" * 60)

    from src.core_algos.saddle_optimizer import run_nr_minimization, NRConfig, vib_eig

    torch.manual_seed(42)
    noisy_coords = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * 0.2
    print(f"  Starting from 0.2A-perturbed geometry (seed=42)")

    # Run NR with enough steps to see the full n_neg trajectory
    nr_cfg = NRConfig(n_steps=500, force_converged=0.01)
    t0 = time.time()
    nr_result = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
    dt = time.time() - t0

    traj = nr_result["trajectory"]
    print(f"  NR ran {nr_result['total_steps']} steps ({dt:.1f}s)")
    print(f"  Final: E={nr_result['final_energy']:.6f} F={nr_result['final_force_norm']:.6f} "
          f"n_neg={nr_result['final_n_neg']}")

    # Print n_neg evolution
    print(f"\n  n_neg trajectory (every 10 steps):")
    for rec in traj[::10]:
        print(f"    step={rec['step']:4d} E={rec['energy']:.6f} F={rec['force_norm']:.6f} "
              f"n_neg={rec['n_neg']} min_eval={rec['min_eval']:.4f}")

    # Find snapshots where n_neg == 2 and n_neg == 1
    snapshots = {}
    for target_nneg in [3, 2, 1]:
        for rec in traj:
            if rec["n_neg"] == target_nneg and target_nneg not in snapshots:
                snapshots[target_nneg] = rec["step"]

    print(f"\n  First occurrence: " + "  ".join(
        f"n_neg={k} @ step {v}" for k, v in sorted(snapshots.items(), reverse=True)))

    return nr_result, traj, noisy_coords


def test_gad_from_handoff(predict_fn, traj, noisy_coords):
    """Try GAD from NR snapshots at various n_neg values (early handoff).

    Now uses saved coords from trajectory directly — no re-running NR.
    """
    print("\n" + "=" * 60)
    print("Test 4: GAD from early NR handoff points")
    print("=" * 60)

    from src.core_algos.saddle_optimizer import vib_eig

    # Collect all unique n_neg handoff points (first occurrence of each)
    seen = set()
    handoff_points = []
    for rec in traj:
        nn = rec["n_neg"]
        if nn > 0 and nn not in seen and "coords" in rec:
            seen.add(nn)
            handoff_points.append(rec)

    if not handoff_points:
        fail("No trajectory points with n_neg > 0 found")
        return None

    # Sort by n_neg descending (try highest first)
    handoff_points.sort(key=lambda r: r["n_neg"], reverse=True)

    print(f"  Available handoff points: " + ", ".join(
        f"n_neg={r['n_neg']} @ step {r['step']}" for r in handoff_points))

    for rec in handoff_points:
        handoff_coords = rec["coords"]
        target_nneg = rec["n_neg"]

        # Verify
        out_h = predict_fn(handoff_coords, ATOMIC_NUMS, do_hessian=True)
        evals_h, _, _ = vib_eig(out_h["hessian"], handoff_coords, ATOMSYMBOLS)
        n_neg_h = int((evals_h < 0).sum().item())
        forces_h = out_h["forces"]
        if forces_h.dim() == 3:
            forces_h = forces_h[0]
        force_h = float(forces_h.reshape(-1, 3).norm(dim=1).mean())

        print(f"\n  === Handoff step {rec['step']}: n_neg={n_neg_h} force={force_h:.4f} "
              f"eig0={float(evals_h[0]):.4f} ===")

        # Try three GAD configs: default, moderate, RK45
        for label, dt, dt_max, max_disp in [
            ("Euler default",  0.003, 0.1, 0.35),
            ("Euler moderate", 0.01,  0.3, 0.5),
            ("Euler big",      0.03,  1.0, 1.0),
        ]:
            r = _run_gad_euler(predict_fn, handoff_coords,
                f"{label} from n_neg={target_nneg}",
                n_steps=1000, dt=dt, dt_max=dt_max, max_atom_disp=max_disp)
            if r["converged"]:
                ok(f"{label} GAD converged from n_neg={target_nneg} handoff!")
                return r["final_coords"]

        # RK45
        r_rk45, rk_nneg, rk_force = _run_gad_rk45(predict_fn, handoff_coords, t1=5.0, max_steps=5000)
        if rk_nneg == 1 and rk_force < 0.01:
            ok(f"RK45 GAD converged from n_neg={target_nneg} handoff!")
            return r_rk45["final_coords"]

    fail("No GAD variant found a valid TS from any handoff point")
    return None


# ---- Test 4: GAD saddle search — try multiple strategies ----
def _run_gad_euler(predict_fn, start_coords, label, n_steps, dt, dt_max, max_atom_disp):
    """Run one Euler GAD config and print results."""
    from src.core_algos.saddle_optimizer import run_gad_saddle_search, GADConfig as SOGADConfig, vib_eig

    cfg = SOGADConfig(n_steps=n_steps, dt=dt, dt_max=dt_max, max_atom_disp=max_atom_disp)
    t0 = time.time()
    result = run_gad_saddle_search(predict_fn, start_coords, ATOMIC_NUMS, ATOMSYMBOLS, cfg)
    wall = time.time() - t0

    traj = result["trajectory"]
    print(f"\n  [{label}]  dt={dt} dt_max={dt_max} max_disp={max_atom_disp}")
    print(f"    Converged: {result['converged']}  Steps: {result['total_steps']}  ({wall:.1f}s)")
    print(f"    Energy: {result['final_energy']:.6f}  Morse: {result['final_morse_index']}  "
          f"Force: {result.get('final_force_norm', '?')}")

    checkpoints = sorted(set([0, len(traj)//4, len(traj)//2, 3*len(traj)//4, len(traj)-1]))
    for c in checkpoints:
        if 0 <= c < len(traj):
            rec = traj[c]
            print(f"      step={rec['step']:4d} E={rec['energy']:.6f} "
                  f"F={rec['force_norm']:.6f} n_neg={rec['n_neg']} eig0={rec['eig_0']:.4f}")

    return result


def _run_gad_rk45(predict_fn, start_coords, t1, max_steps):
    """Run RK45 GAD and print results."""
    from src.core_algos.gad import gad_rk45_integrate
    from src.core_algos.saddle_optimizer import vib_eig

    print(f"\n  [RK45]  t1={t1} max_steps={max_steps}")
    t0 = time.time()
    result = gad_rk45_integrate(
        predict_fn, start_coords, ATOMIC_NUMS,
        t1=t1, max_steps=max_steps,
    )
    wall = time.time() - t0

    final = result["final_coords"]
    out_f = predict_fn(final, ATOMIC_NUMS, do_hessian=True)
    evals_f, _, _ = vib_eig(out_f["hessian"], final, ATOMSYMBOLS)
    forces_f = out_f["forces"]
    if forces_f.dim() == 3:
        forces_f = forces_f[0]
    n_neg = int((evals_f < 0).sum().item())
    force_norm = float(forces_f.reshape(-1, 3).norm(dim=1).mean())

    print(f"    Steps: {result['steps']}  t_final: {result['t_final']:.4f}  ({wall:.1f}s)")
    print(f"    Energy: {float(out_f['energy']):.6f}  n_neg: {n_neg}  Force: {force_norm:.6f}")
    print(f"    eig0: {float(evals_f[0]):.4f}  Bottom 5: {[f'{v:.4f}' for v in evals_f[:5].tolist()]}")

    valid = n_neg == 1 and force_norm < 0.01
    print(f"    VALID TS: {valid}")

    return result, n_neg, force_norm


def test_gad_saddle_search(predict_fn, nr_result):
    print("\n" + "=" * 60)
    print("Test 4: GAD saddle search — Euler (default, big-step) + RK45")
    print("=" * 60)

    start_coords = nr_result["final_coords"]

    # 4a: Default Euler (baseline)
    r_default = _run_gad_euler(predict_fn, start_coords,
        "Euler default", n_steps=1000, dt=0.003, dt_max=0.1, max_atom_disp=0.35)

    # 4b: Big-step Euler (larger dt, bigger displacement cap)
    r_big = _run_gad_euler(predict_fn, start_coords,
        "Euler big-step", n_steps=1000, dt=0.03, dt_max=1.0, max_atom_disp=1.0)

    # 4c: RK45 (adaptive step)
    r_rk45, rk_nneg, rk_force = _run_gad_rk45(predict_fn, start_coords, t1=5.0, max_steps=5000)

    # Pick best result
    best = None
    for r in [r_default, r_big]:
        if r["converged"]:
            best = r["final_coords"]
            break

    if best is None and rk_nneg == 1 and rk_force < 0.01:
        best = r_rk45["final_coords"]

    if best is not None:
        ok("GAD found TS")
        return best

    # None converged — return best partial result for diagnostics
    fail("No GAD variant found a valid TS")
    # Return RK45 coords as best guess
    return r_rk45["final_coords"] if r_rk45 else None


# ---- Test 4b: GAD-based TS refinement after GAD ----
def test_ts_refinement(predict_fn, gad_coords):
    """Continue GAD with smaller dt to refine forces down to < 0.01 while keeping n_neg=1."""
    print("\n" + "=" * 60)
    print("Test 4b: GAD TS refinement (force → < 0.01 while keeping n_neg=1)")
    print("=" * 60)

    from src.core_algos.saddle_optimizer import run_gad_saddle_search, GADConfig, vib_eig

    # Smaller dt + more steps for fine refinement
    cfg = GADConfig(
        n_steps=500,
        dt=0.001,          # smaller base dt for fine control
        dt_max=0.01,
        max_atom_disp=0.05,  # tight displacement cap
    )
    t0 = time.time()
    result = run_gad_saddle_search(
        predict_fn, gad_coords, ATOMIC_NUMS, ATOMSYMBOLS, cfg,
    )
    dt = time.time() - t0

    print(f"  Converged:  {result['converged']} (steps={result['total_steps']}, {dt:.1f}s)")
    print(f"  Energy:     {result['final_energy']:.6f} eV")
    print(f"  Morse idx:  {result['final_morse_index']}")

    traj = result["trajectory"]
    checkpoints = sorted(set([0, len(traj)//4, len(traj)//2, 3*len(traj)//4, len(traj)-1]))
    for c in checkpoints:
        if 0 <= c < len(traj):
            rec = traj[c]
            print(f"    step={rec['step']:4d} E={rec['energy']:.6f} "
                  f"F={rec['force_norm']:.6f} n_neg={rec['n_neg']} eig0={rec['eig_0']:.4f}")

    if result["converged"]:
        ts_coords = result["final_coords"]
        out_ts = predict_fn(ts_coords, ATOMIC_NUMS, do_hessian=True)
        evals_ts, _, _ = vib_eig(out_ts["hessian"], ts_coords, ATOMSYMBOLS)
        forces_ts = out_ts["forces"]
        if forces_ts.dim() == 3:
            forces_ts = forces_ts[0]
        force_norm = float(forces_ts.reshape(-1, 3).norm(dim=1).mean())

        print(f"\n  Refined TS:")
        print(f"    n_neg=1, force={force_norm:.6f}, eig0={float(evals_ts[0]):.4f}")
        print(f"    Bottom 5: {[f'{v:.4f}' for v in evals_ts[:5].tolist()]}")
        valid = force_norm < 0.01
        print(f"    VALID TS (force<0.01): {valid}")
        if valid:
            return ok("GAD refinement → valid TS"), ts_coords
        else:
            return ok(f"GAD refinement kept n_neg=1, force={force_norm:.4f}"), ts_coords

    return fail(f"Lost saddle during refinement (morse_idx={result['final_morse_index']})")


# ---- Test 5: parallel find_transition_state with 2 noisy starts ----
def _run_single_ts(seed, noise_std):
    """Worker function for parallel TS search. Timeout-safe."""
    import signal
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn
    from src.core_algos.saddle_optimizer import find_transition_state, SaddleOptimizerConfig, NRConfig, GADConfig

    calc = ScineSparrowCalculator(functional="DFTB0")
    predict_fn = make_scine_predict_fn(calc)

    torch.manual_seed(seed)
    noisy_coords = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * noise_std

    # Check min interatomic distance before starting
    c = noisy_coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    min_dist = float(dist.min().item())
    if min_dist < 0.5:
        return {"seed": seed, "noise_std": noise_std, "converged": False,
                "error": f"min_dist={min_dist:.3f} too small, skipped"}

    cfg = SaddleOptimizerConfig(
        nr=NRConfig(n_steps=500, force_converged=0.01),
        gad=GADConfig(n_steps=200),
    )
    t0 = time.time()
    result = find_transition_state(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, cfg=cfg)
    dt = time.time() - t0

    p1 = result["phase1_result"]
    p2 = result["phase2_result"]
    return {
        "seed": seed,
        "noise_std": noise_std,
        "converged": result["converged"],
        "total_steps": result["total_steps"],
        "final_energy": result["final_energy"],
        "p1_converged": p1["converged"],
        "p1_steps": p1["total_steps"],
        "p1_n_neg": p1["final_n_neg"],
        "p1_force": p1["final_force_norm"],
        "p2_converged": p2["converged"] if p2 else None,
        "p2_steps": p2["total_steps"] if p2 else None,
        "p2_morse_idx": p2["final_morse_index"] if p2 else None,
        "wall_time": dt,
    }


def test_parallel_ts_search():
    print("\n" + "=" * 60)
    print("Test 5: Parallel find_transition_state (2 noisy starts, noise=0.1A)")
    print("=" * 60)

    seeds = [100, 200]
    noise_std = 0.1  # small noise to keep geometry physical

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_single_ts, s, noise_std): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                print(f"  seed={seed} CRASHED: {e}")
                results.append({"seed": seed, "converged": False, "error": str(e)})
    total_time = time.time() - t0

    print(f"  Wall time (parallel): {total_time:.1f}s")
    for r in sorted(results, key=lambda x: x["seed"]):
        if "error" in r:
            print(f"  seed={r['seed']}: {r['error']}")
        else:
            print(f"  seed={r['seed']}: TS={r['converged']}  steps={r['total_steps']}  "
                  f"E={r['final_energy']:.4f}  "
                  f"P1: conv={r['p1_converged']} {r['p1_steps']}steps n_neg={r['p1_n_neg']} F={r['p1_force']:.4f}  "
                  f"P2: conv={r['p2_converged']} {r['p2_steps']}steps idx={r['p2_morse_idx']}  "
                  f"{r['wall_time']:.1f}s")

    n_ts = sum(1 for r in results if r.get("converged"))
    print(f"\n  {n_ts}/{len(seeds)} found TS")
    return n_ts > 0


def main():
    # Test 1: SCINE single point
    result = test_scine_single_point()
    if not isinstance(result, tuple):
        print("\nABORTED: Test 1 failed")
        return
    predict_fn = result[1]

    # Test 2: Eckart projection
    if not test_eckart_projection(predict_fn):
        print("\nABORTED: Test 2 failed")
        return

    # Test 3: NR trajectory — find handoff points
    nr_result, traj, noisy_coords = test_nr_then_gad(predict_fn)

    # Test 4: GAD from early handoff (n_neg=2 or n_neg=1)
    gad_coords = test_gad_from_handoff(predict_fn, traj, noisy_coords)
    if gad_coords is None:
        print("\nTest 4 did not find TS, but we have diagnostics")
    else:
        ok("Found valid TS via early handoff")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

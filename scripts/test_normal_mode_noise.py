#!/usr/bin/env python
"""Test normal-mode displacement vs uniform noise for starting geometries.

Normal mode displacement:
  1. Minimize isopropanol to equilibrium
  2. Compute vibrational modes at equilibrium
  3. Displace along random linear combination of normal modes

This preserves bond topology better than uniform Gaussian noise.
"""
import sys, os, time, json, torch
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

ISOPROPANOL_COORDS = torch.tensor([
    [ 0.000,  0.000,  0.000], [-1.270,  0.760,  0.000], [ 1.270,  0.760,  0.000],
    [ 0.000, -0.930,  1.100], [ 0.000, -0.670, -0.870], [ 0.000, -0.400,  1.920],
    [-1.270,  1.410,  0.870], [-1.270,  1.410, -0.870], [-2.160,  0.120,  0.000],
    [ 1.270,  1.410,  0.870], [ 1.270,  1.410, -0.870], [ 2.160,  0.120,  0.000],
], dtype=torch.float32)
ATOMIC_NUMS = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
ATOMSYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]


def get_equilibrium_and_modes(functional="DFTB0"):
    """Minimize isopropanol and compute normal modes at equilibrium."""
    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn
    from src.core_algos.saddle_optimizer import run_nr_minimization, NRConfig, vib_eig

    calc = ScineSparrowCalculator(functional=functional)
    predict_fn = make_scine_predict_fn(calc)

    nr_cfg = NRConfig(n_steps=2000, force_converged=0.01)
    nr_result = run_nr_minimization(predict_fn, ISOPROPANOL_COORDS, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)

    eq_coords = nr_result["final_coords"]
    out = predict_fn(eq_coords, ATOMIC_NUMS, do_hessian=True)
    evals, evecs, Q = vib_eig(out["hessian"], eq_coords, ATOMSYMBOLS)

    print(f"Equilibrium: E={nr_result['final_energy']:.6f} F={nr_result['final_force_norm']:.6f}")
    print(f"Vibrational modes: {evals.shape[0]} DOF")
    print(f"Lowest 5 evals: {[f'{v:.4f}' for v in evals[:5].tolist()]}")

    return eq_coords, evals, evecs


def generate_normal_mode_displaced(eq_coords, evals, evecs, seed, amplitude=0.5):
    """Displace equilibrium geometry along random normal mode combination.

    Args:
        amplitude: RMS displacement in Angstroms (controls how far from equilibrium)
    """
    torch.manual_seed(seed)
    n_modes = evals.shape[0]

    # Random coefficients for each mode, weighted by 1/sqrt(eigenvalue)
    # Low-frequency modes get larger displacements (physically motivated)
    weights = 1.0 / torch.sqrt(torch.clamp(evals.abs(), min=0.01))
    coeffs = torch.randn(n_modes) * weights

    # Build displacement in Cartesian space
    displacement = (evecs * coeffs.unsqueeze(0)).sum(dim=1)  # (3N,)

    # Scale to desired RMS amplitude
    rms = float(displacement.reshape(-1, 3).norm(dim=1).mean())
    if rms > 1e-10:
        displacement = displacement * (amplitude / rms)

    noisy_coords = eq_coords.reshape(-1) + displacement
    return noisy_coords.reshape(-1, 3)


def worker(seed, eq_coords, evals, evecs, amplitude, functional):
    """Run TS search from normal-mode-displaced geometry."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn
    from src.core_algos.saddle_optimizer import (
        run_nr_minimization, NRConfig,
        run_gad_saddle_search, GADConfig as SOGADConfig,
        vib_eig,
    )

    calc = ScineSparrowCalculator(functional=functional)
    predict_fn = make_scine_predict_fn(calc)

    start_coords = generate_normal_mode_displaced(eq_coords, evals, evecs, seed, amplitude)

    # Check min interatomic distance
    c = start_coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    min_dist = float(dist.min().item())
    if min_dist < 0.5:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_dist:.3f}"}

    t0 = time.time()

    # NR
    nr_cfg = NRConfig(n_steps=500, force_converged=0.01)
    nr_result = run_nr_minimization(predict_fn, start_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)

    # Handoff candidates
    nneg1 = [r for r in nr_result["trajectory"] if r["n_neg"] == 1 and "coords" in r]
    nneg2 = [r for r in nr_result["trajectory"] if r["n_neg"] == 2 and "coords" in r]

    candidates = []
    if nneg1:
        candidates.append(("last_nneg1", nneg1[-1]))
        if len(nneg1) > 1:
            candidates.append(("first_nneg1", nneg1[0]))
    if nneg2:
        candidates.append(("last_nneg2", nneg2[-1]))

    if not candidates:
        return {"seed": seed, "status": "no_handoff", "nr_steps": nr_result["total_steps"],
                "wall": time.time() - t0}

    gad_cfg = SOGADConfig(n_steps=3000, dt=0.01, dt_max=0.3, max_atom_disp=0.5)

    best_result = None
    best_label = best_step = best_nneg = None
    for label, rec in candidates:
        gad_result = run_gad_saddle_search(predict_fn, rec["coords"], ATOMIC_NUMS, ATOMSYMBOLS, gad_cfg)
        if gad_result["converged"]:
            best_result = gad_result
            best_label, best_step, best_nneg = label, rec["step"], rec["n_neg"]
            break
        if best_result is None or (
            gad_result["final_morse_index"] == 1 and
            gad_result.get("final_force_norm", 999) < best_result.get("final_force_norm", 999)
        ):
            best_result = gad_result
            best_label, best_step, best_nneg = label, rec["step"], rec["n_neg"]

    wall = time.time() - t0
    result = {
        "seed": seed, "status": "ts" if best_result["converged"] else "no_ts",
        "nr_steps": nr_result["total_steps"], "handoff_step": best_step,
        "handoff_nneg": best_nneg, "handoff_label": best_label,
        "gad_steps": best_result["total_steps"], "final_energy": best_result["final_energy"],
        "final_morse_idx": best_result["final_morse_index"],
        "final_force": best_result.get("final_force_norm", None), "wall": wall,
    }

    if best_result["converged"]:
        ts_coords = best_result["final_coords"]
        out_ts = predict_fn(ts_coords, ATOMIC_NUMS, do_hessian=True)
        evals_ts, _, _ = vib_eig(out_ts["hessian"], ts_coords, ATOMSYMBOLS)
        result["ts_eig0"] = float(evals_ts[0])
        result["ts_coords"] = ts_coords.tolist()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--functional", type=str, default="DFTB0")
    args = parser.parse_args()

    print(f"Normal-mode displacement test: {args.n_seeds} seeds, amp={args.amplitude}A, {args.functional}")

    # Get equilibrium and modes (single-threaded)
    eq_coords, evals, evecs = get_equilibrium_and_modes(args.functional)

    seeds = list(range(args.n_seeds))
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(worker, s, eq_coords, evals, evecs, args.amplitude, args.functional): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                if r["status"] == "ts":
                    print(f"  seed={r['seed']:3d} TS! E={r['final_energy']:.4f} "
                          f"eig0={r.get('ts_eig0','?'):.4f} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} {r['wall']:.1f}s")
                elif r["status"] == "no_ts":
                    print(f"  seed={r['seed']:3d} no TS  morse={r['final_morse_idx']} "
                          f"F={r.get('final_force','?'):.4f} {r['wall']:.1f}s")
                else:
                    print(f"  seed={r['seed']:3d} {r['status']}: {r.get('reason','')}")
            except Exception as e:
                print(f"  seed={seed} CRASH: {e}")
                results.append({"seed": seed, "status": "crash", "reason": str(e)})

    total = time.time() - t0
    n_ts = sum(1 for r in results if r["status"] == "ts")
    n_valid = sum(1 for r in results if r["status"] != "skip")
    print(f"\nSummary: {n_ts}/{len(seeds)} TS ({n_ts}/{n_valid} valid = {100*n_ts/max(n_valid,1):.0f}%)")
    print(f"Wall: {total:.1f}s")

    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"normal_mode_{args.functional}_{args.amplitude}A.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Parallel TS search v7: always multi-start + better candidate diversity.

Changes from v6:
  - Always run 2-3 NR starts (full, 60%, 40% noise) for >= 0.4A noise
  - Better candidate diversity: include more trajectory spread
  - Early termination per-candidate (don't burn 5000 steps on diverging trajectories)
  - Post-hoc rescue for all stuck cases (not just morse=2)
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


def _min_dist(coords):
    c = coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    return float(dist.min().item())


def collect_candidates(nr_result, predict_fn, vib_eig, pfx=""):
    traj = nr_result["trajectory"]
    nneg0 = [r for r in traj if r["n_neg"] == 0 and "coords" in r]
    nneg1 = [r for r in traj if r["n_neg"] == 1 and "coords" in r]
    nneg2 = [r for r in traj if r["n_neg"] == 2 and "coords" in r]
    nneg3 = [r for r in traj if r["n_neg"] == 3 and "coords" in r]

    cands = []
    if nneg1:
        cands.append((f"{pfx}last_nneg1", nneg1[-1]))
        if len(nneg1) > 2:
            cands.append((f"{pfx}mid_nneg1", nneg1[len(nneg1)//2]))
        if len(nneg1) > 4:
            cands.append((f"{pfx}q1_nneg1", nneg1[len(nneg1)//4]))
            cands.append((f"{pfx}q3_nneg1", nneg1[3*len(nneg1)//4]))
        if len(nneg1) > 1:
            cands.append((f"{pfx}first_nneg1", nneg1[0]))

    if nneg2:
        cands.append((f"{pfx}last_nneg2", nneg2[-1]))
        if len(nneg2) > 1:
            cands.append((f"{pfx}first_nneg2", nneg2[0]))

    if nneg3:
        cands.append((f"{pfx}last_nneg3", nneg3[-1]))

    if nr_result["converged"] and nneg0:
        mc = nr_result["final_coords"]
        out = predict_fn(mc, ATOMIC_NUMS, do_hessian=True)
        evals, evecs, _ = vib_eig(out["hessian"], mc, ATOMSYMBOLS)
        for ei in range(min(3, evecs.shape[1])):
            v = evecs[:, ei].reshape(-1, 3).to(mc.dtype)
            for s in [0.3, -0.3, 0.5, -0.5]:
                p = mc + s * v
                if _min_dist(p) >= 0.3:
                    cands.append((f"{pfx}min_ev{ei}_{s}", {"step": -1, "n_neg": 0, "coords": p}))

    return cands


def scoop_and_retry(predict_fn, stuck_coords, atomic_nums, atomsymbols,
                     GADConfig, vib_eig, run_nr_minimization, NRConfig,
                     run_gad_saddle_search, max_steps=8000):
    nr_cfg = NRConfig(n_steps=500, force_converged=0.01, min_interatomic_dist=0.3)
    nr_r = run_nr_minimization(predict_fn, stuck_coords, atomic_nums, atomsymbols, nr_cfg)
    used = nr_r["total_steps"]

    if not nr_r["converged"]:
        return None, used

    mc = nr_r["final_coords"]
    out = predict_fn(mc, atomic_nums, do_hessian=True)
    evals, evecs, _ = vib_eig(out["hessian"], mc, atomsymbols)

    rem = max_steps - used
    for ei in range(min(3, evecs.shape[1])):
        v = evecs[:, ei].reshape(-1, 3).to(mc.dtype)
        for s in [0.3, -0.3, 0.5, -0.5, 0.7]:
            if rem <= 500:
                return None, used
            p = mc + s * v
            if _min_dist(p) < 0.3:
                continue
            gad_cfg = GADConfig(n_steps=min(2000, rem), dt=0.01, dt_max=0.3,
                                max_atom_disp=0.5, min_interatomic_dist=0.3)
            r = run_gad_saddle_search(predict_fn, p, atomic_nums, atomsymbols, gad_cfg)
            used += r["total_steps"]
            rem = max_steps - used
            if r["converged"]:
                return r, used
            # Try fine if close
            if r["final_morse_index"] == 1 and r.get("final_force_norm", 999) < 0.5 and rem > 500:
                fc = GADConfig(n_steps=min(2000, rem), dt=0.003, dt_max=0.1,
                               max_atom_disp=0.2, min_interatomic_dist=0.3)
                fr = run_gad_saddle_search(predict_fn, r["final_coords"], atomic_nums, atomsymbols, fc)
                used += fr["total_steps"]
                rem = max_steps - used
                if fr["converged"]:
                    return fr, used
    return None, used


def run_gad_guarded(predict_fn, coords, atomic_nums, atomsymbols,
                     run_gad_saddle_search, GADConfig, gad_steps=5000, max_energy_rise=20.0):
    """GAD with energy and force divergence guard."""
    out0 = predict_fn(coords.reshape(-1, 3), atomic_nums, do_hessian=True)
    e_start = float(out0["energy"].item()) if hasattr(out0["energy"], 'item') else float(out0["energy"])

    chunk = 1000
    cur = coords
    total = 0

    for start in range(0, gad_steps, chunk):
        n = min(chunk, gad_steps - start)
        cfg = GADConfig(n_steps=n, dt=0.01, dt_max=0.3,
                        max_atom_disp=0.5, min_interatomic_dist=0.3)
        r = run_gad_saddle_search(predict_fn, cur, atomic_nums, atomsymbols, cfg)
        total += r["total_steps"]

        if r["converged"]:
            r["total_steps"] = total
            return r, total

        # Energy guard
        if r["final_energy"] > e_start + max_energy_rise:
            r["total_steps"] = total
            return r, total

        cur = r["final_coords"]

    r["total_steps"] = total
    return r, total


def worker(seed, noise_std=0.3):
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

    functional = os.environ.get("TS_FUNCTIONAL", "DFTB0")
    calc = ScineSparrowCalculator(functional=functional)
    predict_fn = make_scine_predict_fn(calc)

    torch.manual_seed(seed)
    noisy_coords = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * noise_std

    min_d = _min_dist(noisy_coords)
    if min_d < 0.3:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_d:.3f}"}

    t0 = time.time()
    noise_vec = noisy_coords - ISOPROPANOL_COORDS

    # Multi-start NR
    all_candidates = []
    total_nr_steps = 0
    nr_steps = 2000 if noise_std >= 0.4 else 1000
    nr_cfg = NRConfig(n_steps=nr_steps, force_converged=0.01, min_interatomic_dist=0.3)

    # Start A: full noise
    nr_a = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
    total_nr_steps += nr_a["total_steps"]
    all_candidates.extend(collect_candidates(nr_a, predict_fn, vib_eig, "A_"))

    # Start B: 60% noise (always for >= 0.4A)
    if noise_std >= 0.4:
        coords_b = ISOPROPANOL_COORDS + noise_vec * 0.6
        if _min_dist(coords_b) >= 0.3:
            nr_b = run_nr_minimization(predict_fn, coords_b, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
            total_nr_steps += nr_b["total_steps"]
            all_candidates.extend(collect_candidates(nr_b, predict_fn, vib_eig, "B_"))

    # Start C: 40% noise (for >= 0.5A)
    if noise_std >= 0.5:
        coords_c = ISOPROPANOL_COORDS + noise_vec * 0.4
        if _min_dist(coords_c) >= 0.3:
            nr_c = run_nr_minimization(predict_fn, coords_c, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
            total_nr_steps += nr_c["total_steps"]
            all_candidates.extend(collect_candidates(nr_c, predict_fn, vib_eig, "C_"))

    if not all_candidates:
        return {
            "seed": seed, "status": "no_handoff",
            "reason": "no suitable handoff point found",
            "nr_steps": total_nr_steps, "wall": time.time() - t0,
        }

    # Sort candidates by priority
    def prio(item):
        l = item[0]
        p = 0
        if "nneg1" in l:
            if "last_" in l: p = 0
            elif "mid_" in l: p = 1
            elif "q" in l: p = 2
            else: p = 3
        elif "nneg2" in l: p = 10
        elif "min_ev0" in l: p = 20
        elif "min_ev" in l: p = 21
        elif "nneg3" in l: p = 30
        else: p = 40
        # Interleave A/B/C at same priority level
        if l.startswith("B_"): p += 0.3
        elif l.startswith("C_"): p += 0.6
        return p

    all_candidates.sort(key=prio)

    # GAD candidate loop
    best_result = None
    best_label = None
    best_step = None
    best_nneg = None
    total_gad_steps = 0

    for label, rec in all_candidates:
        # Use guarded GAD to prevent wasting steps on divergent trajectories
        gad_result, steps = run_gad_guarded(
            predict_fn, rec["coords"], ATOMIC_NUMS, ATOMSYMBOLS,
            run_gad_saddle_search, SOGADConfig, gad_steps=5000)
        total_gad_steps += steps

        if gad_result["converged"]:
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]
            break

        # Fine GAD if close
        if gad_result["final_morse_index"] == 1 and gad_result.get("final_force_norm", 999) < 0.5:
            fine_cfg = SOGADConfig(n_steps=3000, dt=0.003, dt_max=0.1,
                                   max_atom_disp=0.2, min_interatomic_dist=0.3)
            fine_r = run_gad_saddle_search(
                predict_fn, gad_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS, fine_cfg)
            total_gad_steps += fine_r["total_steps"]
            if fine_r["converged"]:
                best_result = fine_r
                best_label = label + "_fine"
                best_step = rec["step"]
                best_nneg = rec["n_neg"]
                break
            gad_result = fine_r

        # Scoop-and-retry for morse=2 with low force
        if gad_result["final_morse_index"] == 2 and gad_result.get("final_force_norm", 999) < 0.1:
            scoop_r, scoop_s = scoop_and_retry(
                predict_fn, gad_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=8000)
            total_gad_steps += scoop_s
            if scoop_r and scoop_r["converged"]:
                best_result = scoop_r
                best_label = label + "_scooped"
                best_step = rec["step"]
                best_nneg = rec["n_neg"]
                break

        if best_result is None or (
            gad_result["final_morse_index"] == 1 and
            gad_result.get("final_force_norm", 999) < best_result.get("final_force_norm", 999)
        ):
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]

    # Post-hoc rescue
    if best_result and not best_result["converged"]:
        # Rescue for morse=2 with low force
        if best_result["final_morse_index"] in (2, 3) and best_result.get("final_force_norm", 999) < 0.1:
            scoop_r, scoop_s = scoop_and_retry(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=10000)
            total_gad_steps += scoop_s
            if scoop_r and scoop_r["converged"]:
                best_result = scoop_r
                best_label += "_posthoc_scoop"

        # Rescue for morse=1 with moderate force
        if (not best_result["converged"] and
            best_result["final_morse_index"] == 1 and
            best_result.get("final_force_norm", 999) < 1.0):
            fine_cfg = SOGADConfig(n_steps=5000, dt=0.002, dt_max=0.05,
                                   max_atom_disp=0.15, min_interatomic_dist=0.3)
            fine_r = run_gad_saddle_search(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS, fine_cfg)
            total_gad_steps += fine_r["total_steps"]
            if fine_r["converged"]:
                best_result = fine_r
                best_label += "_posthoc_fine"

        # Rescue for high-morse with any force: re-minimize and retry
        if (not best_result["converged"] and
            best_result["final_morse_index"] >= 3 and
            best_result.get("final_force_norm", 999) < 1.0):
            scoop_r, scoop_s = scoop_and_retry(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=10000)
            total_gad_steps += scoop_s
            if scoop_r and scoop_r["converged"]:
                best_result = scoop_r
                best_label += "_posthoc_scoop_high"

    wall = time.time() - t0
    gad_result = best_result

    result = {
        "seed": seed,
        "status": "ts" if gad_result["converged"] else "no_ts",
        "nr_steps": total_nr_steps,
        "handoff_step": best_step,
        "handoff_nneg": best_nneg,
        "handoff_label": best_label,
        "gad_steps": total_gad_steps,
        "gad_converged": gad_result["converged"],
        "final_energy": gad_result["final_energy"],
        "final_morse_idx": gad_result["final_morse_index"],
        "final_force": gad_result.get("final_force_norm", None),
        "n_candidates_tried": len(all_candidates),
        "wall": wall,
    }

    if gad_result["converged"]:
        ts_coords = gad_result["final_coords"]
        out_ts = predict_fn(ts_coords, ATOMIC_NUMS, do_hessian=True)
        evals_ts, _, _ = vib_eig(out_ts["hessian"], ts_coords, ATOMSYMBOLS)
        result["ts_eig0"] = float(evals_ts[0])
        result["ts_coords"] = ts_coords.tolist()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--functional", type=str, default="DFTB0")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    os.environ["TS_FUNCTIONAL"] = args.functional

    print(f"Parallel TS search v7: {args.n_seeds} seeds, {args.n_workers} workers, "
          f"noise={args.noise}A, functional={args.functional}")
    print(f"Seeds: {seeds}")
    print()

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(worker, s, args.noise): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                if r["status"] == "ts":
                    print(f"  seed={r['seed']:3d} TS! E={r['final_energy']:.4f} "
                          f"eig0={r.get('ts_eig0', '?'):.4f} "
                          f"NR={r['nr_steps']}->GAD={r['gad_steps']} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"tried={r.get('n_candidates_tried', '?')} "
                          f"{r['wall']:.1f}s")
                elif r["status"] == "no_ts":
                    print(f"  seed={r['seed']:3d} no TS  morse={r['final_morse_idx']} "
                          f"F={r.get('final_force', '?'):.4f} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"tried={r.get('n_candidates_tried', '?')} "
                          f"NR={r['nr_steps']}->GAD={r['gad_steps']} {r['wall']:.1f}s")
                else:
                    print(f"  seed={r['seed']:3d} {r['status']}: {r.get('reason', '')}")
            except Exception as e:
                print(f"  seed={seed} CRASH: {e}")
                import traceback; traceback.print_exc()
                results.append({"seed": seed, "status": "crash", "reason": str(e)})

    total_wall = time.time() - t0

    n_ts = sum(1 for r in results if r["status"] == "ts")
    n_no_ts = sum(1 for r in results if r["status"] == "no_ts")
    n_skip = sum(1 for r in results if r["status"] in ("skip", "no_handoff", "crash"))

    print(f"\n{'='*60}")
    print(f"Summary: {n_ts}/{len(seeds)} TS found  ({n_no_ts} no_ts, {n_skip} skip/crash)")
    print(f"Convergence rate: {n_ts}/{len(seeds)} = {100*n_ts/len(seeds):.0f}%")
    if (n_ts + n_no_ts) > 0:
        print(f"Rate excl. skips: {n_ts}/{n_ts+n_no_ts} = {100*n_ts/(n_ts+n_no_ts):.0f}%")
    print(f"Wall time: {total_wall:.1f}s  ({total_wall/len(seeds):.1f}s/sample sequential equiv)")
    print(f"{'='*60}")

    no_ts = [r for r in results if r["status"] == "no_ts"]
    if no_ts:
        print("\nFailure analysis:")
        for r in sorted(no_ts, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']:3d} morse={r['final_morse_idx']} "
                  f"F={r.get('final_force', '?'):.4f} "
                  f"{r.get('handoff_label','?')} tried={r.get('n_candidates_tried', '?')}")

    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parallel_results_v7.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

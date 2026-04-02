#!/usr/bin/env python
"""Test DFTB2/DFTB3 TS search in single process (no multiprocessing)."""
import sys, os, traceback, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dependencies.scine_calculator import ScineSparrowCalculator
from src.dependencies.calculators import make_scine_predict_fn
from src.core_algos.saddle_optimizer import (
    run_nr_minimization, NRConfig,
    run_gad_saddle_search, GADConfig, vib_eig,
)

coords = torch.tensor([
    [ 0.000,  0.000,  0.000], [-1.270,  0.760,  0.000], [ 1.270,  0.760,  0.000],
    [ 0.000, -0.930,  1.100], [ 0.000, -0.670, -0.870], [ 0.000, -0.400,  1.920],
    [-1.270,  1.410,  0.870], [-1.270,  1.410, -0.870], [-2.160,  0.120,  0.000],
    [ 1.270,  1.410,  0.870], [ 1.270,  1.410, -0.870], [ 2.160,  0.120,  0.000],
], dtype=torch.float32)
Z = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
syms = ["C","C","C","O","H","H","H","H","H","H","H","H"]

for func in ["DFTB2", "DFTB3"]:
    for seed in [0, 42, 7]:
        print(f"\n=== {func} seed={seed} ===")
        try:
            calc = ScineSparrowCalculator(functional=func)
            pf = make_scine_predict_fn(calc)

            torch.manual_seed(seed)
            noisy = coords + torch.randn_like(coords) * 0.2

            nr = run_nr_minimization(pf, noisy, Z, syms,
                NRConfig(n_steps=300, force_converged=0.01))
            conv = nr["converged"]
            steps = nr["total_steps"]
            nneg = nr["final_n_neg"]
            print(f"  NR: conv={conv} steps={steps} n_neg={nneg}")

            nneg1 = [r for r in nr["trajectory"] if r["n_neg"] == 1 and "coords" in r]
            if nneg1:
                gad = run_gad_saddle_search(pf, nneg1[-1]["coords"], Z, syms,
                    GADConfig(n_steps=1000, dt=0.01, dt_max=0.3, max_atom_disp=0.5))
                morse = gad["final_morse_index"]
                force = gad.get("final_force_norm", "?")
                print(f"  GAD: conv={gad['converged']} steps={gad['total_steps']} "
                      f"morse={morse} force={force}")
            else:
                print("  No n_neg=1 handoff found")
        except Exception:
            traceback.print_exc()

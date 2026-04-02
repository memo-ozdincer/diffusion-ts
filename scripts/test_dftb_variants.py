#!/usr/bin/env python
"""Test DFTB0/DFTB2/DFTB3 on clean isopropanol geometry."""
import sys, os, traceback, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dependencies.scine_calculator import ScineSparrowCalculator
from src.dependencies.calculators import make_scine_predict_fn

coords = torch.tensor([
    [ 0.000,  0.000,  0.000], [-1.270,  0.760,  0.000], [ 1.270,  0.760,  0.000],
    [ 0.000, -0.930,  1.100], [ 0.000, -0.670, -0.870], [ 0.000, -0.400,  1.920],
    [-1.270,  1.410,  0.870], [-1.270,  1.410, -0.870], [-2.160,  0.120,  0.000],
    [ 1.270,  1.410,  0.870], [ 1.270,  1.410, -0.870], [ 2.160,  0.120,  0.000],
], dtype=torch.float32)
Z = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])

for func in ["DFTB0", "DFTB2", "DFTB3"]:
    print(f"\n--- {func} (clean geometry) ---")
    try:
        calc = ScineSparrowCalculator(functional=func)
        pf = make_scine_predict_fn(calc)
        out = pf(coords, Z, do_hessian=True)
        e = float(out["energy"])
        f = out["forces"]
        if f.dim() == 3: f = f[0]
        fn = float(f.reshape(-1,3).norm(dim=1).mean())
        h = out["hessian"]
        print(f"  Energy: {e:.6f} eV  Force: {fn:.6f} eV/A  Hessian: {h.shape}")
    except Exception:
        traceback.print_exc()

    # Also test with 0.1A noise
    torch.manual_seed(42)
    noisy = coords + torch.randn_like(coords) * 0.1
    print(f"--- {func} (0.1A noise) ---")
    try:
        calc2 = ScineSparrowCalculator(functional=func)
        pf2 = make_scine_predict_fn(calc2)
        out2 = pf2(noisy, Z, do_hessian=True)
        e2 = float(out2["energy"])
        print(f"  Energy: {e2:.6f} eV  OK")
    except Exception:
        traceback.print_exc()

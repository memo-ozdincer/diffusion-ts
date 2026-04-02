# Step 3: Adjoint Sampling for TS Generation — Analysis

## 1. What we have

### TS dataset (87 isopropanol TS geometries)

| Property | Value |
|----------|-------|
| N samples | 87 |
| Atoms per molecule | 12 (C3H8O) |
| Coords shape | (87, 12, 3) |
| Energy range | -311.94 to -306.16 eV (5.78 eV span) |
| eig0 range | -7.10 to -0.001 |
| Convergence | n_neg=1, rms_force < 0.01 eV/A |
| Backend | SCINE Sparrow DFTB0 |

### Structural diversity

- **50 near-duplicates** (NN-RMSD < 0.05 A): clustered into tight cores
- **31 structurally unique** (NN-RMSD > 0.1 A): distinct TS geometries
- **5 reaction families**: H-migration (2 types), C-C ring closure, C-C dissociation, H→O transfer
- **27 singletons**: rare multi-bond TS with no structural neighbors

### Key bond distortions (vs equilibrium isopropanol)

| Bond | Equilibrium | TS range | Interpretation |
|------|-------------|----------|---------------|
| C0-H | 1.09 A | 1.07-3.43 A | H migration (main reaction coordinate) |
| C0-C1 | 1.48 A | 1.28-3.31 A | C-C stretch/break |
| C0-O | 1.43 A | 1.08-3.76 A | C-O distortion |
| C1-C2 | 2.54 A | 1.90-3.49 A | Ring closure or expansion |

---

## 2. How adjoint sampling works

### Core idea

Standard diffusion models learn `score(x,t) = grad_x log p_t(x)` from data samples. Adjoint sampling instead learns a **controller** `u(x,t)` that guides a stochastic trajectory toward low-energy configurations, trained using **energy gradients** rather than samples from a target distribution.

### The SDE framework

Forward diffusion (from sample to noise):
```
dx_t = u(t, x_t) dt + g(t) dW_t
```

The controller `u(t, x_t)` is a neural network (EGNN) that predicts the drift. Training minimizes:
```
L = ||u(t, x_t) - score_target||^2

score_target = -(grad_E(x_1) - x_1/sigma^2)
             = negative (energy gradient + prior score at t=1)
```

### Key difference from standard diffusion

| | Standard diffusion | Adjoint sampling |
|---|---|---|
| **Training data** | Samples from target p(x) | Energy function E(x) + its gradient |
| **What's learned** | Score function grad log p(x) | Controller u(t,x) guiding SDE |
| **Can sample without data** | No (need reference samples) | Yes (only need energy gradients) |
| **Amortized** | Per-distribution | Across molecular types (SMILES-conditioned) |

### Architecture (Cartesian mode)

- **EGNN** (Equivariant Graph Neural Network): 12 layers, 128 hidden features
- **Input**: molecular graph (positions, atomic types, edges) + time t
- **Output**: per-atom displacement vectors (score prediction)
- **Equivariance**: translation-invariant (COM subtraction), rotation-equivariant via EGNN
- **Energy model**: eSEN (equivariant score-based energy network from FAIR Chemistry)

---

## 3. Fit analysis: our data vs adjoint sampling

### 3.1. Critical advantage: we DON'T need training data

Adjoint sampling learns to sample from `p(x) ~ exp(-E(x)/kT)` using only energy gradients, not reference samples. This is **perfectly suited** for TS generation because:

1. **TS are saddle points, not minima** — Boltzmann sampling never visits them. You can't collect a training set by standard MD.
2. **We have 87 TS** — far too few for standard diffusion training (SPICE uses thousands of molecules with multiple conformers each).
3. **We have a differentiable energy function** — DFTB0 via SCINE provides energy + forces (= -grad_E) at every geometry. This is exactly what adjoint sampling needs.

**The 87 TS are useful for evaluation/initialization, not training.** The adjoint matching loss uses energy gradients computed on-the-fly, not a fixed dataset.

### 3.2. The energy function question

Adjoint sampling's SPICE experiments use **eSEN** — a GPU ML potential trained on DFT data. Our pipeline uses **DFTB0** via SCINE Sparrow. Key considerations:

| | eSEN | DFTB0 (ours) |
|---|---|---|
| Accuracy | DFT-level (SPICE trained) | Semi-empirical (lower) |
| Speed | GPU, fast batch inference | CPU, serial per call |
| Differentiability | PyTorch autograd | Finite difference (forces from SCINE) |
| Saddle point quality | Unknown for TS | Verified: finds real TS |

**Problem**: Adjoint sampling requires `grad_E(x)` at every training step. The energy model is called inside the training loop (sampler.py line 107: `F = energy_model(graph_state)`). This needs to be:
- Fast (called thousands of times per epoch)
- Differentiable (for force computation)
- Accurate at TS geometries (not just near equilibrium)

**Options**:
1. **Use eSEN directly** — fast, differentiable, but not trained on DFTB0 TS data. May not have correct TS energetics for isopropanol.
2. **Wrap DFTB0 as energy model** — accurate for our TS, but slow (CPU) and forces come from SCINE, not autograd.
3. **Train a small ML potential on DFTB0 data** — collect (geometry, energy, forces) triples from our NR/GAD trajectories, train a small equivariant model, use that as the energy function. Best of both worlds.

### 3.3. Dimensionality: 12 atoms = 36 DOF (30 after Eckart)

Isopropanol has 12 atoms = 36 Cartesian DOF. After removing 3 translations + 3 rotations = 30 vibrational DOF. This is **very small** compared to SPICE molecules (up to 100 atoms, 300 DOF).

**Implication**: A much smaller model should suffice. The EGNN with 12 layers / 128 hidden is massive overkill for 12 atoms. Could use:
- 4-6 layers, 64 hidden features
- Faster training, fewer samples needed
- Possibly even a simple MLP on internal coordinates

### 3.4. Single molecule vs amortized

The SPICE setup is **amortized** across many molecule types: the model takes SMILES as conditioning and learns conformer generation for diverse molecules simultaneously. Our setup is:
- **Single molecule** (isopropanol)
- **Goal**: sample TS geometries (not equilibrium conformers)

This means we can:
1. Drop the SMILES conditioning entirely
2. Focus the entire model capacity on one 12-atom molecule
3. Use a single molecular graph topology (fixed edges)

### 3.5. Target distribution: saddle points, not minima

Standard adjoint sampling targets `p(x) ~ exp(-E(x)/kT)`. This samples low-energy regions (minima). We want **saddle points** — these are exponentially rare in the Boltzmann distribution.

**This is the fundamental challenge.** Options:

#### Option A: Modified energy landscape
Define a TS-biased energy:
```
E_TS(x) = E(x) + lambda * penalty(n_neg(x) != 1)
```
where `penalty` is large when the geometry is not a TS. Problem: computing n_neg requires a Hessian (expensive), and the penalty is discontinuous.

#### Option B: Saddle-point-focused loss
Instead of sampling from `exp(-E)`, define the target as geometries where `||grad E|| = 0` and `n_neg = 1`. The adjoint score target becomes:
```
score_target = -grad(||F||^2) = -2 * H^T * F
```
This drives the diffusion toward zero-force geometries. Combined with an n_neg constraint, this could work.

#### Option C: Conditional generation from our TS data
Use the 87 TS as a **conditional** target: train a standard conditional diffusion model to generate geometries that look like known TS. This abandons the "no training data" advantage but leverages our TS collection. The 87 TS become the denoising targets.

#### Option D: Two-stage approach
1. **Stage 1**: Use adjoint sampling with standard `exp(-E)` energy to learn the equilibrium manifold of isopropanol
2. **Stage 2**: Fine-tune/modify the controller to target saddle points by changing the score target to incorporate Hessian information

### 3.6. Noise schedule considerations

The geometric noise schedule uses sigma_min=1e-3, sigma_max=1.0. Our TS coordinate spread:
- Per-atom std across all 87 TS: 0.50-0.80 A
- Key reaction coordinate (C0-H) range: 1.07-3.43 A (2.36 A span)

sigma_max=1.0 may be insufficient to cover the TS coordinate diversity. May need sigma_max=2.0 or higher. The noise schedule should be calibrated to the actual coordinate range of TS geometries.

### 3.7. Bond regularizer

Adjoint sampling includes a `bond_structure_regularizer` that penalizes unrealistic bond lengths. This is crucial for keeping geometries physical. However, **TS geometries naturally have stretched bonds** — the regularizer must be relaxed for the bonds that are being broken/formed.

For our 5 reaction families:
- F1: C1-H stretched to 2.17 A (would be penalized as broken)
- F3/F4: C0-H stretched to 2.12 A
- F5: C0-C1 stretched to 2.72 A

The regularizer needs to either be disabled for specific bonds or have its thresholds increased to accommodate TS geometries.

---

## 4. What we need to build

### 4.1. Energy model adapter

Wrap DFTB0 (via SCINE) to match the `FairChemEnergy` interface:
```python
class DFTB0Energy(torch.nn.Module):
    def forward(self, graph_state):
        positions = graph_state["positions"]
        # Call SCINE predict_fn
        result = predict_fn(positions, atomic_nums, do_hessian=False)
        return {
            "energy": result["energy"],
            "forces": result["forces"],
            "reg_energy": 0,
            "reg_forces": torch.zeros_like(result["forces"]),
        }
```

Or better: pre-compute a dataset of (geometry, energy, forces) along NR/GAD trajectories and train a small equivariant model.

### 4.2. Dataset loader for isopropanol TS

Instead of SMILES + RDKit conformers, we need:
- Fixed molecular graph (12 atoms, known connectivity)
- Starting positions: either random noise or our 87 TS geometries
- No SMILES conditioning needed

### 4.3. Modified score target for saddle points

The key research question. The standard adjoint score target:
```
score_target = -(grad_E - x/sigma^2)
```
drives sampling toward energy minima. For TS, we need to modify this to drive toward saddle points. Possible approaches:

**Approach 1: Force-norm minimization**
```
E_TS(x) = ||F(x)||^2 / 2
grad E_TS = H^T @ F   (Hessian-transpose times force)
```
This has zero gradient at all stationary points (minima AND saddle points). Need additional mechanism to select saddle points specifically.

**Approach 2: GAD-inspired score**
The GAD algorithm follows `F_GAD = F - 2*(F . v)*v` where v is the lowest eigenvector. This converts the TS search into an effective minimization. Could define:
```
score_target = -(F_GAD_grad - x/sigma^2)
```

**Approach 3: Conditional on reaction family**
Train separate controllers for each of our 5 reaction families, using family-specific TS geometries as initialization.

### 4.4. Reduced model for 12 atoms

```yaml
controller:
  n_atoms: 12        # fixed (vs 100 for SPICE)
  hidden_nf: 64      # smaller (vs 128)
  n_layers: 6        # fewer (vs 12)

noise_schedule:
  sigma_min: 1e-3
  sigma_max: 2.0     # larger to cover TS coordinate spread

num_epochs: 1000     # fewer (single molecule)
batch_size: 32       # smaller
buffer_size: 200     # smaller
```

---

## 5. Concrete gap analysis

### What we have vs what we need

| Component | Have | Need | Gap |
|-----------|------|------|-----|
| TS geometries | 87 converged TS | Evaluation set | Small — 87 is enough for eval |
| Energy + forces | DFTB0/SCINE (CPU) | Fast differentiable energy | **Large** — need GPU energy model |
| Molecular graph | Atom types + coords | PyG Data objects | Small — conversion needed |
| Score target | Standard (energy min) | Saddle-point-aware | **Large** — core research question |
| Neural network | EGNN (too large) | Smaller EGNN | Small — just reduce hyperparams |
| Noise schedule | Geometric | Calibrated to TS range | Small — tune sigma_max |
| Bond regularizer | Standard bond constraints | TS-aware relaxed constraints | Medium — need custom thresholds |
| Evaluation metrics | RMSD to reference | TS validity (n_neg=1, force<0.01) | Medium — need custom eval |

### Critical path

1. **Energy model** (high priority): Either adapt eSEN for isopropanol TS or train a small ML potential on DFTB0 trajectory data
2. **Score target modification** (high priority, research): How to make adjoint sampling target saddle points instead of minima
3. **Data pipeline** (low priority): Convert our data to PyG format, write dataset loader
4. **Model adaptation** (low priority): Smaller EGNN, tune hyperparams

---

## 6. Data size assessment

### Is 87 TS enough?

For **standard diffusion training**: No. 87 samples is far too few to train a neural network.

For **adjoint sampling**: The 87 TS aren't training data — they're evaluation targets. The training uses energy gradients computed on-the-fly from the energy model. The question is whether the energy function (DFTB0 or ML surrogate) is good enough, not whether we have enough TS.

However, we may want more TS for:
- **Validation**: Are generated TS diverse enough? Do they cover all 5 reaction families?
- **Initialization**: Using known TS as starting points for the SDE could improve convergence
- **ML potential training**: If we train a surrogate energy model, we need thousands of (geometry, energy, force) triples — these come from NR/GAD trajectories, not just converged TS

### Data we can cheaply generate

Our NR/GAD pipeline produces not just converged TS but entire **trajectories**. Each seed generates ~1000 NR steps + ~500 GAD steps, each with (coords, energy, forces). For 87 converged seeds:
- ~87,000 NR trajectory points
- ~43,000 GAD trajectory points
- **~130,000 (geometry, energy, forces) triples**

This is more than enough to train a small ML potential as an energy surrogate.

---

## 7. Recommended next steps

### Step 3a: Build energy model surrogate
- Extract (coords, energy, forces) from NR/GAD trajectory data
- Train a small SchNet/PaiNN/MACE model on this data
- Validate against DFTB0 on held-out TS geometries

### Step 3b: Prototype data pipeline
- Convert isopropanol TS to PyG Data format
- Write minimal dataset class (single SMILES, fixed graph)
- Verify we can run adjoint sampling forward pass

### Step 3c: Investigate score target for saddle points
- Literature review: has anyone used diffusion/flow matching for saddle point sampling?
- Prototype the force-norm and GAD-inspired score targets
- Test on LJ potential first (fast, analytical Hessian available)

### Step 3d: Train and evaluate
- Start with standard adjoint sampling on isopropanol (sample equilibrium conformers)
- Modify score target for TS
- Evaluate: do generated geometries converge to TS under our NR/GAD optimizer?

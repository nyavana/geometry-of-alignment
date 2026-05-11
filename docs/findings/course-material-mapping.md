# EECS 6699 Lectures → Geometry-of-Alignment Project

How the 12 lectures in `docs/lectures/` map onto the four modules of this project.

## Lecture-by-lecture connection

| # | Title | Main topics | Project link |
|---|-------|-------------|--------------|
| 1 | Introduction | Course overview: approximation/expressivity, NTK, double descent, generalization, residual nets, transformers | **background** — sets up the high-dim geometry / generalization / transformer lens that the project applies to alignment. |
| 2 | Expressive/Approximation Power of Shallow Nets | ERM, hypothesis classes, regularization, curse of dimensionality, low-dim structure | **`mechanistic/`** — "low-dim structure in high-dim data" motivates extracting a *single* refusal direction from the 2560-dim residual stream. |
| 3 | Expressiveness of Shallow Nets, Part 2 | Function approximation, uniform continuity, regularity conditions | **background** — formal foundation for treating MLP blocks as approximators. |
| 4 | Error Rates, Depth, Kernels & Neural Nets | Sobolev spaces, ε-approximation rates, depth separation | **background** — supports "depth matters" framing; no direct module use. |
| 5 | Interpolation, Over-parametrization, Kernels, **High-Dim Geometry** | Kolmogorov superposition, sparse grids, volume concentration, near-orthogonality of random vectors | **`mechanistic/`, `abliterate/`** — near-orthogonality and shell-concentration in high dim are exactly *why* a rank-1 perturbation along one direction can be surgical. |
| 6 | Convergence of GD, Concentration Inequalities, **Matrix Perturbation** | n-cube/n-ball volume concentration, Hoffman–Wielandt eigenvalue perturbation bound | **`abliterate/`, `weight_diff/`** — Hoffman–Wielandt bounds how much `W → W − α·d·dᵀW` can shift eigenstructure; the math behind "low-norm rank-1 edit preserves most of the network." |
| 7 | Over-parametrization, Convergence, NTK | Eigendecomposition, **matrix norms** (spectral / Frobenius), matrix perturbation theory | **`weight_diff/`** — Frobenius/spectral norms and the eigendecomposition machinery are exactly what `svd_analysis.py` computes on the weight diffs. |
| 8 | Lazy Training; PAC Learning & Generalization | NTK formalism, lazy regime, linearization around init, generalization bounds | **`abliterate/`** — lazy/linearization view legitimizes treating a tiny rank-1 edit as a localized intervention rather than a retrain. |
| 9 | Lazy Training & Rademacher Complexity | NTK scaling (α = 1/√m), tangent model, weights stay near init | **`abliterate/`, `weight_diff/`** — abliteration diffs *are* small perturbations off the pretrained weights; the lazy-training picture predicts behavior should be approximately a linear functional of `d`. |
| 10 | Mean Field Regime, Rademacher, Generalization Bounds | Mean-field vs lazy, weight movement, NN generalization | **background** — frames why fine-tuning (mean-field, weights move a lot) differs structurally from abliteration (lazy, rank-1 edit). |
| 11 | Generalization Bounds: A Priori vs A Posteriori | Empirical risk, hypothesis classes, regularized vs unregularized bounds | **`benchmark/`** — generalization framing for why a 344-prompt held-out benchmark validates behavior shifts; conceptual only. |
| 12 | VC Dimension, Implicit Bias of GD, ResNets & ODEs | VC-dim, shattering, implicit bias of GD, ResNet-as-ODE | **`mechanistic/`** — ResNet-as-ODE view legitimizes treating the residual stream as an evolving state and intervening layer-by-layer (forward hooks, layer sweep). |

## Module-by-module summary

### `mechanistic/` — refusal direction extraction, layer analysis, UMAP/t-SNE
- **Lec 2 & 5** — low-dim structure in high-dim data justifies a single mean-difference direction over a high-rank probe.
- **Lec 12** — ResNet-as-ODE underwrites per-layer hooks on the residual stream.

### `abliterate/` — rank-1 edit `W ← W − α·d·dᵀW`, ablation study, selective safety
- **Lec 5** — high-dim geometry (volume concentration, near-orthogonality) is the load-bearing result. In 2560 dims, random vectors are near-orthogonal and mass concentrates in thin shells, which is *why* one rank-1 direction can suppress refusal without scrambling task-relevant subspaces.
- **Lec 6** — Hoffman–Wielandt gives the formal upper bound on eigenstructure shift under the rank-1 edit.
- **Lec 7–9** — lazy/NTK regime predicts the output change is approximately a linear functional of `d`. Consistent with the M6 finding that rank-1 alone is only ~40% effective on Gemma 4: the lazy picture says it should be linear, not complete, and that's what was observed.

### `weight_diff/` — CPU diff of base vs OBLITERATUS / TrevorJS, SVD analysis
- **Lec 6–7** — matrix norms (Frobenius, spectral), eigendecomposition, perturbation theory. Exactly the machinery `svd_analysis.py` runs on the diff matrices. The course gives the bounds; the diffs give the empirical singular-value spectra.

### `benchmark/` — refusal evaluation across models
- **Lec 10–11** — generalization theory frames why a held-out 344-prompt benchmark across 8 categories is a valid measurement of behavior shift after a weight edit.

## Synthesis: the central course threads for this project

The thesis sits at the intersection of three course threads:

1. **High-dim geometry (Lec 5–6)** — the *why* of rank-1 abliteration working at all.
2. **Matrix perturbation + spectral theory (Lec 6–7)** — the *how* (and the SVD tooling).
3. **Lazy / NTK regime (Lec 7–9)** — the *limit*: predicts rank-1 is linear, hence partial, hence the M6 result that Gram-Schmidt residual structure also matters.

Lectures **5, 6, 7, and 9** form the tightest mathematical anchor for the abliteration claim; Lectures **2 and 12** anchor the mechanistic side. Lectures 1, 3, 4, 8, 10, 11 are background framing rather than directly-used machinery.

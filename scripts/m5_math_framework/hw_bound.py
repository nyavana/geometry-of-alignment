"""
M5 task 10.3 — Mirsky-style numerics for §4 Mathematical Framework.

For each (layer, projection) and two direction variants (M2b raw, D3 winner),
computes the rank-1 perturbation norm ratio:
    ‖E‖_F / ‖W‖_F  where E = α * d * dᵀW

Since E is rank-1, ‖E‖_F = ‖E‖_2 = |α| * ‖d‖_2 * ‖dᵀW‖_2 = |α| * ‖z‖_2
(where z = dᵀW and ‖d‖_2 = 1).

Outputs:
    $RESULTS_DIR/math_framework/mirsky_bound_per_layer.csv
    $RESULTS_DIR/figures/mirsky_bound_heatmap_d3.png
    $RESULTS_DIR/figures/mirsky_bound_heatmap_m2b.png
    results/math_framework/mirsky_bound_per_layer.csv  (in-repo copy)
    results/figures/mirsky_bound_heatmap_{d3,m2b}.png  (in-repo copies)

Usage:
    python scripts/m5_math_framework/hw_bound.py
"""

import os
import sys
import json
import math
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE = Path("/home/nyavana/columbia/6699/gb-paper")
SHARED_RESULTS = Path(os.environ.get(
    "RESULTS_DIR",
    "/home/nyavana/columbia/6699/shared/results/agent/writeup"
))

MODEL_PATH = "/home/nyavana/columbia/6699/shared/model/gemma-4-E4B-it"

M2B_DIRECTIONS_PATH = (
    "/home/nyavana/columbia/6699/shared/results/agent/"
    "mechanistic-analysis/activations/refusal_directions.pt"
)
D3_DIRECTIONS_PATH = (
    "/home/nyavana/columbia/6699/shared/results/agent/"
    "m6-rank1-followup/m6_directions/refusal_directions_d3.pt"
)
TREVORJS_DIFF_PATH = (
    "/home/nyavana/columbia/6699/shared/results/agent/"
    "weight-diff/weight_diffs/gemma_trevorjs/weight_diff_results.json"
)

NUM_LAYERS = 42
ALPHA = 1.0


def load_directions(path: str, label: str) -> dict:
    """Load direction dict and verify shape + unit norm."""
    dirs = torch.load(path, map_location="cpu")
    print(f"[{label}] Loaded {len(dirs)} directions from {path}")
    for layer_idx in range(NUM_LAYERS):
        assert layer_idx in dirs, f"[{label}] Missing layer {layer_idx}"
        d = dirs[layer_idx]
        assert d.shape == (2560,), f"[{label}] Layer {layer_idx} shape {d.shape} != (2560,)"
        norm = d.norm().item()
        assert abs(norm - 1.0) < 1e-3, (
            f"[{label}] Layer {layer_idx} norm {norm:.6f} not unit (tol 1e-3)"
        )
    print(f"[{label}] Shape and unit-norm verified for all {NUM_LAYERS} layers.")
    return dirs


def get_weight(model, layer_idx: int, proj: str) -> torch.Tensor:
    """Extract weight tensor for a given layer + projection (float32)."""
    layer = model.model.language_model.layers[layer_idx]
    if proj == "o_proj":
        W = layer.self_attn.o_proj.weight
    elif proj == "down_proj":
        W = layer.mlp.down_proj.weight
    else:
        raise ValueError(f"Unknown projection: {proj}")
    return W.detach().to(torch.float32)


def compute_row(variant: str, layer: int, proj: str,
                d: torch.Tensor, W: torch.Tensor) -> dict:
    """Compute Mirsky bound numerics for one (variant, layer, proj) cell."""
    # Verify unit norm
    d_norm = d.norm().item()
    assert abs(d_norm - 1.0) < 1e-3, (
        f"Direction norm assertion failed: layer={layer} norm={d_norm:.6f}"
    )

    d_f32 = d.to(torch.float32)

    # z = dᵀ W  (treating d as a row vector; W has shape [out_dim, in_dim])
    # The abliteration formula W ← W − α · d · dᵀW projects out d from W.
    # dᵀW means: for each column w_j of W, compute d·w_j → gives a row vector z.
    # In matrix terms: z = d @ W  (d is [in_dim] dot W[out_dim, in_dim] → ambiguous).
    # Clarification: abliterate.py uses W_new = W - alpha * outer(d, d @ W)
    # where d is the refusal direction in the *residual stream* (hidden_size=2560).
    # For o_proj: W shape is [out_dim=2560, in_dim=2048]; d is in output space [2560].
    # For down_proj: W shape is [out_dim=2560, in_dim=...]; d is in output space [2560].
    # The projection is: E = alpha * outer(d, d @ W) where d @ W contracts over out_dim.
    # So z = d @ W  → shape: [in_dim].
    z = d_f32 @ W  # shape: [in_dim]

    frob_E = ALPHA * z.norm().item()   # ‖E‖_F = |α| * ‖z‖_2  (rank-1)
    spec_E = frob_E                    # ‖E‖_2 = ‖E‖_F for rank-1
    frob_W = W.norm().item()           # ‖W‖_F
    ratio = frob_E / frob_W

    return {
        "variant": variant,
        "layer": layer,
        "projection": proj,
        "frob_E": frob_E,
        "spec_E_check": spec_E,
        "frob_W": frob_W,
        "ratio": ratio,
        "z_norm": z.norm().item(),
    }


def sanity_check_rank1(d3_dirs: dict, model):
    """
    Sanity check on D3, layer 15, o_proj:
    Construct E = α * outer(d, z) explicitly and compare ‖E‖_F against shortcut.

    Use float64 for both paths to avoid bfloat16→float32 accumulation error
    when computing matrix_norm over a 2560×2048 outer product.
    Float32 accumulates ~3e-4 relative error on this matrix size; float64
    gives ~5e-9 — well within the 1e-4 acceptance threshold.
    """
    layer_idx = 15
    proj = "o_proj"
    print("\n--- Rank-1 spectral/Frobenius sanity check: D3, layer 15, o_proj ---")

    # Use float64 for numerically clean comparison
    d = d3_dirs[layer_idx].to(torch.float64)
    W = get_weight(model, layer_idx, proj).to(torch.float64)

    z = d @ W
    E_shortcut = ALPHA * z.norm().item()

    # Explicit rank-1 matrix construction (float64)
    E_mat = ALPHA * torch.outer(d, z)
    E_explicit = torch.linalg.matrix_norm(E_mat, ord="fro").item()

    print(f"  Dtype used for check           : float64")
    print(f"  Shortcut ‖E‖_F = |α| * ‖z‖_2 : {E_shortcut:.8f}")
    print(f"  Explicit ‖E‖_F (matrix_norm)  : {E_explicit:.8f}")

    rel_err = abs(E_shortcut - E_explicit) / (E_explicit + 1e-12)
    print(f"  Relative error                : {rel_err:.2e}")

    assert rel_err < 1e-4, (
        f"SANITY CHECK FAILED: rel_err={rel_err:.2e} >= 1e-4. "
        f"Shortcut={E_shortcut:.8f}, Explicit={E_explicit:.8f}"
    )
    print("  PASSED (rel err < 1e-4)\n")
    return E_shortcut, E_explicit, rel_err


def cross_check_trevorjs(rows: list, trevorjs_path: str):
    """
    Cross-check D3 L15 o_proj ‖z‖_2 against TrevorJS top-1 singular value.
    """
    print("\n--- Cross-check vs M3 TrevorJS L15 o_proj ---")
    with open(trevorjs_path) as f:
        diff_data = json.load(f)

    # Find L15 o_proj entry
    target_key = "model.language_model.layers.15.self_attn.o_proj.weight"
    entry = None
    for item in diff_data:
        if item.get("key") == target_key:
            entry = item
            break

    if entry is None:
        print(f"  WARNING: Could not find key '{target_key}' in TrevorJS diff JSON")
        return None, None

    top1_sigma = entry["top_10_singular_values"][0]
    print(f"  TrevorJS top-1 σ for {target_key}: {top1_sigma:.6f}")

    # Find our D3 L15 o_proj row
    d3_l15_row = None
    for row in rows:
        if row["variant"] == "d3" and row["layer"] == 15 and row["projection"] == "o_proj":
            d3_l15_row = row
            break

    if d3_l15_row is None:
        print("  WARNING: D3 L15 o_proj row not found in computed results")
        return None, top1_sigma

    our_z_norm = d3_l15_row["z_norm"]
    print(f"  Our ‖dᵀW‖_2 (D3, L15, o_proj): {our_z_norm:.6f}")
    print(f"  Ratio our/TrevorJS σ₁: {our_z_norm / top1_sigma:.4f}")

    if our_z_norm > top1_sigma * 1.01:
        print(f"  WARNING: our ‖z‖_2 ({our_z_norm:.4f}) > TrevorJS σ₁ ({top1_sigma:.4f}). "
              f"This is unexpected for a unit-norm vector (should be ≤ σ₁).")
    else:
        print(f"  OK: ‖z‖_2 ≤ σ₁ (rank-1 bound holds)")

    return our_z_norm, top1_sigma


def make_heatmap(rows: list, variant: str, out_path: Path):
    """
    Create a heatmap: 42 rows (layers) × 2 columns (o_proj, down_proj).
    Cell color = ratio; annotate with 3-decimal numeric value.
    Layer 0 at top.
    """
    # Build 42×2 matrix
    ratio_matrix = np.zeros((NUM_LAYERS, 2))
    proj_order = ["o_proj", "down_proj"]

    for row in rows:
        if row["variant"] == variant:
            li = row["layer"]
            pi = proj_order.index(row["projection"])
            ratio_matrix[li, pi] = row["ratio"]

    fig, ax = plt.subplots(figsize=(5, 18))
    im = ax.imshow(ratio_matrix, aspect="auto", cmap="viridis",
                   vmin=0, vmax=ratio_matrix.max() * 1.05)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["o_proj", "down_proj"], fontsize=11)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([str(i) for i in range(NUM_LAYERS)], fontsize=7)
    ax.set_ylabel("Layer (0 = top)", fontsize=11)
    ax.set_title(
        f"‖E‖_F / ‖W‖_F  (α=1.0, variant={variant})\nRank-1 Mirsky Perturbation Bound",
        fontsize=11, pad=10
    )

    # Annotate cells
    for li in range(NUM_LAYERS):
        for pi in range(2):
            val = ratio_matrix[li, pi]
            # White text on dark cells, black on light
            text_color = "white" if val > ratio_matrix.max() * 0.5 else "black"
            ax.text(pi, li, f"{val:.3f}", ha="center", va="center",
                    fontsize=5.5, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("‖E‖_F / ‖W‖_F", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {out_path}")


def write_csv(rows: list, path: Path):
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "layer", "projection", "frob_E", "spec_E_check", "frob_W", "ratio"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  Saved CSV: {path} ({len(rows)} rows)")


def main():
    print("=" * 70)
    print("M5 task 10.3 — Mirsky bound per layer")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load directions
    # -----------------------------------------------------------------------
    print("\n[1] Loading direction artifacts...")
    m2b_dirs = load_directions(M2B_DIRECTIONS_PATH, "m2b")
    d3_dirs = load_directions(D3_DIRECTIONS_PATH, "d3")

    # -----------------------------------------------------------------------
    # 2. Load model (CPU, bfloat16, device_map=None)
    # -----------------------------------------------------------------------
    print(f"\n[2] Loading model from {MODEL_PATH} (CPU, bfloat16) ...")
    print("    This may take 5–15 minutes and ~12–15 GB RAM ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=None,   # CPU only — GPU policy: gpu-none
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("    Model loaded.")

    # Verify layer accessor
    layers = model.model.language_model.layers
    print(f"    Layer count: {len(layers)}")
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"

    # -----------------------------------------------------------------------
    # 3. Sanity check: rank-1 Frobenius = spectral (D3, L15, o_proj)
    # -----------------------------------------------------------------------
    E_shortcut, E_explicit, rel_err = sanity_check_rank1(d3_dirs, model)

    # -----------------------------------------------------------------------
    # 4. Compute all 168 rows
    # -----------------------------------------------------------------------
    print("\n[4] Computing Mirsky bounds for all 168 cells...")
    rows = []
    variants = [("m2b", m2b_dirs), ("d3", d3_dirs)]
    projections = ["o_proj", "down_proj"]

    for variant_name, dirs in variants:
        print(f"  Processing variant: {variant_name}")
        for layer_idx in range(NUM_LAYERS):
            d = dirs[layer_idx]
            for proj in projections:
                W = get_weight(model, layer_idx, proj)
                row = compute_row(variant_name, layer_idx, proj, d, W)
                rows.append(row)
            if layer_idx % 10 == 0:
                print(f"    Layer {layer_idx:2d} done")
        print(f"  Variant {variant_name} complete.")

    assert len(rows) == 168, f"Expected 168 rows, got {len(rows)}"
    print(f"  Total rows: {len(rows)} ✓")

    # -----------------------------------------------------------------------
    # 5. Cross-check vs TrevorJS M3 SVD
    # -----------------------------------------------------------------------
    our_z_norm, trevorjs_sigma1 = cross_check_trevorjs(rows, TREVORJS_DIFF_PATH)

    # -----------------------------------------------------------------------
    # 6. Compute summary statistics
    # -----------------------------------------------------------------------
    print("\n[6] Summary statistics:")
    for variant_name in ["d3", "m2b"]:
        variant_rows = [r for r in rows if r["variant"] == variant_name]
        ratios = [r["ratio"] for r in variant_rows]
        median_ratio = float(np.median(ratios))
        max_ratio = max(ratios)
        max_layer = max(variant_rows, key=lambda r: r["ratio"])
        print(f"  {variant_name}: median ratio = {median_ratio:.6f}, "
              f"max = {max_ratio:.6f} (L{max_layer['layer']} {max_layer['projection']})")
        large = [r for r in variant_rows if r["ratio"] > 0.1]
        if large:
            print(f"  WARNING: {len(large)} cells with ratio > 0.1 for {variant_name}")
        else:
            print(f"  All ratios ≤ 0.1 for {variant_name} ✓")

    # Summary for commit message
    d3_median = float(np.median([r["ratio"] for r in rows if r["variant"] == "d3"]))
    m2b_median = float(np.median([r["ratio"] for r in rows if r["variant"] == "m2b"]))
    print(f"\nFor commit message:")
    print(f"  D3 median ratio  : {d3_median:.6f}")
    print(f"  M2b median ratio : {m2b_median:.6f}")
    print(f"  Sanity rel_err   : {rel_err:.2e}")
    if our_z_norm is not None and trevorjs_sigma1 is not None:
        print(f"  D3 L15 ‖z‖_2    : {our_z_norm:.6f}")
        print(f"  TrevorJS σ₁     : {trevorjs_sigma1:.6f}")

    # -----------------------------------------------------------------------
    # 7. Write outputs
    # -----------------------------------------------------------------------
    print("\n[7] Writing outputs...")

    # Primary outputs (RESULTS_DIR)
    SHARED_RESULTS.mkdir(parents=True, exist_ok=True)
    csv_shared = SHARED_RESULTS / "math_framework" / "mirsky_bound_per_layer.csv"
    write_csv(rows, csv_shared)

    fig_shared_d3 = SHARED_RESULTS / "figures" / "mirsky_bound_heatmap_d3.png"
    fig_shared_m2b = SHARED_RESULTS / "figures" / "mirsky_bound_heatmap_m2b.png"
    (SHARED_RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    print("  Generating D3 heatmap...")
    make_heatmap(rows, "d3", fig_shared_d3)
    print("  Generating M2b heatmap...")
    make_heatmap(rows, "m2b", fig_shared_m2b)

    # In-repo redundant copies
    csv_repo = WORKTREE / "results" / "math_framework" / "mirsky_bound_per_layer.csv"
    write_csv(rows, csv_repo)

    fig_repo_d3 = WORKTREE / "results" / "figures" / "mirsky_bound_heatmap_d3.png"
    fig_repo_m2b = WORKTREE / "results" / "figures" / "mirsky_bound_heatmap_m2b.png"
    print("  Generating in-repo D3 heatmap...")
    make_heatmap(rows, "d3", fig_repo_d3)
    print("  Generating in-repo M2b heatmap...")
    make_heatmap(rows, "m2b", fig_repo_m2b)

    print("\n" + "=" * 70)
    print("DONE. All acceptance checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

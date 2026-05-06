"""
Cross-method comparison of OBLITERATUS vs TrevorJS weight diffs.

Reads both variants' weight_diff_results.json files and the significant_diff_svd.pt
files produced by compute_diff.py, then:

1. Overlaid per-layer Frobenius bar chart for both methods.
2. For each weight present (and significantly modified) in both variants, compute
   cosine similarity between their top-1 left singular vectors.
3. Output: CSV table + PNG heatmap/dot-plot of cosine similarities.

Usage:
    python -m src.weight_diff.cross_method \\
        --obliteratus-results <path/to/gemma_obliteratus/weight_diff_results.json> \\
        --trevorjs-results   <path/to/gemma_trevorjs/weight_diff_results.json> \\
        --output             <output_dir/>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def per_layer_frobenius(results: list[dict]) -> dict[int, float]:
    """Sum Frobenius norm of all changed weights per layer index."""
    layer_norm: dict[int, float] = defaultdict(float)
    for r in results:
        if r["frobenius_norm"] <= 1e-6:
            continue
        parts = r["key"].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_norm[int(parts[i + 1])] += r["frobenius_norm"]
                except ValueError:
                    pass
                break
    return dict(layer_norm)


def significant_keys(results: list[dict], threshold: float = 0.001) -> set[str]:
    """Return weight names with relative_change > threshold."""
    return {r["key"] for r in results if r.get("relative_change", 0) > threshold}


def load_top1_left_singular_vectors(svd_pt_path: str) -> dict[str, np.ndarray]:
    """
    Load the top-1 left singular vectors from significant_diff_svd.pt.

    The .pt file is a dict keyed by weight name:
        {key: {"U_top5": Tensor[d_out, 5], "S_top5": Tensor[5], "Vh_top5": Tensor[5, d_in]}}

    Returns {key: unit_vector_np} (first column of U_top5, L2-normalised).
    """
    data: dict = torch.load(svd_pt_path, map_location="cpu", weights_only=True)
    out: dict[str, np.ndarray] = {}
    for key, svd in data.items():
        u = svd["U_top5"]  # shape [d_out, 5]
        v = u[:, 0].float().numpy()
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm
        out[key] = v
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; vectors are expected to be pre-normalised."""
    dot = float(np.dot(a, b))
    # clamp to [-1, 1] to guard against fp rounding
    return max(-1.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41}  # Gemma 4 E4B global attention layers


def plot_per_layer_overlay(
    obliteratus_layers: dict[int, float],
    trevorjs_layers: dict[int, float],
    output_path: str,
) -> None:
    all_layers = sorted(set(obliteratus_layers) | set(trevorjs_layers))
    x = np.arange(len(all_layers))
    width = 0.4

    obl = [obliteratus_layers.get(l, 0.0) for l in all_layers]
    trv = [trevorjs_layers.get(l, 0.0) for l in all_layers]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars_obl = ax.bar(x - width / 2, obl, width, label="OBLITERATUS", alpha=0.8, color="#1f77b4")
    bars_trv = ax.bar(x + width / 2, trv, width, label="TrevorJS", alpha=0.8, color="#ff7f0e")

    # Mark global attention layers
    for i, l in enumerate(all_layers):
        if l in GLOBAL_LAYERS:
            ax.axvline(x=i, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_layers, rotation=90, fontsize=7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Total ||ΔW||_F per Layer")
    ax.set_title(
        "Per-Layer Weight Modification: OBLITERATUS vs TrevorJS\n"
        "(dotted vertical lines = global attention layers)"
    )
    ax.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved overlay chart: {output_path}")


def plot_cosine_heatmap(
    rows: list[dict],
    output_path: str,
) -> None:
    """
    Heatmap: rows = layer indices, columns = weight suffixes (e.g. o_proj.weight),
    colour = cosine similarity.  If > 60 rows, switch to a dot-plot strip instead.
    """
    # Build 2-D grid
    layer_set: set[int] = set()
    wname_set: set[str] = set()
    for r in rows:
        layer_set.add(r["layer_idx"])
        wname_set.add(r["weight_suffix"])

    layers = sorted(layer_set)
    wnames = sorted(wname_set)

    grid = np.full((len(layers), len(wnames)), np.nan)
    l_idx = {l: i for i, l in enumerate(layers)}
    w_idx = {w: i for i, w in enumerate(wnames)}

    for r in rows:
        grid[l_idx[r["layer_idx"]], w_idx[r["weight_suffix"]]] = r["cosine_similarity"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(layers) <= 60:
        fig, ax = plt.subplots(figsize=(max(6, len(wnames) * 1.5), max(6, len(layers) * 0.35)))
        im = ax.imshow(grid, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_xticks(range(len(wnames)))
        ax.set_xticklabels(wnames, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel("Weight Name")
        ax.set_ylabel("Layer Index")
        ax.set_title("Top-1 Left Singular Vector Cosine Similarity\nOBLITERATUS vs TrevorJS")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        # dot-plot strip: x = layer, y = cosine, colour per weight type
        fig, ax = plt.subplots(figsize=(16, 5))
        cmap = plt.cm.get_cmap("tab10", len(wnames))
        for wi, wname in enumerate(wnames):
            xs, ys = [], []
            for li, l in enumerate(layers):
                v = grid[li, wi]
                if not np.isnan(v):
                    xs.append(l)
                    ys.append(v)
            ax.scatter(xs, ys, label=wname, color=cmap(wi), s=15, alpha=0.7)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Top-1 Left Singular Vector Cosine Similarity\nOBLITERATUS vs TrevorJS")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close()
    print(f"Saved cosine figure: {output_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(
    obliteratus_results_path: str,
    trevorjs_results_path: str,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading weight diff results...")
    obl_results = load_results(obliteratus_results_path)
    trv_results = load_results(trevorjs_results_path)

    # 1. Per-layer Frobenius overlay chart
    obl_layers = per_layer_frobenius(obl_results)
    trv_layers = per_layer_frobenius(trv_results)

    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = str(figures_dir / "weight_diff_per_layer_overlay.png")
    plot_per_layer_overlay(obl_layers, trv_layers, overlay_path)

    # 2. Load top-1 left singular vectors from both variants
    obl_svd_path = Path(obliteratus_results_path).parent / "significant_diff_svd.pt"
    trv_svd_path = Path(trevorjs_results_path).parent / "significant_diff_svd.pt"

    obl_vecs: dict[str, np.ndarray] = {}
    trv_vecs: dict[str, np.ndarray] = {}

    if obl_svd_path.exists():
        print(f"Loading OBLITERATUS singular vectors from {obl_svd_path}")
        obl_vecs = load_top1_left_singular_vectors(str(obl_svd_path))
    else:
        print(f"WARNING: {obl_svd_path} not found — cosine analysis will be empty.")

    if trv_svd_path.exists():
        print(f"Loading TrevorJS singular vectors from {trv_svd_path}")
        trv_vecs = load_top1_left_singular_vectors(str(trv_svd_path))
    else:
        print(f"WARNING: {trv_svd_path} not found — cosine analysis will be empty.")

    # 3. Cosine similarity for keys significant in both variants
    obl_sig = significant_keys(obl_results)
    trv_sig = significant_keys(trv_results)
    common_sig = obl_sig & trv_sig
    common_with_vecs = common_sig & set(obl_vecs) & set(trv_vecs)

    print(f"Significant in OBLITERATUS: {len(obl_sig)}")
    print(f"Significant in TrevorJS:    {len(trv_sig)}")
    print(f"Significant in both:        {len(common_sig)}")
    print(f"Both + have SVD vectors:    {len(common_with_vecs)}")

    cosine_rows: list[dict] = []
    for key in sorted(common_with_vecs):
        cos = cosine_sim(obl_vecs[key], trv_vecs[key])
        parts = key.split(".")
        # Extract layer index
        layer_idx: int = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
                break
        # Weight suffix = last two parts (e.g. "self_attn.o_proj.weight" → "o_proj.weight")
        weight_suffix = ".".join(parts[-2:]) if len(parts) >= 2 else key
        cosine_rows.append({
            "layer_idx": layer_idx,
            "weight_name": key,
            "weight_suffix": weight_suffix,
            "cosine_similarity": round(cos, 6),
        })

    # Write CSV
    csv_path = out / "cross_method_cosine_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_idx", "weight_name",
                                                "weight_suffix", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(cosine_rows)
    print(f"Saved cosine table: {csv_path}")

    # Summary statistics
    if cosine_rows:
        cosines = [r["cosine_similarity"] for r in cosine_rows]
        cos_min = min(cosines)
        cos_med = float(np.median(cosines))
        cos_max = max(cosines)
        print(f"\nCosine similarity range: min={cos_min:.4f}, median={cos_med:.4f}, max={cos_max:.4f}")
    else:
        print("\nNo cosine rows to summarise (check SVD .pt files exist).")
        cos_min = cos_med = cos_max = float("nan")

    # Cosine figure
    cos_fig_path = str(figures_dir / "cross_method_singular_vectors.png")
    if cosine_rows:
        plot_cosine_heatmap(cosine_rows, cos_fig_path)
    else:
        print("Skipping cosine figure (no data).")

    print("\nCross-method analysis complete.")
    print(f"Outputs:")
    print(f"  {overlay_path}")
    print(f"  {csv_path}")
    if cosine_rows:
        print(f"  {cos_fig_path}")

    return cos_min, cos_med, cos_max


def main():
    parser = argparse.ArgumentParser(description="Cross-method weight diff comparison")
    parser.add_argument("--obliteratus-results", required=True,
                        help="Path to OBLITERATUS weight_diff_results.json")
    parser.add_argument("--trevorjs-results", required=True,
                        help="Path to TrevorJS weight_diff_results.json")
    parser.add_argument("--output", required=True,
                        help="Output directory for figures and CSV")
    args = parser.parse_args()

    run(
        obliteratus_results_path=args.obliteratus_results,
        trevorjs_results_path=args.trevorjs_results,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

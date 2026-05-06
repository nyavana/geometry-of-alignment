"""
Visualization of refuse vs comply activations using UMAP and t-SNE.

Usage:
    python -m src.mechanistic.visualize \
        --activations results/activations/ \
        --layers 0,5,10,15,20,25,30,35,41
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_single_layer(refuse_acts: dict, comply_acts: dict,
                           layer_idx: int, method: str = "umap",
                           output_dir: str = "results/figures",
                           filename: str | None = None):
    """2D projection of activations at a given layer.

    If ``filename`` is provided it overrides the default
    ``activation_viz_layer{idx}.png`` naming. Used by the M2b 4.7 driver
    to emit ``umap_layer_{idx:02d}.png`` / ``tsne_layer_{idx:02d}.png``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_refuse = len(refuse_acts[layer_idx])
    n_comply = len(comply_acts[layer_idx])
    all_acts = torch.cat([refuse_acts[layer_idx], comply_acts[layer_idx]], dim=0)
    is_refuse = np.array([True] * n_refuse + [False] * n_comply)

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, len(all_acts) - 1),
                       random_state=42, init="pca")
        embedding = reducer.fit_transform(all_acts.numpy())
    else:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1)
        embedding = reducer.fit_transform(all_acts.numpy())

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        embedding[is_refuse, 0], embedding[is_refuse, 1],
        c="#d62728", alpha=0.55, s=24, edgecolor="black", linewidth=0.2,
        label=f"Refuse (n={n_refuse})",
    )
    ax.scatter(
        embedding[~is_refuse, 0], embedding[~is_refuse, 1],
        c="#1f77b4", alpha=0.85, s=40, edgecolor="black", linewidth=0.4,
        marker="s", label=f"Comply (n={n_comply})",
    )
    ax.set_title(f"Layer {layer_idx} — {method.upper()} projection of residual stream")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(loc="best")
    plt.tight_layout()

    if filename is None:
        filename = f"activation_viz_layer{layer_idx}.png"
    out_path = Path(output_dir) / filename
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close()


def visualize_multi_layer(refuse_acts: dict, comply_acts: dict,
                          layers_to_show: list[int],
                          method: str = "umap",
                          output_dir: str = "results/figures"):
    """Grid of projections across multiple layers."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n = len(layers_to_show)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > cols else [axes] if n == 1 else axes.flatten()

    for i, layer_idx in enumerate(layers_to_show):
        all_acts = torch.cat([refuse_acts[layer_idx], comply_acts[layer_idx]])
        labels = np.array([1] * len(refuse_acts[layer_idx]) +
                          [0] * len(comply_acts[layer_idx]))

        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, len(all_acts) - 1),
                           random_state=42)
            emb = reducer.fit_transform(all_acts.numpy())
        else:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb = reducer.fit_transform(all_acts.numpy())

        axes[i].scatter(emb[:, 0], emb[:, 1], c=labels, cmap="RdBu",
                        alpha=0.5, s=15)
        axes[i].set_title(f"Layer {layer_idx}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Refuse vs Comply Activations Across Layers", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_layer_{method}.png", dpi=150)
    print(f"Saved multi-layer {method} to {output_dir}/multi_layer_{method}.png")
    plt.close()


def visualize_layer_grid(refuse_acts: dict, comply_acts: dict,
                         layers: list[int], method: str = "umap",
                         output_dir: str = "results/figures"):
    """
    M2b 4.7 driver: emit one ``{method}_layer_{idx:02d}.png`` per requested
    layer (refuse vs comply, 2D projection). No combined-grid figure — the
    spec asks for one PNG per layer, named consistently.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generating {method} layer grid for layers: {layers}")
    for layer_idx in layers:
        filename = f"{method}_layer_{layer_idx:02d}.png"
        visualize_single_layer(
            refuse_acts, comply_acts, layer_idx,
            method=method, output_dir=output_dir, filename=filename,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize refuse/comply activations")
    parser.add_argument("--activations", default="results/activations/")
    parser.add_argument("--layers", default="0,5,10,15,20,25,30,35,41",
                        help="Comma-separated layer indices to visualize")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--output", default="results/figures/")
    parser.add_argument(
        "--mode", default="legacy", choices=["legacy", "grid"],
        help=("legacy: emit multi_layer_<method>.png + activation_viz_layer<idx>.png. "
              "grid: emit only <method>_layer_<idx:02d>.png (M2b 4.7 deliverable)."),
    )
    args = parser.parse_args()

    act_dir = Path(args.activations)
    refuse_acts = torch.load(act_dir / "refuse_activations.pt", weights_only=True,
                             map_location="cpu")
    comply_acts = torch.load(act_dir / "comply_activations.pt", weights_only=True,
                             map_location="cpu")

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Visualizing layers: {layers}")

    if args.mode == "grid":
        visualize_layer_grid(refuse_acts, comply_acts, layers, args.method,
                             args.output)
        return

    visualize_multi_layer(refuse_acts, comply_acts, layers, args.method, args.output)

    # Also save individual plots for key layers
    for layer_idx in layers:
        visualize_single_layer(refuse_acts, comply_acts, layer_idx,
                               args.method, args.output)


if __name__ == "__main__":
    main()

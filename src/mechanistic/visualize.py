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
                           output_dir: str = "results/figures"):
    """2D projection of activations at a given layer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_acts = torch.cat([refuse_acts[layer_idx], comply_acts[layer_idx]], dim=0)
    labels = np.array([1] * len(refuse_acts[layer_idx]) +
                      [0] * len(comply_acts[layer_idx]))

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, len(all_acts) - 1),
                       random_state=42)
        embedding = reducer.fit_transform(all_acts.numpy())
    else:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(all_acts.numpy())

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap="RdBu", alpha=0.6, s=30
    )
    ax.set_title(f"Layer {layer_idx} — Refuse (red) vs Comply (blue)")
    plt.colorbar(scatter, ax=ax, label="Refuse ← → Comply")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/activation_viz_layer{layer_idx}.png", dpi=150)
    print(f"Saved layer {layer_idx} visualization")
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


def main():
    parser = argparse.ArgumentParser(description="Visualize refuse/comply activations")
    parser.add_argument("--activations", default="results/activations/")
    parser.add_argument("--layers", default="0,5,10,15,20,25,30,35,41",
                        help="Comma-separated layer indices to visualize")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    act_dir = Path(args.activations)
    refuse_acts = torch.load(act_dir / "refuse_activations.pt", weights_only=True)
    comply_acts = torch.load(act_dir / "comply_activations.pt", weights_only=True)

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Visualizing layers: {layers}")

    visualize_multi_layer(refuse_acts, comply_acts, layers, args.method, args.output)

    # Also save individual plots for key layers
    for layer_idx in layers:
        visualize_single_layer(refuse_acts, comply_acts, layer_idx,
                               args.method, args.output)


if __name__ == "__main__":
    main()

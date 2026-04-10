"""
Analyze the refusal direction across layers:
- Signal strength per layer
- PCA rank of the refusal subspace
- Sliding vs global attention layer comparison

Usage:
    python -m src.mechanistic.layer_analysis \
        --activations results/activations/
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Gemma 4 E4B layer types — 42 layers total
# IMPORTANT: verify these indices from config's layer_types field
# Pattern: every 6th layer starting from 5 is global attention
# Adjust after running print(model.config.text_config.layer_types) or similar
GLOBAL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41]


def get_layer_type(layer_idx: int) -> str:
    """Return 'global' or 'sliding' for a given layer index."""
    return "global" if layer_idx in GLOBAL_ATTENTION_LAYERS else "sliding"


def compute_signal_strength(refuse_acts: dict, comply_acts: dict) -> dict:
    """
    For each layer, compute how well refuse/comply are separated.
    """
    results = {}

    for layer_idx in sorted(refuse_acts.keys()):
        mean_r = refuse_acts[layer_idx].mean(dim=0)
        mean_c = comply_acts[layer_idx].mean(dim=0)
        diff = mean_r - mean_c

        signal_norm = diff.norm().item()

        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_r.unsqueeze(0), mean_c.unsqueeze(0)
        ).item()

        direction = diff / diff.norm()
        refuse_proj = (refuse_acts[layer_idx] @ direction).numpy()
        comply_proj = (comply_acts[layer_idx] @ direction).numpy()

        all_proj = np.concatenate([refuse_proj, comply_proj])
        separation = (refuse_proj.mean() - comply_proj.mean()) / (np.std(all_proj) + 1e-8)

        results[layer_idx] = {
            "signal_norm": signal_norm,
            "cosine_similarity": cosine_sim,
            "separation_score": separation,
            "layer_type": get_layer_type(layer_idx),
        }

    return results


def analyze_refusal_rank(refuse_acts: dict, comply_acts: dict,
                         n_components: int = 20) -> dict:
    """
    PCA rank analysis at each layer.
    Returns how many dimensions the refusal subspace occupies.
    """
    rank_results = {}

    for layer_idx in sorted(refuse_acts.keys()):
        all_acts = torch.cat([refuse_acts[layer_idx], comply_acts[layer_idx]], dim=0)
        centered = all_acts - all_acts.mean(dim=0)

        n_comp = min(n_components, centered.shape[0] - 1, centered.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(centered.numpy())

        cumulative = np.cumsum(pca.explained_variance_ratio_)

        rank_results[layer_idx] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumulative.tolist(),
            "rank_95": int((cumulative < 0.95).sum()) + 1,
            "rank_99": int((cumulative < 0.99).sum()) + 1,
            "top1_energy": float(pca.explained_variance_ratio_[0]),
        }

    return rank_results


def plot_signal_strength(results: dict, output_dir: str = "results/figures"):
    """Plot signal strength across layers, colored by attention type."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layers = sorted(results.keys())
    separations = [results[l]["separation_score"] for l in layers]
    colors = ["#2196F3" if results[l]["layer_type"] == "sliding" else "#FF5722"
              for l in layers]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(layers, separations, color=colors, alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Refuse/Comply Separation Score")
    ax.set_title("Refusal Signal Strength by Layer (Gemma 4 E4B)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Sliding Attention (35 layers)"),
        Patch(facecolor="#FF5722", label="Global Attention (7 layers)"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/layer_signal_strength.png", dpi=150)
    print(f"Saved to {output_dir}/layer_signal_strength.png")
    plt.close()


def plot_rank_analysis(rank_results: dict, output_dir: str = "results/figures"):
    """Plot effective rank and top-1 energy across layers."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layers = sorted(rank_results.keys())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top-1 energy fraction
    top1 = [rank_results[l]["top1_energy"] for l in layers]
    colors = ["#2196F3" if get_layer_type(l) == "sliding" else "#FF5722" for l in layers]
    axes[0].bar(layers, top1, color=colors, alpha=0.8)
    axes[0].set_ylabel("Top-1 PCA Component\nVariance Fraction")
    axes[0].set_title("Is Refusal Approximately Rank-1?")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    axes[0].legend()

    # Effective rank at 95%
    rank95 = [rank_results[l]["rank_95"] for l in layers]
    axes[1].bar(layers, rank95, color=colors, alpha=0.8)
    axes[1].set_ylabel("Effective Rank\n(95% variance)")
    axes[1].set_xlabel("Layer Index")
    axes[1].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Rank 1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rank_analysis.png", dpi=150)
    print(f"Saved to {output_dir}/rank_analysis.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Layer-wise refusal analysis")
    parser.add_argument("--activations", default="results/activations/")
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    act_dir = Path(args.activations)
    refuse_acts = torch.load(act_dir / "refuse_activations.pt", weights_only=True)
    comply_acts = torch.load(act_dir / "comply_activations.pt", weights_only=True)

    print(f"Loaded activations: {len(refuse_acts)} layers, "
          f"{refuse_acts[0].shape[0]} refuse prompts, "
          f"{comply_acts[0].shape[0]} comply prompts")

    # Signal strength
    print("\nComputing signal strength...")
    signal_results = compute_signal_strength(refuse_acts, comply_acts)
    plot_signal_strength(signal_results, args.output)

    # Print summary
    sliding_sep = [v["separation_score"] for v in signal_results.values()
                   if v["layer_type"] == "sliding"]
    global_sep = [v["separation_score"] for v in signal_results.values()
                  if v["layer_type"] == "global"]
    print(f"\nMean separation — sliding: {np.mean(sliding_sep):.3f}, "
          f"global: {np.mean(global_sep):.3f}")

    # Rank analysis
    print("\nComputing rank analysis...")
    rank_results = analyze_refusal_rank(refuse_acts, comply_acts)
    plot_rank_analysis(rank_results, args.output)

    # Save numerical results
    results_path = Path(args.activations) / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump({"signal_strength": signal_results, "rank_analysis": rank_results},
                  f, indent=2)
    print(f"\nSaved analysis to {results_path}")


if __name__ == "__main__":
    main()

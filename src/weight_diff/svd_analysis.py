"""
Visualization and deep analysis of weight diff SVD results.

Usage:
    python -m src.weight_diff.svd_analysis \
        --results results/weight_diffs/qwen/weight_diff_results.json \
        --output results/figures/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path


def load_results(results_path: str) -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def plot_param_type_changes(results: list[dict], output_dir: str):
    """Which parameter types were modified most?"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    changed = [r for r in results if r["frobenius_norm"] > 1e-6]
    type_changes = defaultdict(list)

    for r in changed:
        key = r["key"]
        if "self_attn" in key:
            ptype = "attention"
        elif "mlp" in key and "expert" not in key:
            ptype = "mlp"
        elif "expert" in key:
            ptype = "expert"
        elif "gate" in key or "router" in key:
            ptype = "router"
        elif "embed" in key:
            ptype = "embedding"
        elif "norm" in key:
            ptype = "norm"
        else:
            ptype = "other"
        type_changes[ptype].append(r["relative_change"])

    if not type_changes:
        print("No changed parameters found")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    types = sorted(type_changes.keys())
    means = [np.mean(type_changes[t]) for t in types]
    counts = [len(type_changes[t]) for t in types]

    bars = ax.bar(types, means)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'n={count}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Mean Relative Change")
    ax.set_title("Which Parameter Types Were Modified Most?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/param_type_changes.png", dpi=150)
    print(f"Saved {output_dir}/param_type_changes.png")
    plt.close()


def plot_modification_ranks(results: list[dict], output_dir: str):
    """Effective rank of weight modifications — low rank = abliteration signature."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ranked = [(r["key"], r["rank_95"], r.get("top1_energy_fraction", 0))
              for r in results if "rank_95" in r and r["frobenius_norm"] > 1e-6]

    if not ranked:
        print("No rank data available")
        return

    ranked.sort(key=lambda x: x[1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Rank at 95%
    names = [r[0].split(".")[-2] + "." + r[0].split(".")[-1] for r in ranked[:30]]
    rank_vals = [r[1] for r in ranked[:30]]
    axes[0].bar(range(len(names)), rank_vals, alpha=0.7)
    axes[0].set_ylabel("Effective Rank (95%)")
    axes[0].set_title("Rank of Weight Modifications (top 30 by rank)")
    axes[0].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Rank 1")
    axes[0].legend()
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=7)

    # Top-1 energy fraction
    top1 = [r[2] for r in ranked[:30]]
    axes[1].bar(range(len(names)), top1, alpha=0.7, color="orange")
    axes[1].set_ylabel("Top-1 SV Energy Fraction")
    axes[1].set_xlabel("Parameter")
    axes[1].axhline(y=0.9, color="red", linestyle="--", alpha=0.5,
                     label="90% (near rank-1)")
    axes[1].legend()
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=90, fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/modification_ranks.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir}/modification_ranks.png")
    plt.close()


def plot_per_layer_change(results: list[dict], output_dir: str):
    """Per-layer total Frobenius norm of weight changes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layer_changes = defaultdict(float)
    for r in results:
        if r["frobenius_norm"] <= 1e-6:
            continue
        parts = r["key"].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    layer_changes[layer_idx] += r["frobenius_norm"]
                except ValueError:
                    pass
                break

    if not layer_changes:
        print("No per-layer data available")
        return

    layers = sorted(layer_changes.keys())
    changes = [layer_changes[l] for l in layers]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(layers, changes, alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Total ||ΔW||_F")
    ax.set_title("Which Layers Were Modified Most by Cracking?")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_layer_weight_change.png", dpi=150)
    print(f"Saved {output_dir}/per_layer_weight_change.png")
    plt.close()


def plot_sv_spectra(results: list[dict], output_dir: str, top_n: int = 5):
    """Singular value spectra for the most modified weight matrices."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with_svs = [r for r in results
                if "top_10_singular_values" in r and r.get("relative_change", 0) > 0.01]
    with_svs.sort(key=lambda x: x["relative_change"], reverse=True)

    for r in with_svs[:top_n]:
        fig, ax = plt.subplots(figsize=(8, 4))
        svs = r["top_10_singular_values"]
        ax.bar(range(len(svs)), svs)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value")
        key_short = r["key"].replace("model.layers.", "L").replace(".", "/")
        ax.set_title(f"SV Spectrum of ΔW — {key_short}")
        plt.tight_layout()
        safe_name = r["key"].replace(".", "_")
        plt.savefig(f"{output_dir}/sv_spectrum_{safe_name}.png", dpi=150)
        plt.close()

    print(f"Saved {min(top_n, len(with_svs))} SV spectrum plots")


def main():
    parser = argparse.ArgumentParser(description="SVD analysis of weight diffs")
    parser.add_argument("--results", required=True, help="weight_diff_results.json path")
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    results = load_results(args.results)

    plot_param_type_changes(results, args.output)
    plot_modification_ranks(results, args.output)
    plot_per_layer_change(results, args.output)
    plot_sv_spectra(results, args.output)

    print("\nAll SVD analysis plots generated.")


if __name__ == "__main__":
    main()

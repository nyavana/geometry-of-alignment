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

    The canonical signal-strength metric (M2b 4.5) is a Cohen's-d-style
    separation along the difference-in-means direction:

        d_dir = (mean_r - mean_c) / ||mean_r - mean_c||
        signal = ||mean_r - mean_c||
                  / sqrt( (std(refuse_proj)^2 + std(comply_proj)^2) / 2 )

    where ``refuse_proj``/``comply_proj`` are the per-row projections of the
    activations onto ``d_dir``. This is exactly equivalent to univariate
    Cohen's d of those projections, since the numerator
    ``mean(refuse_proj) - mean(comply_proj) == ||mean_r - mean_c||`` by the
    choice of ``d_dir``.
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

        direction = diff / (diff.norm() + 1e-12)
        refuse_proj = (refuse_acts[layer_idx] @ direction).numpy()
        comply_proj = (comply_acts[layer_idx] @ direction).numpy()

        # Legacy "separation score": z-score against pooled-everything std.
        all_proj = np.concatenate([refuse_proj, comply_proj])
        separation = (refuse_proj.mean() - comply_proj.mean()) / (np.std(all_proj) + 1e-8)

        # Canonical Cohen's-d-style signal strength along the diff-in-means
        # direction (M2b 4.5 spec).
        std_r = float(np.std(refuse_proj, ddof=0))
        std_c = float(np.std(comply_proj, ddof=0))
        pooled = float(np.sqrt((std_r ** 2 + std_c ** 2) / 2.0))
        cohens_d = signal_norm / (pooled + 1e-12)

        results[layer_idx] = {
            "signal_norm": float(signal_norm),
            "cosine_similarity": float(cosine_sim),
            "separation_score": float(separation),
            "cohens_d": float(cohens_d),
            "std_refuse_proj": float(std_r),
            "std_comply_proj": float(std_c),
            "layer_type": get_layer_type(layer_idx),
        }

    return results


def analyze_refusal_rank(refuse_acts: dict, comply_acts: dict,
                         n_components: int = 20) -> dict:
    """
    PCA rank analysis at each layer.

    For each layer:
      * Stack `refuse + comply` rows, center, run PCA, retain ``n_components``.
      * Record per-component explained-variance ratios and cumulative curve.
      * Compute the effective rank at 95%, 99%, 99.9% thresholds.
      * Test the "rank-1 refusal" hypothesis quantitatively: project the
        diff-in-means vector ``Δμ = mean_r - mean_c`` onto each principal
        component and report the squared cosine ``cos²(PC_i, Δμ)`` — i.e.,
        the fraction of ``||Δμ||²`` captured by each component. The "top-1
        diff capture" is ``cos²(PC_1, Δμ)`` and is the canonical rank-1
        figure-of-merit (M2b 4.6 / 4.9).
    """
    rank_results = {}

    for layer_idx in sorted(refuse_acts.keys()):
        refuse = refuse_acts[layer_idx]
        comply = comply_acts[layer_idx]
        all_acts = torch.cat([refuse, comply], dim=0)
        centered = all_acts - all_acts.mean(dim=0)

        n_comp = min(n_components, centered.shape[0] - 1, centered.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(centered.numpy())

        cumulative = np.cumsum(pca.explained_variance_ratio_)

        # Diff-in-means capture per component: cos^2 between Δμ and each PC.
        diff = (refuse.mean(dim=0) - comply.mean(dim=0)).numpy()
        diff_norm_sq = float(np.dot(diff, diff))
        components = pca.components_  # shape (n_comp, hidden_dim)
        if diff_norm_sq > 0:
            diff_unit = diff / np.sqrt(diff_norm_sq)
            cos_sq = (components @ diff_unit) ** 2
        else:
            cos_sq = np.zeros(components.shape[0])

        rank_results[layer_idx] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumulative.tolist(),
            "rank_95": int((cumulative < 0.95).sum()) + 1,
            "rank_99": int((cumulative < 0.99).sum()) + 1,
            "rank_999": int((cumulative < 0.999).sum()) + 1,
            "top1_energy": float(pca.explained_variance_ratio_[0]),
            "diff_capture_per_pc": [float(x) for x in cos_sq.tolist()],
            "diff_capture_top1": float(cos_sq[0]),
            "diff_capture_top3": float(cos_sq[:3].sum()),
            "diff_norm": float(np.sqrt(diff_norm_sq)),
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


def plot_signal_vs_layer(results: dict, output_dir: str = "results/figures"):
    """
    Canonical M2b 4.5 figure: Cohen's-d-style signal strength per layer with
    sliding vs global attention layers marked distinctly. Saved as
    ``signal_vs_layer.png``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layers = sorted(results.keys())
    cohens = [results[l]["cohens_d"] for l in layers]

    sliding_x = [l for l in layers if results[l]["layer_type"] == "sliding"]
    sliding_y = [results[l]["cohens_d"] for l in sliding_x]
    global_x = [l for l in layers if results[l]["layer_type"] == "global"]
    global_y = [results[l]["cohens_d"] for l in global_x]

    fig, ax = plt.subplots(figsize=(14, 5))
    # Connecting line so the trajectory across layers is legible.
    ax.plot(layers, cohens, color="#888888", linewidth=1.0, alpha=0.6,
            zorder=1, label="_nolegend_")
    ax.scatter(sliding_x, sliding_y, color="#2196F3", s=55, marker="o",
               edgecolor="black", linewidth=0.4, zorder=2,
               label="Sliding attention (35 layers)")
    ax.scatter(global_x, global_y, color="#FF5722", s=110, marker="^",
               edgecolor="black", linewidth=0.6, zorder=3,
               label="Global attention (7 layers)")

    peak_layer = max(layers, key=lambda l: results[l]["cohens_d"])
    ax.annotate(
        f"peak: layer {peak_layer}\nd={results[peak_layer]['cohens_d']:.2f}",
        xy=(peak_layer, results[peak_layer]["cohens_d"]),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Signal strength (Cohen's $d$)")
    ax.set_title(
        "Refusal-vs-comply signal per layer — Gemma 4 E4B (8-bit)\n"
        r"$d = \|\mu_R - \mu_C\| \,/\, \sqrt{(\sigma_R^2 + \sigma_C^2)/2}$ "
        "along the diff-in-means direction"
    )
    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/signal_vs_layer.png", dpi=150)
    print(f"Saved to {output_dir}/signal_vs_layer.png")
    plt.close()


def plot_pca_variance_per_layer(rank_results: dict,
                                output_dir: str = "results/figures",
                                k_components: int = 10):
    """
    Canonical M2b 4.6 figure: PCA variance behaviour per layer. Saved as
    ``pca_variance_per_layer.png``.

    Three panels:
      (top) Cumulative variance vs component index, one line per layer,
            color-graded by depth. 95%/99% reference lines.
      (mid) Effective rank to capture 95% / 99% variance vs layer index;
            global-attention layers marked.
      (bot) Diff-in-means capture: cos²(PC_1, Δμ) vs layer index — the
            quantitative "rank-1 refusal" check.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layers = sorted(rank_results.keys())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: cumulative variance curves, one per layer.
    ax = axes[0]
    for l in layers:
        cum = rank_results[l]["cumulative_variance"][:k_components]
        x = np.arange(1, len(cum) + 1)
        color = cmap(l / max(layers))
        ax.plot(x, cum, color=color, alpha=0.55, linewidth=1.2)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
    ax.axhline(0.99, color="darkred", linestyle=":", alpha=0.5, label="99% threshold")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(
        f"PCA cumulative variance, top-{k_components} PCs — color graded by layer "
        "index (dark=early, bright=late)"
    )
    ax.set_xticks(np.arange(1, k_components + 1))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(layers)))
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("Layer index")

    # Panel 2: rank-95 / rank-99 vs layer index.
    ax = axes[1]
    rank95 = [rank_results[l]["rank_95"] for l in layers]
    rank99 = [rank_results[l]["rank_99"] for l in layers]
    ax.plot(layers, rank95, marker="o", color="#2196F3", label="Rank to 95%",
            linewidth=1.5)
    ax.plot(layers, rank99, marker="s", color="#0d47a1", label="Rank to 99%",
            linewidth=1.5)
    for gl in [l for l in layers if get_layer_type(l) == "global"]:
        ax.axvline(gl, color="#FF5722", alpha=0.25, linewidth=1.0, linestyle=":")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank vs layer (orange dashed lines: global-attention layers)")
    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 3: diff-in-means capture by top-1 PC vs layer index.
    ax = axes[2]
    top1_capture = [rank_results[l]["diff_capture_top1"] for l in layers]
    top3_capture = [rank_results[l]["diff_capture_top3"] for l in layers]
    sliding_x = [l for l in layers if get_layer_type(l) == "sliding"]
    sliding_y = [rank_results[l]["diff_capture_top1"] for l in sliding_x]
    global_x = [l for l in layers if get_layer_type(l) == "global"]
    global_y = [rank_results[l]["diff_capture_top1"] for l in global_x]
    ax.plot(layers, top3_capture, color="#888888", linewidth=1.0, alpha=0.5,
            label=r"Top-3 PC capture of $\|\Delta\mu\|^2$")
    ax.plot(layers, top1_capture, color="#444444", linewidth=1.0, alpha=0.4,
            zorder=1)
    ax.scatter(sliding_x, sliding_y, color="#2196F3", s=55, marker="o",
               edgecolor="black", linewidth=0.4, zorder=2,
               label="Sliding (top-1 capture)")
    ax.scatter(global_x, global_y, color="#FF5722", s=110, marker="^",
               edgecolor="black", linewidth=0.6, zorder=3,
               label="Global (top-1 capture)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Squared cosine $\cos^2(\mathrm{PC}_k, \Delta\mu)$")
    ax.set_title(
        r"Rank-1 hypothesis: fraction of $\|\Delta\mu\|^2$ captured by leading PCs"
    )
    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_variance_per_layer.png", dpi=150)
    print(f"Saved to {output_dir}/pca_variance_per_layer.png")
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
    plot_signal_vs_layer(signal_results, args.output)

    # Print summary
    sliding_sep = [v["separation_score"] for v in signal_results.values()
                   if v["layer_type"] == "sliding"]
    global_sep = [v["separation_score"] for v in signal_results.values()
                  if v["layer_type"] == "global"]
    sliding_d = [v["cohens_d"] for v in signal_results.values()
                 if v["layer_type"] == "sliding"]
    global_d = [v["cohens_d"] for v in signal_results.values()
                if v["layer_type"] == "global"]
    layers_sorted = sorted(signal_results.keys())
    top3 = sorted(layers_sorted, key=lambda l: signal_results[l]["cohens_d"],
                  reverse=True)[:3]
    print(f"\nMean separation — sliding: {np.mean(sliding_sep):.3f}, "
          f"global: {np.mean(global_sep):.3f}")
    print(f"Mean Cohen's d — sliding: {np.mean(sliding_d):.3f}, "
          f"global: {np.mean(global_d):.3f}, "
          f"gap: {np.mean(global_d) - np.mean(sliding_d):+.3f}")
    print("Top-3 layers by Cohen's d: "
          + ", ".join(f"L{l}={signal_results[l]['cohens_d']:.3f}"
                      f"({signal_results[l]['layer_type']})" for l in top3))

    # Rank analysis
    print("\nComputing rank analysis...")
    rank_results = analyze_refusal_rank(refuse_acts, comply_acts)
    plot_rank_analysis(rank_results, args.output)
    plot_pca_variance_per_layer(rank_results, args.output)

    # Rank-1 hypothesis summary at the peak layers (by Cohen's d).
    peak_layers = top3
    top1_caps = [rank_results[l]["diff_capture_top1"] for l in peak_layers]
    top3_caps = [rank_results[l]["diff_capture_top3"] for l in peak_layers]
    print(
        f"\nRank-1 capture at peak layers {peak_layers}: "
        + ", ".join(f"L{l} top1={rank_results[l]['diff_capture_top1']:.3f} "
                    f"top3={rank_results[l]['diff_capture_top3']:.3f}"
                    for l in peak_layers)
    )
    print(f"Mean top-1 diff capture over peak layers: {np.mean(top1_caps):.3f}")
    print(f"Mean top-3 diff capture over peak layers: {np.mean(top3_caps):.3f}")
    rank95_at_peak = [rank_results[l]["rank_95"] for l in peak_layers]
    rank99_at_peak = [rank_results[l]["rank_99"] for l in peak_layers]
    print(f"rank_95 at peak layers: {rank95_at_peak}")
    print(f"rank_99 at peak layers: {rank99_at_peak}")

    # Save numerical results
    results_path = Path(args.activations) / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump({"signal_strength": signal_results, "rank_analysis": rank_results},
                  f, indent=2)
    print(f"\nSaved analysis to {results_path}")


if __name__ == "__main__":
    main()

"""
MoE-specific analysis for Qwen3.5-35B-A3B weight diffs.

Analyzes which experts, routers, and shared experts were modified
to determine whether safety is encoded in routing vs expert weights.

Qwen3.5-35B-A3B: 40 layers, 256 experts, 8 routed + 1 shared per token

Usage:
    python -m src.weight_diff.moe_expert_analysis \
        --results results/weight_diffs/qwen/weight_diff_results.json \
        --output results/figures/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def analyze_expert_modifications(results_path: str, output_dir: str):
    """Analyze which experts were modified in the cracked model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    expert_changes = defaultdict(lambda: defaultdict(float))
    router_changes = defaultdict(float)
    shared_expert_changes = defaultdict(float)
    attention_changes = defaultdict(float)
    mlp_changes = defaultdict(float)

    for r in results:
        if r["frobenius_norm"] <= 1e-6:
            continue

        key = r["key"]
        parts = key.split(".")

        # Extract layer index
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
                break

        if layer_idx is None:
            continue

        if "experts" in key:
            try:
                expert_idx = int(parts[parts.index("experts") + 1])
                expert_changes[layer_idx][expert_idx] += r["frobenius_norm"]
            except (ValueError, IndexError):
                pass
        elif "gate" in key or "router" in key:
            router_changes[layer_idx] += r["frobenius_norm"]
        elif "shared_expert" in key:
            shared_expert_changes[layer_idx] += r["frobenius_norm"]
        elif "self_attn" in key:
            attention_changes[layer_idx] += r["frobenius_norm"]
        elif "mlp" in key:
            mlp_changes[layer_idx] += r["frobenius_norm"]

    # ── Print Summary ──
    print("\n" + "=" * 60)
    print("MoE CRACKING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Layers with expert modifications:        {len(expert_changes)}")
    print(f"Layers with router modifications:        {len(router_changes)}")
    print(f"Layers with shared expert modifications: {len(shared_expert_changes)}")
    print(f"Layers with attention modifications:     {len(attention_changes)}")
    print(f"Layers with MLP modifications:           {len(mlp_changes)}")

    if expert_changes:
        all_counts = [len(expert_changes[l]) for l in expert_changes]
        print(f"\nExperts modified per layer: {np.mean(all_counts):.1f} "
              f"+/- {np.std(all_counts):.1f} (out of 256)")

    if router_changes:
        print(f"\nRouter was modified: YES")
        print(f"  -> Safety may be partly encoded in ROUTING")
    else:
        print(f"\nRouter was modified: NO")
        print(f"  -> Safety is encoded in expert weights, not routing")

    # ── Expert modification heatmap ──
    if expert_changes:
        layers = sorted(expert_changes.keys())
        max_expert = max(max(ec.keys()) for ec in expert_changes.values()) + 1
        max_expert = min(max_expert, 256)

        heatmap = np.zeros((len(layers), max_expert))
        for i, l in enumerate(layers):
            for e, change in expert_changes[l].items():
                if e < max_expert:
                    heatmap[i, e] = change

        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(heatmap, aspect="auto", cmap="hot")
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_title("Expert Weight Modifications in Cracked Model (Qwen3.5-A3B)")
        plt.colorbar(im, ax=ax, label="||ΔW||_F")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/expert_modification_heatmap.png", dpi=150)
        print(f"\nSaved {output_dir}/expert_modification_heatmap.png")
        plt.close()

    # ── Component type bar chart ──
    component_totals = {
        "attention": sum(attention_changes.values()),
        "mlp": sum(mlp_changes.values()),
        "experts": sum(sum(ec.values()) for ec in expert_changes.values()),
        "router": sum(router_changes.values()),
        "shared_expert": sum(shared_expert_changes.values()),
    }
    component_totals = {k: v for k, v in component_totals.items() if v > 0}

    if component_totals:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(component_totals.keys(), component_totals.values())
        ax.set_ylabel("Total ||ΔW||_F")
        ax.set_title("Modification Magnitude by Component Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/component_type_changes.png", dpi=150)
        print(f"Saved {output_dir}/component_type_changes.png")
        plt.close()

    # Save numerical results
    summary = {
        "expert_layers_modified": len(expert_changes),
        "router_modified": len(router_changes) > 0,
        "shared_expert_modified": len(shared_expert_changes) > 0,
        "component_totals": component_totals,
    }
    with open(f"{output_dir}/moe_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MoE expert analysis")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="results/figures/")
    args = parser.parse_args()

    analyze_expert_modifications(args.results, args.output)


if __name__ == "__main__":
    main()

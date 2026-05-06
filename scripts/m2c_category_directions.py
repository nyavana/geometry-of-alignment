"""M2c task 5.9 — category-specific refusal directions (CPU-only).

Slices M2b's `refuse_activations.pt` by category metadata, computes a
per-category "(category - safe_control)" mean-difference direction at
each layer, and emits pairwise cosine similarity tables/heatmaps.

If pairwise cosine ≈ 1 across layers → refusal is monolithic and
selective abliteration is unlikely to separate over-refusal categories
from genuinely-harmful ones. If cosine drops at some layers (e.g., the
peak signal band L4–L17 from M2b 4.5) → those layers are candidates for
selective abliteration in 5.10.

Outputs:
    $RESULTS_DIR/activations/category_directions.pt
    $RESULTS_DIR/activations/category_cosine_summary.json
    $RESULTS_DIR/figures/category_cosine_heatmap.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def slice_by_category(refuse_acts: dict[int, torch.Tensor],
                      refuse_meta: list[dict],
                      categories: list[str]) -> dict[str, dict[int, torch.Tensor]]:
    """For each requested category, return the per-layer slice of the refuse
    activation tensor whose rows correspond to that category in the metadata."""
    by_cat = {}
    for cat in categories:
        rows = [i for i, m in enumerate(refuse_meta) if m["category"] == cat]
        if not rows:
            continue
        idx = torch.tensor(rows, dtype=torch.long)
        by_cat[cat] = {layer: t.index_select(0, idx) for layer, t in refuse_acts.items()}
    return by_cat


def category_directions(category_acts: dict[str, dict[int, torch.Tensor]],
                        comply_acts: dict[int, torch.Tensor]) -> dict[str, dict[int, torch.Tensor]]:
    """Per-category, per-layer (mean_category - mean_safe_control) unit vector."""
    out = {}
    for cat, layer_to_acts in category_acts.items():
        per_layer = {}
        for layer, acts in layer_to_acts.items():
            mean_cat = acts.mean(dim=0)
            mean_base = comply_acts[layer].mean(dim=0)
            d = mean_cat - mean_base
            d = d / d.norm()
            per_layer[layer] = d.unsqueeze(0)  # (1, hidden_dim)
        out[cat] = per_layer
    return out


def pairwise_cosine(per_cat_dirs: dict[str, dict[int, torch.Tensor]]) -> dict:
    """Pairwise cosine similarity per layer + summary statistics."""
    cats = list(per_cat_dirs.keys())
    layers = sorted(per_cat_dirs[cats[0]].keys())
    summary = {}
    for i, ca in enumerate(cats):
        for cb in cats[i + 1:]:
            pair = f"{ca}_vs_{cb}"
            per_layer = {}
            for layer in layers:
                da = per_cat_dirs[ca][layer].squeeze()
                db = per_cat_dirs[cb][layer].squeeze()
                per_layer[layer] = float(torch.nn.functional.cosine_similarity(
                    da.unsqueeze(0), db.unsqueeze(0)
                ).item())
            vals = list(per_layer.values())
            summary[pair] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "peak_layer_4_17_mean": float(np.mean(
                    [per_layer[l] for l in layers if 4 <= l <= 17]
                )),
                "per_layer": per_layer,
            }
    return summary


def plot_heatmap(summary: dict, output_path: Path) -> None:
    """Heatmap rows = category pairs, columns = layers, cells = cosine."""
    import matplotlib.pyplot as plt
    pairs = list(summary.keys())
    layers = sorted(next(iter(summary.values()))["per_layer"].keys())
    matrix = np.array([
        [summary[p]["per_layer"][l] for l in layers] for p in pairs
    ])
    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(pairs))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], rotation=90, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title("Pairwise cosine similarity between category-specific refusal directions\n"
                 "(refuse-class category mean - safe_control mean, per layer)")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="M2c 5.9 category-specific refusal directions")
    parser.add_argument("--m2b-activations",
                        default="/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations")
    parser.add_argument("--output-activations",
                        default=None,
                        help="Output directory for category_directions.pt and summary JSON. "
                             "Default: $RESULTS_DIR/activations/")
    parser.add_argument("--output-figures",
                        default=None,
                        help="Output directory for the heatmap. Default: $RESULTS_DIR/figures/")
    parser.add_argument("--categories",
                        default="emergency_medical,wilderness_survival,should_refuse,home_safety,chemistry_safety,mental_health")
    args = parser.parse_args()

    import os
    rd = os.environ.get("RESULTS_DIR")
    out_act = Path(args.output_activations) if args.output_activations else Path(rd) / "activations"
    out_fig = Path(args.output_figures) if args.output_figures else Path(rd) / "figures"
    out_act.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    src = Path(args.m2b_activations)
    refuse_acts = torch.load(src / "refuse_activations.pt", weights_only=True)
    comply_acts = torch.load(src / "comply_activations.pt", weights_only=True)
    with open(src / "prompt_metadata.json") as f:
        meta = json.load(f)
    refuse_meta = meta["refuse"]

    categories = args.categories.split(",")
    print(f"Categories: {categories}")
    print(f"refuse meta rows: {len(refuse_meta)}; refuse_acts layers: {len(refuse_acts)}")

    cat_acts = slice_by_category(refuse_acts, refuse_meta, categories)
    print(f"Per-category prompt counts: "
          f"{ {c: cat_acts[c][0].shape[0] for c in cat_acts} }")

    cat_dirs = category_directions(cat_acts, comply_acts)
    summary = pairwise_cosine(cat_dirs)

    # Save category directions tensor (for 5.10 selective abliteration consumer).
    torch.save(cat_dirs, out_act / "category_directions.pt")

    # Save summary JSON (without full per-layer dict, save flattened too).
    with open(out_act / "category_cosine_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap(summary, out_fig / "category_cosine_heatmap.png")

    print(f"\nSummary (mean cosine across all 42 layers):")
    for pair, stats in summary.items():
        print(f"  {pair:60s} mean={stats['mean']:+.3f} (peak L4-17={stats['peak_layer_4_17_mean']:+.3f}) "
              f"min={stats['min']:+.3f} max={stats['max']:+.3f}")

    # Headline finding: medical_vs_should_refuse is the key pair for selective safety.
    key_pair = "emergency_medical_vs_should_refuse"
    if key_pair in summary:
        s = summary[key_pair]
        print(f"\nHEADLINE: {key_pair} cosine: mean={s['mean']:+.3f}, "
              f"peak-band-mean(L4-17)={s['peak_layer_4_17_mean']:+.3f}")
        if s["peak_layer_4_17_mean"] < 0.7:
            print("  -> Below 0.7 in peak band: selective abliteration MAY work.")
        else:
            print("  -> >=0.7 in peak band: refusal direction is largely shared "
                  "across emergency_medical and should_refuse; selective abliteration "
                  "unlikely to cleanly separate them.")

    print(f"\nWrote:\n"
          f"  {out_act / 'category_directions.pt'}\n"
          f"  {out_act / 'category_cosine_summary.json'}\n"
          f"  {out_fig / 'category_cosine_heatmap.png'}")


if __name__ == "__main__":
    main()

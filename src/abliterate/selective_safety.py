"""
Selective safety: remove over-refusal on specific categories
while preserving refusal on genuinely harmful queries.

Computes category-specific refusal directions and tests whether
they can be independently removed.

Usage:
    python -m src.abliterate.selective_safety \
        --model google/gemma-4-E4B-it \
        --benchmark data/benchmark_prompts.json \
        --use-8bit
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.mechanistic.extract_activations import (
    load_model_and_tokenizer, ActivationCollector, extract_activations_for_prompts,
)
from src.abliterate.abliterate import abliterate_model
from src.abliterate.ablation_study import quick_evaluate


def compute_category_directions(model, tokenizer, collector: ActivationCollector,
                                benchmark: dict, categories: list[str],
                                baseline_category: str = "safe_control") -> dict:
    """
    Compute separate refusal directions for different categories.

    Each category's direction = mean(category_acts) - mean(baseline_acts)

    Returns:
        dict {category: {layer_idx: direction_tensor (1, hidden_dim)}}
    """
    # Get baseline (safe control) activations
    baseline_prompts = [p["prompt"] for p in benchmark["prompts"]
                        if p["category"] == baseline_category]
    print(f"\nExtracting baseline activations ({baseline_category}, "
          f"{len(baseline_prompts)} prompts)...")
    baseline_acts = extract_activations_for_prompts(
        model, tokenizer, baseline_prompts, collector
    )

    category_directions = {}

    for category in categories:
        cat_prompts = [p["prompt"] for p in benchmark["prompts"]
                       if p["category"] == category]
        if not cat_prompts:
            print(f"  Warning: no prompts for category '{category}', skipping")
            continue

        print(f"Extracting activations for '{category}' ({len(cat_prompts)} prompts)...")
        cat_acts = extract_activations_for_prompts(
            model, tokenizer, cat_prompts, collector
        )

        directions = {}
        for layer_idx in cat_acts:
            mean_cat = cat_acts[layer_idx].mean(dim=0)
            mean_base = baseline_acts[layer_idx].mean(dim=0)
            d = mean_cat - mean_base
            d = d / d.norm()
            directions[layer_idx] = d.unsqueeze(0)

        category_directions[category] = directions

    return category_directions


def analyze_direction_similarity(category_directions: dict) -> dict:
    """
    Compute pairwise cosine similarity between category refusal directions.

    If similarity is low -> selective abliteration is feasible.
    If similarity is high -> refusal is monolithic, hard to separate.
    """
    categories = list(category_directions.keys())
    layers = sorted(category_directions[categories[0]].keys())

    results = {}

    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            pair_name = f"{cat_a}_vs_{cat_b}"
            sims = []
            for layer_idx in layers:
                d_a = category_directions[cat_a][layer_idx].squeeze()
                d_b = category_directions[cat_b][layer_idx].squeeze()
                sim = torch.nn.functional.cosine_similarity(
                    d_a.unsqueeze(0), d_b.unsqueeze(0)
                ).item()
                sims.append(sim)

            results[pair_name] = {
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "per_layer": {l: s for l, s in zip(layers, sims)},
            }
            print(f"  {pair_name}: mean cosine similarity = "
                  f"{np.mean(sims):.4f} +/- {np.std(sims):.4f}")

    return results


def run_selective_abliteration(model_name: str, category_directions: dict,
                               tokenizer, benchmark_path: str,
                               use_8bit: bool,
                               remove_categories: list[str],
                               keep_categories: list[str]) -> dict:
    """
    Remove refusal directions for specific categories while keeping others.

    Args:
        remove_categories: categories whose refusal direction will be projected out
        keep_categories: categories where we expect refusal to be preserved
    """
    print(f"\nSelective abliteration: removing {remove_categories}, keeping {keep_categories}")

    model, _ = load_model_and_tokenizer(model_name, use_8bit)

    # Only project out directions for the categories we want to de-censor
    for cat in remove_categories:
        if cat in category_directions:
            abliterate_model(model, category_directions[cat], alpha=1.0)

    scores = quick_evaluate(model, tokenizer, benchmark_path)

    del model
    torch.cuda.empty_cache()

    return scores


def main():
    parser = argparse.ArgumentParser(description="Selective safety experiments")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--benchmark", default="data/benchmark_prompts.json")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--output", default="results/ablation_results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    model, tokenizer = load_model_and_tokenizer(args.model, args.use_8bit)

    collector = ActivationCollector(model)
    collector.register_hooks()

    # Compute category-specific directions
    categories = ["emergency_medical", "wilderness_survival", "should_refuse"]
    cat_dirs = compute_category_directions(
        model, tokenizer, collector, benchmark, categories
    )

    collector.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Analyze similarity
    print("\n" + "=" * 50)
    print("DIRECTION SIMILARITY ANALYSIS")
    print("=" * 50)
    similarity = analyze_direction_similarity(cat_dirs)

    # Run selective abliteration experiments
    print("\n" + "=" * 50)
    print("SELECTIVE ABLITERATION")
    print("=" * 50)

    experiments = [
        {"name": "remove_medical_only",
         "remove": ["emergency_medical"],
         "keep": ["should_refuse"]},
        {"name": "remove_survival_only",
         "remove": ["wilderness_survival"],
         "keep": ["should_refuse"]},
        {"name": "remove_medical_and_survival",
         "remove": ["emergency_medical", "wilderness_survival"],
         "keep": ["should_refuse"]},
    ]

    experiment_results = []
    for exp in experiments:
        print(f"\n--- {exp['name']} ---")
        scores = run_selective_abliteration(
            args.model, cat_dirs, tokenizer, args.benchmark, args.use_8bit,
            exp["remove"], exp["keep"]
        )
        experiment_results.append({"experiment": exp["name"], **scores})
        print(f"  Refusal rate: {scores['refusal_rate']:.1%}, "
              f"Over-refusal: {scores['over_refusal_rate']:.1%}")

    # Save results
    results = {
        "direction_similarity": similarity,
        "selective_experiments": experiment_results,
    }
    with open(output_dir / "selective_safety_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'selective_safety_results.json'}")


if __name__ == "__main__":
    main()

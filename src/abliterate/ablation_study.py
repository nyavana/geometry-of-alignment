"""
Systematic ablation experiments across multiple axes:
- Alpha (projection strength)
- Layer subsets (which layers to modify)
- Prompt count (how many examples needed to compute direction)
- Direction method (mean-diff vs PCA)
- Control (random direction)

Usage:
    python -m src.abliterate.ablation_study \
        --model google/gemma-4-E4B-it \
        --activations results/activations/ \
        --benchmark data/benchmark_prompts.json \
        --output results/ablation_results/
"""

import torch
import json
import argparse
from pathlib import Path
from copy import deepcopy

from src.mechanistic.extract_activations import load_model_and_tokenizer
from src.abliterate.abliterate import abliterate_model, compute_refusal_direction
from src.benchmark.evaluate import evaluate_with_transformers


# Gemma 4 E4B layer subsets — verify these indices
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
SLIDING_LAYERS = [i for i in range(42) if i not in GLOBAL_LAYERS]

LAYER_CONFIGS = {
    "all_42":      list(range(42)),
    "global_only": GLOBAL_LAYERS,
    "sliding_only": SLIDING_LAYERS,
    "first_half":  list(range(21)),
    "second_half": list(range(21, 42)),
    "last_10":     list(range(32, 42)),
    "middle_14":   list(range(14, 28)),
}


def quick_evaluate(model, tokenizer, benchmark_path: str,
                   max_prompts: int = 50) -> dict:
    """
    Quick evaluation on a subset of the benchmark.
    Returns refusal rate and over-refusal rate.
    """
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    prompts = benchmark["prompts"][:max_prompts]

    from src.benchmark.classify_refusal import classify_refusal

    refused = 0
    over_refused = 0
    total = 0

    for p in prompts:
        messages = [{"role": "user", "content": p["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=256, temperature=0.1, do_sample=True,
            )

        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        is_refusal = classify_refusal(response)
        total += 1
        if is_refusal:
            refused += 1
        if p["expected"] == "comply" and is_refusal:
            over_refused += 1

    return {
        "total": total,
        "refused": refused,
        "refusal_rate": refused / total if total > 0 else 0,
        "over_refused": over_refused,
        "over_refusal_rate": over_refused / total if total > 0 else 0,
    }


def run_alpha_sweep(model_name: str, directions: dict,
                    tokenizer, benchmark_path: str, use_8bit: bool,
                    output_dir: Path) -> list[dict]:
    """Sweep alpha from 0 to 2.0."""
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    results = []

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        model, _ = load_model_and_tokenizer(model_name, use_8bit)
        abliterate_model(model, directions, alpha=alpha)
        scores = quick_evaluate(model, tokenizer, benchmark_path)
        results.append({"axis": "alpha", "alpha": alpha, "layers": "all", **scores})
        print(f"  Refusal rate: {scores['refusal_rate']:.1%}, "
              f"Over-refusal: {scores['over_refusal_rate']:.1%}")
        del model
        torch.cuda.empty_cache()

    return results


def run_layer_sweep(model_name: str, directions: dict,
                    tokenizer, benchmark_path: str, use_8bit: bool,
                    output_dir: Path) -> list[dict]:
    """Sweep which layers to abliterate."""
    results = []

    for config_name, layers in LAYER_CONFIGS.items():
        print(f"\n--- Layers: {config_name} ({len(layers)} layers) ---")
        model, _ = load_model_and_tokenizer(model_name, use_8bit)
        abliterate_model(model, directions, layers_to_modify=layers, alpha=1.0)
        scores = quick_evaluate(model, tokenizer, benchmark_path)
        results.append({"axis": "layers", "alpha": 1.0,
                        "layers": config_name, "layer_count": len(layers), **scores})
        print(f"  Refusal rate: {scores['refusal_rate']:.1%}")
        del model
        torch.cuda.empty_cache()

    return results


def run_random_control(model_name: str, tokenizer, benchmark_path: str,
                       use_8bit: bool, hidden_dim: int = 2560) -> dict:
    """Control: abliterate with random directions (should NOT work)."""
    print("\n--- CONTROL: Random direction ---")
    model, _ = load_model_and_tokenizer(model_name, use_8bit)

    random_dirs = {}
    for i in range(42):
        d = torch.randn(1, hidden_dim)
        d = d / d.norm()
        random_dirs[i] = d

    abliterate_model(model, random_dirs, alpha=1.0)
    scores = quick_evaluate(model, tokenizer, benchmark_path)
    print(f"  Refusal rate: {scores['refusal_rate']:.1%} (should be ~same as original)")
    del model
    torch.cuda.empty_cache()

    return {"axis": "control_random", "alpha": 1.0, **scores}


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--activations", default="results/activations/")
    parser.add_argument("--benchmark", default="data/benchmark_prompts.json")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--output", default="results/ablation_results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load directions
    directions = torch.load(
        Path(args.activations) / "refusal_directions.pt", weights_only=True
    )
    for k in directions:
        if directions[k].dim() == 1:
            directions[k] = directions[k].unsqueeze(0)

    # Load tokenizer once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    all_results = []

    # Alpha sweep
    print("\n" + "=" * 50)
    print("ALPHA SWEEP")
    print("=" * 50)
    all_results.extend(
        run_alpha_sweep(args.model, directions, tokenizer,
                        args.benchmark, args.use_8bit, output_dir)
    )

    # Layer sweep
    print("\n" + "=" * 50)
    print("LAYER SWEEP")
    print("=" * 50)
    all_results.extend(
        run_layer_sweep(args.model, directions, tokenizer,
                        args.benchmark, args.use_8bit, output_dir)
    )

    # Random control
    print("\n" + "=" * 50)
    print("RANDOM CONTROL")
    print("=" * 50)
    all_results.append(
        run_random_control(args.model, tokenizer, args.benchmark, args.use_8bit)
    )

    # Save all results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()

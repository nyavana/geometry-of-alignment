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


def _get_layers(model):
    """Return the iterable of transformer layers, handling Gemma 4 multimodal."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def snapshot_target_weights(model) -> dict:
    """Clone .data of the o_proj + down_proj matrices abliterate mutates.

    Same device as the live weights — sufficient for restoring between
    sweep iterations without disk reloads. abliterate_model() writes to
    weight.data via _project_out, so cloning .data is the right granularity
    regardless of 8-bit Int8Params wrappers.
    """
    snap = {}
    for i, layer in enumerate(_get_layers(model)):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            snap[(i, "o_proj")] = layer.self_attn.o_proj.weight.data.detach().clone()
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            snap[(i, "down_proj")] = layer.mlp.down_proj.weight.data.detach().clone()
    return snap


def restore_target_weights(model, snap: dict) -> None:
    """Restore o_proj + down_proj from a snapshot taken before abliteration."""
    for i, layer in enumerate(_get_layers(model)):
        if (i, "o_proj") in snap:
            layer.self_attn.o_proj.weight.data = snap[(i, "o_proj")].clone()
        if (i, "down_proj") in snap:
            layer.mlp.down_proj.weight.data = snap[(i, "down_proj")].clone()


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
    # Peak signal band per M2b 4.5: highest Cohen's d at L15, high-signal band L4-L17.
    "peak_band_4_17": list(range(4, 18)),
    "peak_layer_15_only": [15],
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
        # transformers v5: apply_chat_template returns a BatchEncoding —
        # explicitly extract the input_ids tensor.
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=256, temperature=0.1, do_sample=True,
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


def run_alpha_sweep(model, tokenizer, directions: dict, snapshot: dict,
                    benchmark_path: str) -> list[dict]:
    """Sweep alpha from 0 to 2.0. Restores model weights from snapshot
    before each iteration so the model object is reused across sweeps."""
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    results = []

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        restore_target_weights(model, snapshot)
        abliterate_model(model, directions, alpha=alpha)
        scores = quick_evaluate(model, tokenizer, benchmark_path)
        results.append({"axis": "alpha", "alpha": alpha, "layers": "all", **scores})
        print(f"  Refusal rate: {scores['refusal_rate']:.1%}, "
              f"Over-refusal: {scores['over_refusal_rate']:.1%}")
        torch.cuda.empty_cache()

    return results


def run_layer_sweep(model, tokenizer, directions: dict, snapshot: dict,
                    benchmark_path: str) -> list[dict]:
    """Sweep which layers to abliterate."""
    results = []

    for config_name, layers in LAYER_CONFIGS.items():
        print(f"\n--- Layers: {config_name} ({len(layers)} layers) ---")
        restore_target_weights(model, snapshot)
        abliterate_model(model, directions, layers_to_modify=layers, alpha=1.0)
        scores = quick_evaluate(model, tokenizer, benchmark_path)
        results.append({"axis": "layers", "alpha": 1.0,
                        "layers": config_name, "layer_count": len(layers), **scores})
        print(f"  Refusal rate: {scores['refusal_rate']:.1%}")
        torch.cuda.empty_cache()

    return results


def run_random_control(model, tokenizer, snapshot: dict,
                       benchmark_path: str, hidden_dim: int = 2560) -> dict:
    """Control: abliterate with random directions (should NOT work)."""
    print("\n--- CONTROL: Random direction ---")
    restore_target_weights(model, snapshot)

    random_dirs = {}
    for i in range(42):
        d = torch.randn(1, hidden_dim)
        d = d / d.norm()
        random_dirs[i] = d

    abliterate_model(model, random_dirs, alpha=1.0)
    scores = quick_evaluate(model, tokenizer, benchmark_path)
    print(f"  Refusal rate: {scores['refusal_rate']:.1%} (should be ~same as original)")
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

    # Single load — model + tokenizer are reused across all sweeps via the
    # snapshot/restore mechanism, avoiding O(N) reloads from disk.
    print("\nLoading model and tokenizer once for the full sweep ensemble...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.use_8bit)
    snapshot = snapshot_target_weights(model)

    all_results = []

    # Alpha sweep
    print("\n" + "=" * 50)
    print("ALPHA SWEEP")
    print("=" * 50)
    all_results.extend(
        run_alpha_sweep(model, tokenizer, directions, snapshot, args.benchmark)
    )

    # Layer sweep
    print("\n" + "=" * 50)
    print("LAYER SWEEP")
    print("=" * 50)
    all_results.extend(
        run_layer_sweep(model, tokenizer, directions, snapshot, args.benchmark)
    )

    # Random control
    print("\n" + "=" * 50)
    print("RANDOM CONTROL")
    print("=" * 50)
    all_results.append(
        run_random_control(model, tokenizer, snapshot, args.benchmark)
    )

    # Save all results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()

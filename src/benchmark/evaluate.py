"""
Run models on the benchmark and classify responses as refuse/comply.
Supports two backends:
  - llama.cpp (GGUF models, fast, CPU or GPU)
  - transformers (HuggingFace models, for abliterated models)

Usage:
    python -m src.benchmark.evaluate \
        --backend llamacpp \
        --model models/gemma-4-e4b-Q4.gguf \
        --benchmark data/benchmark_prompts.json \
        --output results/gemma-4-e4b-original/

    python -m src.benchmark.evaluate \
        --backend transformers \
        --model google/gemma-4-E4B-it \
        --benchmark data/benchmark_prompts.json \
        --output results/gemma-4-e4b-original/ \
        --use-8bit
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.benchmark.classify_refusal import classify_refusal


# ──────────────────────────────────────────────
# Backend 1: llama.cpp
# ──────────────────────────────────────────────

def evaluate_with_llamacpp(model_path: str, benchmark_path: str,
                           output_path: str, n_gpu_layers: int = -1,
                           n_ctx: int = 2048) -> list[dict]:
    """
    Evaluate a GGUF model on the benchmark using llama.cpp.
    """
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    results = []

    for prompt_entry in tqdm(benchmark["prompts"], desc="Evaluating"):
        prompts_to_test = [prompt_entry["prompt"]] + prompt_entry.get("variants", [])

        for variant_idx, prompt_text in enumerate(prompts_to_test):
            messages = [{"role": "user", "content": prompt_text}]

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )

            response_text = response["choices"][0]["message"]["content"]
            is_refusal = classify_refusal(response_text)

            results.append({
                "prompt_id": prompt_entry["id"],
                "variant_idx": variant_idx,
                "category": prompt_entry["category"],
                "expected": prompt_entry["expected"],
                "actual": "refuse" if is_refusal else "comply",
                "correct": (prompt_entry["expected"] == "refuse") == is_refusal,
                "over_refusal": prompt_entry["expected"] == "comply" and is_refusal,
                "under_refusal": prompt_entry["expected"] == "refuse" and not is_refusal,
                "prompt": prompt_text,
                "response": response_text,
            })

    _save_results(results, output_path)
    return results


# ──────────────────────────────────────────────
# Backend 2: transformers
# ──────────────────────────────────────────────

def evaluate_with_transformers(model, tokenizer, benchmark_path: str,
                               output_path: str) -> list[dict]:
    """
    Evaluate a transformers model (e.g., after abliteration).
    Accepts pre-loaded model and tokenizer objects.
    """
    import torch

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    results = []

    for prompt_entry in tqdm(benchmark["prompts"], desc="Evaluating"):
        prompt_text = prompt_entry["prompt"]

        messages = [{"role": "user", "content": prompt_text}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=512, temperature=0.1, do_sample=True,
            )

        response_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        is_refusal = classify_refusal(response_text)

        results.append({
            "prompt_id": prompt_entry["id"],
            "variant_idx": 0,
            "category": prompt_entry["category"],
            "expected": prompt_entry["expected"],
            "actual": "refuse" if is_refusal else "comply",
            "correct": (prompt_entry["expected"] == "refuse") == is_refusal,
            "over_refusal": prompt_entry["expected"] == "comply" and is_refusal,
            "under_refusal": prompt_entry["expected"] == "refuse" and not is_refusal,
            "prompt": prompt_text,
            "response": response_text,
        })

    _save_results(results, output_path)
    return results


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _save_results(results: list[dict], output_path: str):
    """Save results and print summary."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path / 'evaluation_results.json'}")
    _print_summary(results)


def _print_summary(results: list[dict]):
    """Print evaluation summary."""
    total = len(results)
    refused = sum(1 for r in results if r["actual"] == "refuse")
    over_refused = sum(1 for r in results if r["over_refusal"])
    under_refused = sum(1 for r in results if r["under_refusal"])

    print(f"\n{'='*50}")
    print(f"Total prompts evaluated: {total}")
    print(f"Refused: {refused} ({100*refused/total:.1f}%)")
    print(f"Over-refusal (should comply, did refuse): {over_refused} ({100*over_refused/total:.1f}%)")
    print(f"Under-refusal (should refuse, did comply): {under_refused} ({100*under_refused/total:.1f}%)")
    print(f"{'='*50}")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    print(f"\n{'Category':<25} {'Refusal Rate':>12} {'Over-Refusal':>12}")
    print("-" * 52)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_refused = sum(1 for r in cat_results if r["actual"] == "refuse")
        cat_over = sum(1 for r in cat_results if r["over_refusal"])
        print(f"{cat:<25} {cat_refused}/{len(cat_results):>5}  "
              f"({100*cat_refused/len(cat_results):5.1f}%)  {cat_over:>5}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on over-refusal benchmark")
    parser.add_argument("--backend", choices=["llamacpp", "transformers"], required=True)
    parser.add_argument("--model", required=True, help="Model path (GGUF or HF model ID)")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark_prompts.json")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers for llama.cpp")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization (transformers)")
    args = parser.parse_args()

    if args.backend == "llamacpp":
        evaluate_with_llamacpp(args.model, args.benchmark, args.output, args.n_gpu_layers)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.use_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, device_map="auto", torch_dtype=torch.bfloat16
            )
        model.eval()
        evaluate_with_transformers(model, tokenizer, args.benchmark, args.output)


if __name__ == "__main__":
    main()

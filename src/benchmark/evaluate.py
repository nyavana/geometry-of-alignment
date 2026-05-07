"""
Run models on the benchmark and classify responses as refuse/comply.
Supports two backends:
  - llama.cpp via llama-server (GGUF models; OpenAI-compatible HTTP API)
  - transformers (HuggingFace models, for abliterated models)

Usage:
    # Start llama-server first (in another terminal), e.g.:
    #     llama-server -m model.gguf -ngl 99 --host 127.0.0.1 --port 8088
    python -m src.benchmark.evaluate \
        --backend llamacpp \
        --model gemma-4-e4b-it \
        --server-url http://127.0.0.1:8088 \
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
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

from src.benchmark.classify_refusal import classify_refusal


# ──────────────────────────────────────────────
# Backend 1: llama.cpp via llama-server
# ──────────────────────────────────────────────

def evaluate_with_llamacpp(server_url: str, model_name: str,
                           benchmark_path: str, output_path: str,
                           request_timeout: float = 300.0,
                           limit: int = None,
                           base_only: bool = False) -> list[dict]:
    """
    Evaluate a model served by llama-server (upstream llama.cpp) on the benchmark.

    The server must already be running and have a GGUF model loaded; the `model_name`
    argument is sent in the OpenAI-compatible request payload as a label only — the
    server uses whichever model was loaded with -m at launch.

    Args:
        limit: if set, evaluate only the first N prompts (useful for smoke tests).
        base_only: if True, skip variants and only evaluate the base prompt (variant_idx=0).
    """
    import requests

    endpoint = server_url.rstrip("/") + "/v1/chat/completions"

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    prompts = benchmark["prompts"]
    if limit is not None:
        prompts = prompts[:limit]

    results = []

    for prompt_entry in tqdm(prompts, desc="Evaluating"):
        if base_only:
            prompts_to_test = [prompt_entry["prompt"]]
        else:
            prompts_to_test = [prompt_entry["prompt"]] + prompt_entry.get("variants", [])

        for variant_idx, prompt_text in enumerate(prompts_to_test):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 2048,
                "temperature": 0.1,
            }

            response = requests.post(endpoint, json=payload, timeout=request_timeout)
            response.raise_for_status()
            resp_json = response.json()
            message = resp_json["choices"][0]["message"]
            # Gemma 4 can return an empty "content" when extended thinking uses
            # all tokens; fall back to "reasoning_content" so we don't mis-classify
            # real responses as refusals.  If both are empty, the model truly
            # produced no output and we treat that as a soft-refuse.
            response_text = message.get("content") or message.get("reasoning_content", "")
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
                               output_path: str,
                               limit: int = None,
                               base_only: bool = False,
                               max_new_tokens: int = 512) -> list[dict]:
    """
    Evaluate a transformers model (e.g., after abliteration).
    Accepts pre-loaded model and tokenizer objects.

    Args:
        limit: if set, evaluate only the first N prompts (useful for smoke tests).
        base_only: if True, skip variants and only evaluate the base prompt.
    """
    import torch

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    prompts = benchmark["prompts"]
    if limit is not None:
        prompts = prompts[:limit]

    results = []

    for prompt_entry in tqdm(prompts, desc="Evaluating"):
        prompt_text = prompt_entry["prompt"]

        messages = [{"role": "user", "content": prompt_text}]
        # transformers v5 + Gemma 4 (multimodal): apply_chat_template returns a
        # BatchEncoding when return_dict=True. Extract input_ids/attention_mask
        # explicitly so model.generate sees tensors regardless of tokenizer version.
        template_output = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            return_dict=True,
        )

        if hasattr(template_output, "input_ids"):
            # BatchEncoding path (multimodal tokenizer / return_dict=True)
            model_inputs = {k: v.to(model.device) for k, v in template_output.items()
                            if hasattr(v, "to")}
            input_ids = model_inputs["input_ids"]
            input_len = input_ids.shape[1]
        else:
            # Plain tensor path (older tokenizer versions)
            input_ids = template_output.to(model.device)
            model_inputs = {"input_ids": input_ids}
            input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
            )

        response_text = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
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
    """Save results (json + csv) and print summary."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = output_path / "evaluation_results.csv"
    if results:
        fieldnames = ["prompt_id", "variant_idx", "category", "expected", "actual",
                      "correct", "over_refusal", "under_refusal", "prompt", "response"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print(f"\nSaved {len(results)} results to {json_path} and {csv_path}")
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
    parser.add_argument("--model", required=True,
                        help="HF model ID (transformers) or label sent to llama-server (llamacpp)")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark_prompts.json")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--server-url", default="http://127.0.0.1:8088",
                        help="llama-server base URL (llamacpp backend; 8088 because Windows-side WSL2 binds 8080)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization (transformers)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N prompts (useful for smoke tests)")
    parser.add_argument("--base-only", action="store_true",
                        help="Skip variants, evaluate only the canonical base prompt per entry")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Generation cap; refusal classification needs ~50 tokens, "
                             "so 128 is safe for smoke runs and ~4x faster than 512.")
    args = parser.parse_args()

    if args.backend == "llamacpp":
        evaluate_with_llamacpp(args.server_url, args.model, args.benchmark, args.output,
                               limit=args.limit, base_only=args.base_only)
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
        evaluate_with_transformers(model, tokenizer, args.benchmark, args.output,
                                   limit=args.limit, base_only=args.base_only,
                                   max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()

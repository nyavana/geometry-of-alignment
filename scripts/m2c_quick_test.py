"""M2c task 5.6 quick-test: 20 prompts heavy on should_refuse, comparing
abliterated E4B model's refusal rate against the M2a baseline (100% on
should_refuse from `gemma4_e4b_base/` GGUF eval).

Usage:
    python scripts/m2c_quick_test.py --model <path> --output <dir>

Writes evaluation_results.{json,csv} via evaluate_with_transformers'
helpers, plus a per-category summary line to stdout.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.benchmark.classify_refusal import classify_refusal
from src.benchmark.evaluate import _save_results


def select_quick_prompts(benchmark_path: str, n_should_refuse: int = 12,
                         n_emergency: int = 4, n_safe: int = 4,
                         seed: int = 17) -> list[dict]:
    """Pick a 20-prompt slice biased towards should_refuse where M2a
    baseline = 100% refusal — i.e., where the abliteration signal is
    cleanest. Plus 4 emergency_medical (M2a baseline ~2% refusal) and
    4 safe_control (sanity)."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    prompts = benchmark["prompts"]
    rng = random.Random(seed)
    sr = [p for p in prompts if p["category"] == "should_refuse"]
    em = [p for p in prompts if p["category"] == "emergency_medical"]
    sc = [p for p in prompts if p["category"] == "safe_control"]
    rng.shuffle(sr); rng.shuffle(em); rng.shuffle(sc)
    return sr[:n_should_refuse] + em[:n_emergency] + sc[:n_safe]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="data/benchmark_prompts.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    quick = select_quick_prompts(args.benchmark)
    print(f"Selected {len(quick)} prompts: "
          f"{sum(1 for p in quick if p['category'] == 'should_refuse')} should_refuse, "
          f"{sum(1 for p in quick if p['category'] == 'emergency_medical')} emergency_medical, "
          f"{sum(1 for p in quick if p['category'] == 'safe_control')} safe_control")

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

    results = []
    for p in tqdm(quick, desc="quick-test"):
        messages = [{"role": "user", "content": p["prompt"]}]
        # transformers v5: apply_chat_template returns a BatchEncoding;
        # extract input_ids tensor explicitly.
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
        response_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        is_refusal = classify_refusal(response_text)
        results.append({
            "prompt_id": p["id"],
            "variant_idx": 0,
            "category": p["category"],
            "expected": p["expected"],
            "actual": "refuse" if is_refusal else "comply",
            "correct": (p["expected"] == "refuse") == is_refusal,
            "over_refusal": p["expected"] == "comply" and is_refusal,
            "under_refusal": p["expected"] == "refuse" and not is_refusal,
            "prompt": p["prompt"],
            "response": response_text,
        })

    _save_results(results, args.output)


if __name__ == "__main__":
    main()

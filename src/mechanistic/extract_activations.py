"""
Activation extraction framework for mechanistic analysis of refusal behavior.

Hooks into each transformer layer's residual stream output, collects
activations for refuse vs comply prompts, and computes refusal directions.

For Gemma 4 E4B-it:
  - 42 layers (35 sliding attention + 7 global attention)
  - hidden_size = 2560
  - At 8-bit quantization: ~7.5 GB VRAM, leaves ~8.5 GB headroom

Usage:
    python -m src.mechanistic.extract_activations \
        --model google/gemma-4-E4B-it \
        --benchmark data/benchmark_prompts.json \
        --use-8bit \
        --output results/activations/
"""

import torch
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name: str, use_8bit: bool = True):
    """Load model with optional 8-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


class ActivationCollector:
    """
    Hooks into each transformer layer and collects residual stream activations.

    Usage:
        collector = ActivationCollector(model)
        collector.register_hooks()

        with torch.no_grad():
            model(**inputs)

        activations = collector.get_activations()
        collector.clear()
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def _hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically a tuple; first element is the hidden states
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Store on CPU to save VRAM
            self.activations[layer_idx] = hidden_states.detach().cpu().float()
        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        # For Gemma 4, layers are at model.model.layers
        # IMPORTANT: verify this path by running print(model) first
        layers = self.model.model.layers
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._hook_fn(idx))
            self.hooks.append(hook)
        print(f"Registered hooks on {len(self.hooks)} layers")

    def get_activations(self) -> dict:
        """Return collected activations."""
        return dict(self.activations)

    def clear(self):
        """Clear stored activations for next forward pass."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def extract_activations_for_prompts(model, tokenizer, prompts: list[str],
                                     collector: ActivationCollector,
                                     position: str = "last") -> dict:
    """
    Run forward pass on each prompt and collect activations.

    Args:
        position: "last" = take last token position
                  "mean" = mean over all positions

    Returns:
        Dict[layer_idx, Tensor of shape (num_prompts, hidden_dim)]
    """
    all_activations = defaultdict(list)

    for prompt_text in tqdm(prompts, desc="Extracting activations"):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        collector.clear()
        with torch.no_grad():
            model(**inputs)

        layer_acts = collector.get_activations()
        for layer_idx, act in layer_acts.items():
            # act shape: (1, seq_len, hidden_dim)
            if position == "last":
                vec = act[0, -1, :]
            elif position == "mean":
                vec = act[0].mean(dim=0)
            else:
                raise ValueError(f"Unknown position: {position}")
            all_activations[layer_idx].append(vec)

    result = {}
    for layer_idx, vecs in all_activations.items():
        result[layer_idx] = torch.stack(vecs)

    return result


def compute_refusal_directions(refuse_acts: dict, comply_acts: dict) -> dict:
    """
    Compute normalized refusal direction at each layer.
    direction = normalize(mean(refuse) - mean(comply))
    """
    directions = {}
    for layer_idx in refuse_acts:
        mean_refuse = refuse_acts[layer_idx].mean(dim=0)
        mean_comply = comply_acts[layer_idx].mean(dim=0)
        direction = mean_refuse - mean_comply
        direction = direction / direction.norm()
        directions[layer_idx] = direction
    return directions


def main():
    parser = argparse.ArgumentParser(description="Extract activations for refusal analysis")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--benchmark", default="data/benchmark_prompts.json")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--output", default="results/activations/")
    parser.add_argument("--position", default="last", choices=["last", "mean"])
    args = parser.parse_args()

    # Load benchmark
    with open(args.benchmark) as f:
        benchmark = json.load(f)

    refuse_prompts = [p["prompt"] for p in benchmark["prompts"]
                      if p["category"] == "should_refuse"]
    comply_prompts = [p["prompt"] for p in benchmark["prompts"]
                      if p["category"] == "safe_control"]

    # Also include expected-comply categories that models tend to over-refuse
    over_refuse_categories = ["emergency_medical", "wilderness_survival",
                              "home_safety", "chemistry_safety", "mental_health"]
    additional_refuse = [p["prompt"] for p in benchmark["prompts"]
                         if p["category"] in over_refuse_categories]
    # These are prompts the model SHOULD comply with but may refuse —
    # we add them to the refuse set if they actually get refused
    # For initial direction computation, use the clear-cut categories

    print(f"Refuse prompts: {len(refuse_prompts)}")
    print(f"Comply prompts: {len(comply_prompts)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.use_8bit)

    # Set up collector
    collector = ActivationCollector(model)
    collector.register_hooks()

    # Extract
    print("\nExtracting REFUSE activations...")
    refuse_acts = extract_activations_for_prompts(
        model, tokenizer, refuse_prompts, collector, position=args.position
    )

    print("\nExtracting COMPLY activations...")
    comply_acts = extract_activations_for_prompts(
        model, tokenizer, comply_prompts, collector, position=args.position
    )

    # Save activations
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(refuse_acts, output_dir / "refuse_activations.pt")
    torch.save(comply_acts, output_dir / "comply_activations.pt")
    print(f"\nSaved activations to {output_dir}")

    # Compute and save refusal directions
    directions = compute_refusal_directions(refuse_acts, comply_acts)
    torch.save(directions, output_dir / "refusal_directions.pt")
    print(f"Saved refusal directions for {len(directions)} layers")

    collector.remove_hooks()


if __name__ == "__main__":
    main()

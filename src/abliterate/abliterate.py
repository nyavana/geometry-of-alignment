"""
Core abliteration algorithm.

Removes the refusal direction from model weights via rank-1 projection:
    W_new = W - alpha * d * (d^T @ W)

This is a rank-1 update to weight matrices that projects out the component
of the output along the refusal direction d.

Usage:
    python -m src.abliterate.abliterate \
        --model google/gemma-4-E4B-it \
        --directions results/activations/refusal_directions.pt \
        --alpha 1.0 \
        --output models/gemma-4-e4b-abliterated/
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.decomposition import PCA


def compute_refusal_direction(refuse_acts: dict, comply_acts: dict,
                               layer_idx: int, method: str = "mean_diff"):
    """
    Compute the refusal direction at a specific layer.

    Methods:
        "mean_diff": simple mean difference (standard abliteration)
        "pca": first PCA component of the difference space
    """
    mean_refuse = refuse_acts[layer_idx].mean(dim=0)
    mean_comply = comply_acts[layer_idx].mean(dim=0)

    if method == "mean_diff":
        direction = mean_refuse - mean_comply
        direction = direction / direction.norm()
        return direction.unsqueeze(0)  # (1, hidden_dim)

    elif method == "pca":
        all_acts = torch.cat([refuse_acts[layer_idx], comply_acts[layer_idx]])
        pca = PCA(n_components=1)
        pca.fit(all_acts.numpy())
        direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        direction = direction / direction.norm()

        if (mean_refuse - mean_comply) @ direction < 0:
            direction = -direction
        return direction.unsqueeze(0)

    else:
        raise ValueError(f"Unknown method: {method}")


def abliterate_model(model, refusal_directions: dict,
                     layers_to_modify: list[int] | None = None,
                     alpha: float = 1.0,
                     target_weights: str = "residual",
                     norm_preserving: bool = False):
    """
    Apply abliteration to a model in-place.

    Args:
        model: HuggingFace model
        refusal_directions: dict {layer_idx: tensor (k, hidden_dim)}
            Each tensor contains k direction vectors to project out.
            For standard abliteration, k=1.
        layers_to_modify: list of layer indices (None = all available)
        alpha: projection strength (0=no change, 1=full removal, >1=over-removal)
        target_weights: which weights to modify
            "residual" — both attention and MLP output projections
            "mlp"      — MLP output projection only
            "attn"     — attention output projection only
        norm_preserving: if True, rescale each row of W after projection
            so that ‖W'_i‖ == ‖W_i‖ for every row i. Targets RMSNorm
            sensitivity to row-norm changes (TrevorJS biprojection style).
            Removes the d-component from the column space (output) while
            keeping per-row magnitude untouched, since uniform rescaling
            of a row does not reintroduce d-content.
    """
    if layers_to_modify is None:
        layers_to_modify = list(refusal_directions.keys())

    modified_count = 0

    for layer_idx in tqdm(layers_to_modify, desc=f"Abliterating (alpha={alpha})"):
        if layer_idx not in refusal_directions:
            continue

        directions = refusal_directions[layer_idx]
        if directions.dim() == 1:
            directions = directions.unsqueeze(0)

        # Gemma 4 multimodal: text layers live at model.model.language_model.layers.
        # Older Gemma/Llama-style locate them at model.model.layers. Try Gemma 4 first.
        if hasattr(model.model, "language_model"):
            layer = model.model.language_model.layers[layer_idx]
        else:
            layer = model.model.layers[layer_idx]

        def _project_out(weight_matrix, directions, alpha):
            """W_new = W - alpha * sum(d * (d^T @ W)) for each direction d.

            If norm_preserving is set on the outer abliterate_model call,
            each row of W is rescaled afterward so that
            ‖W'_i‖ == ‖W_i‖ — preserving RMSNorm-relevant row magnitudes
            while still removing the d-component from the residual-stream
            output (uniform per-row scaling does not reintroduce
            d-content).
            """
            W = weight_matrix.data.float()
            if norm_preserving:
                row_norms_orig = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
            for d in directions:
                d = d.to(W.device)
                proj = d @ W  # (in_features,)
                W = W - alpha * torch.outer(d, proj)
            if norm_preserving:
                row_norms_proj = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
                W = W * (row_norms_orig / row_norms_proj)
            weight_matrix.data = W.to(weight_matrix.dtype)

        if target_weights in ("attn", "residual"):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                _project_out(layer.self_attn.o_proj.weight, directions, alpha)
                modified_count += 1

        if target_weights in ("mlp", "residual"):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                _project_out(layer.mlp.down_proj.weight, directions, alpha)
                modified_count += 1

    print(f"Modified {modified_count} weight matrices across "
          f"{len(layers_to_modify)} layers (alpha={alpha})")
    return model


def save_abliterated_model(model, tokenizer, output_path: str):
    """Save the modified model for later evaluation."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved abliterated model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Abliterate a model")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--directions", default="results/activations/refusal_directions.pt",
                        help="Path to saved refusal directions")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--target-weights", default="residual",
                        choices=["residual", "mlp", "attn"])
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--norm-preserving", action="store_true",
                        help="After rank-1 projection, rescale each row of W "
                             "so that ‖W'_i‖ == ‖W_i‖ — RMSNorm-sensitive on "
                             "Gemma 4. Stage 3a hypothesis: vanilla projection "
                             "fails because it changes row norms.")
    parser.add_argument("--output", default="models/gemma-4-e4b-abliterated/")
    args = parser.parse_args()

    # Load directions
    raw_directions = torch.load(args.directions, weights_only=True)
    # Ensure each direction is (1, hidden_dim) for consistency
    directions = {}
    for k, v in raw_directions.items():
        if v.dim() == 1:
            directions[k] = v.unsqueeze(0)
        else:
            directions[k] = v

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None

    # Load model
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
            args.model, device_map="auto", torch_dtype=torch.bfloat16,
        )

    # Abliterate
    abliterate_model(model, directions, layers, args.alpha, args.target_weights,
                     norm_preserving=args.norm_preserving)

    # Save
    save_abliterated_model(model, tokenizer, args.output)


if __name__ == "__main__":
    main()

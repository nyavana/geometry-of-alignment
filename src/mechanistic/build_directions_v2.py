"""
Build refusal directions from existing activation tensors with composable
direction-quality ingredients.

Consumes the activation pair produced by `extract_activations.py`
(refuse_activations*.pt / comply_activations*.pt — keyed by layer_idx,
each value a tensor of shape (n_prompts, hidden_dim)) and emits a
single direction-per-layer artifact (dict[layer_idx -> tensor(hidden_dim)]),
unit-norm.

Three composable knobs, each tested in isolation by Stage 2 of the M6
follow-up plan:

    --winsorize-pct 99.5
        Clip activations element-wise at the per-layer 99.5th percentile
        of |a| BEFORE taking the mean. This dampens GeGLU outlier neurons
        that distort the raw mean-diff direction on Gemma 4.

    --orthogonalize-against-harmless-mean
        After computing direction = normalize(mean_refuse - mean_comply),
        project out the component along normalize(mean_comply) via two
        passes of Gram-Schmidt for numerical stability. Removes
        task-relevant signal that leaks via the harmless mean.

The two flags compose: D2 = winsorize-only, D3 = winsorize + orthogonalize.
"""

import torch
import argparse
import json
from pathlib import Path


def winsorize_per_layer(acts: dict, pct: float) -> dict:
    """
    Element-wise clip activations at the per-layer percentile of |a|.

    For each layer's activation tensor (shape n_prompts × hidden_dim),
    compute c = quantile(abs(layer_acts).flatten(), pct/100), then clip
    each element to [-c, +c]. Returns a new dict (does not mutate input).
    """
    p = pct / 100.0
    out = {}
    for layer_idx, t in acts.items():
        flat = t.flatten().abs()
        c = torch.quantile(flat, p)
        out[layer_idx] = t.clamp(min=-c.item(), max=c.item())
    return out


def compute_mean_diff_directions(refuse_acts: dict, comply_acts: dict) -> dict:
    """direction = normalize(mean_refuse - mean_comply), per layer."""
    directions = {}
    for layer_idx in refuse_acts:
        if layer_idx not in comply_acts:
            continue
        mean_refuse = refuse_acts[layer_idx].mean(dim=0)
        mean_comply = comply_acts[layer_idx].mean(dim=0)
        d = mean_refuse - mean_comply
        d = d / d.norm()
        directions[layer_idx] = d
    return directions


def orthogonalize_against_comply(directions: dict, comply_acts: dict) -> dict:
    """
    Two-pass Gram-Schmidt: remove the component of each direction along
    normalize(mean_comply) at the same layer. Two passes guard against
    numerical drift; the second pass is essentially a no-op when float32
    is enough but stabilizes float16/bf16 inputs.
    """
    out = {}
    for layer_idx, d in directions.items():
        if layer_idx not in comply_acts:
            out[layer_idx] = d
            continue
        mc = comply_acts[layer_idx].mean(dim=0)
        mc_unit = mc / mc.norm()
        d2 = d - (d @ mc_unit) * mc_unit
        d2 = d2 / d2.norm()
        # second pass for stability
        d2 = d2 - (d2 @ mc_unit) * mc_unit
        d2 = d2 / d2.norm()
        out[layer_idx] = d2
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build M6 Stage 2 direction artifacts (D2/D3) from "
                    "existing chat-template activations."
    )
    parser.add_argument("--refuse-activations", required=True,
                        help="Path to refuse_activations*.pt from extract_activations.py")
    parser.add_argument("--comply-activations", required=True,
                        help="Path to comply_activations*.pt from extract_activations.py")
    parser.add_argument("--winsorize-pct", type=float, default=None,
                        help="If set, clip activations element-wise at this "
                             "per-layer percentile of |a| before the mean.")
    parser.add_argument("--orthogonalize-against-harmless-mean", action="store_true",
                        help="After mean-diff, two-pass Gram-Schmidt against "
                             "normalize(mean_comply) at each layer.")
    parser.add_argument("--output", required=True,
                        help="Output .pt path for the directions dict.")
    parser.add_argument("--summary", default=None,
                        help="Optional .json path for a per-layer summary "
                             "(direction norms, orthogonality residuals).")
    args = parser.parse_args()

    print(f"Loading refuse activations from {args.refuse_activations}")
    refuse_acts = torch.load(args.refuse_activations,
                             weights_only=False, map_location="cpu")
    refuse_acts = {k: v.float() for k, v in refuse_acts.items()}
    print(f"  {len(refuse_acts)} layers; sample shape "
          f"{next(iter(refuse_acts.values())).shape}")

    print(f"Loading comply activations from {args.comply_activations}")
    comply_acts = torch.load(args.comply_activations,
                             weights_only=False, map_location="cpu")
    comply_acts = {k: v.float() for k, v in comply_acts.items()}
    print(f"  {len(comply_acts)} layers; sample shape "
          f"{next(iter(comply_acts.values())).shape}")

    pre_max_norms = {k: v.abs().max().item() for k, v in refuse_acts.items()}

    if args.winsorize_pct is not None:
        print(f"\nWinsorizing at the {args.winsorize_pct}th percentile per layer "
              f"(refuse and comply, element-wise)...")
        refuse_acts = winsorize_per_layer(refuse_acts, args.winsorize_pct)
        comply_acts = winsorize_per_layer(comply_acts, args.winsorize_pct)

    post_max_norms = {k: v.abs().max().item() for k, v in refuse_acts.items()}

    print("\nComputing mean-diff directions...")
    directions = compute_mean_diff_directions(refuse_acts, comply_acts)
    print(f"  built {len(directions)} per-layer directions")

    orth_residuals = {}
    if args.orthogonalize_against_harmless_mean:
        print("\nOrthogonalizing against normalize(mean_comply) "
              "(two-pass Gram-Schmidt)...")
        directions = orthogonalize_against_comply(directions, comply_acts)
        for li, d in directions.items():
            mc = comply_acts[li].mean(dim=0)
            mc_unit = mc / mc.norm()
            orth_residuals[li] = float((d @ mc_unit).item())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, out_path)
    print(f"\nSaved directions to {out_path}")

    if args.summary:
        summary = {
            "n_layers": len(directions),
            "winsorize_pct": args.winsorize_pct,
            "orthogonalized": args.orthogonalize_against_harmless_mean,
            "pre_clip_max_abs": pre_max_norms,
            "post_clip_max_abs": post_max_norms,
            "direction_norms": {li: float(d.norm().item()) for li, d in directions.items()},
            "orth_residual_dot": orth_residuals,
        }
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.summary}")

    # Print a compact per-layer report so downstream verification can
    # read it from the run log without loading the .pt file.
    print("\nLayer  pre|max|   post|max|   norm   orth_resid")
    for li in sorted(directions.keys()):
        pre = pre_max_norms.get(li, float("nan"))
        post = post_max_norms.get(li, float("nan"))
        n = float(directions[li].norm().item())
        r = orth_residuals.get(li, float("nan"))
        print(f"  L{li:2d}    {pre:7.3f}  {post:7.3f}  {n:.4f}   {r:+.2e}")


if __name__ == "__main__":
    main()

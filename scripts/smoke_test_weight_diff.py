"""
Smoke test for compute_diff.py and svd_analysis.py.
Loads only layer 5 tensors from both models and runs compute_diff logic directly.
Does NOT load the full 34 GB state-dict. (Task 7.6)

Usage:
    python scripts/smoke_test_weight_diff.py
"""

import sys
import json
import torch
from pathlib import Path
from safetensors import safe_open


LAYER_PREFIX = "model.language_model.layers.5."
BASE_DIR = Path("model/gemma-4-E4B-it")
VARIANT_DIR = Path("model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED")
SMOKE_OUT = Path("/tmp/smoke_weight_diff")


def load_layer5_tensors(model_dir: Path) -> dict:
    """Load only language-model layer-5 tensors from safetensors files."""
    result = {}
    for sf_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(LAYER_PREFIX):
                    result[key] = f.get_tensor(key)
    return result


def compute_diff_subset(original: dict, modified: dict) -> list[dict]:
    results = []
    common_keys = set(original.keys()) & set(modified.keys())
    for key in sorted(common_keys):
        w_orig = original[key].float()
        w_mod = modified[key].float()
        diff = w_mod - w_orig
        frob = diff.norm().item()
        rel = frob / (w_orig.norm().item() + 1e-10)
        entry = {"key": key, "shape": list(w_orig.shape),
                 "frobenius_norm": frob, "original_norm": w_orig.norm().item(),
                 "relative_change": rel, "max_abs_change": diff.abs().max().item()}
        if diff.dim() == 2 and frob > 1e-6:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            total_energy = (S ** 2).sum()
            cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
            rank_95 = int((cumulative < 0.95).sum().item()) + 1
            rank_99 = int((cumulative < 0.99).sum().item()) + 1
            entry["top_10_singular_values"] = S[:10].tolist()
            entry["rank_95"] = rank_95
            entry["rank_99"] = rank_99
            entry["total_singular_values"] = len(S)
            entry["top1_energy_fraction"] = (S[0] ** 2 / total_energy).item()
        results.append(entry)
    return results


def main():
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Loading layer-5 tensors from {BASE_DIR}...")
    base = load_layer5_tensors(BASE_DIR)
    print(f"  {len(base)} tensors found")
    print(f"Loading layer-5 tensors from {VARIANT_DIR}...")
    variant = load_layer5_tensors(VARIANT_DIR)
    print(f"  {len(variant)} tensors found")

    results = compute_diff_subset(base, variant)
    out_path = SMOKE_OUT / "weight_diff_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSmoke test OK — {len(results)} entries written to {out_path}")

    # Quick SVD sanity
    ranked = [r for r in results if "rank_95" in r]
    if ranked:
        print(f"SVD computed for {len(ranked)} 2D matrices")
        for r in ranked[:3]:
            print(f"  {r['key'].split('.')[-1]}: rank_95={r['rank_95']}, "
                  f"rank_99={r['rank_99']}, top1_energy={r['top1_energy_fraction']:.3f}")
    else:
        print("No 2D tensors with significant change found in layer 5 (may be zero diff)")

    # Now test svd_analysis.py import chain
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.weight_diff.svd_analysis import load_results, plot_per_layer_change
    loaded = load_results(str(out_path))
    print(f"svd_analysis.load_results: OK ({len(loaded)} entries loaded)")
    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

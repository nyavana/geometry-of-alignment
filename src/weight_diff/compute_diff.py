"""
Compare weights between a base model and a published abliterated variant.
Runs entirely on CPU using safetensors for memory-efficient loading.

Usage:
    python -m src.weight_diff.compute_diff \
        --original model/gemma-4-E4B-it/ \
        --modified model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/ \
        --output results/weight_diffs/gemma_obliteratus/
"""

import sys
import torch
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from safetensors import safe_open
from tqdm import tqdm


def load_state_dict_from_safetensors(model_dir: str) -> dict:
    """Load model weights from safetensors files on CPU."""
    model_dir = Path(model_dir)
    state_dict = {}

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        safetensor_files = sorted(model_dir.glob("model*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    for sf_path in tqdm(safetensor_files, desc=f"Loading {model_dir.name}"):
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors from {model_dir}")
    return state_dict


def _check_compat(original: dict, modified: dict, modified_dir: str,
                  output_dir: Path, strict: bool) -> bool:
    """
    Pre-flight key/shape compatibility check.

    Asserts variant.keys ⊆ base.keys and that common keys have matching shapes.
    Logs failures to <output_dir>/../.compat_log.md (one centralized log across
    all variants in the same parent dir, e.g. results/weight_diffs/.compat_log.md).
    Returns True if compatible. If strict and incompatible, returns False — caller
    should exit nonzero rather than produce a corrupt diff. If not strict and
    incompatible, logs and returns True so the caller proceeds with the
    intersection (TrevorJS-style fallback per task 7.5).
    """
    base_keys = set(original.keys())
    var_keys = set(modified.keys())
    extra_in_variant = var_keys - base_keys
    shape_mismatches = []
    for k in base_keys & var_keys:
        if original[k].shape != modified[k].shape:
            shape_mismatches.append((k, tuple(original[k].shape), tuple(modified[k].shape)))

    incompatible = bool(extra_in_variant) or bool(shape_mismatches)
    if not incompatible:
        return True

    log_path = output_dir.parent / ".compat_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} — {modified_dir}\n\n")
        f.write(f"- mode: {'strict' if strict else 'lenient'}\n")
        f.write(f"- base keys: {len(base_keys)}\n")
        f.write(f"- variant keys: {len(var_keys)}\n")
        if extra_in_variant:
            f.write(f"- variant has {len(extra_in_variant)} keys not in base: ")
            f.write(", ".join(sorted(extra_in_variant)[:10]))
            if len(extra_in_variant) > 10:
                f.write(f", ... ({len(extra_in_variant) - 10} more)")
            f.write("\n")
        if shape_mismatches:
            f.write(f"- {len(shape_mismatches)} shape mismatches:\n")
            for k, s_orig, s_mod in shape_mismatches[:10]:
                f.write(f"  - `{k}`: base={s_orig} vs variant={s_mod}\n")
            if len(shape_mismatches) > 10:
                f.write(f"  - ... ({len(shape_mismatches) - 10} more)\n")
        f.write(f"- decision: {'ABORT (strict)' if strict else 'proceed with intersection'}\n")

    print(f"Compatibility issues logged to {log_path}", file=sys.stderr)
    return not strict


def compute_weight_diffs(original_dir: str, modified_dir: str,
                         output_dir: str, strict: bool = False) -> list[dict]:
    """
    Layer-by-layer comparison of original vs modified model weights.
    Computes Frobenius norm, relative change, SVD rank for each parameter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original model weights...")
    original = load_state_dict_from_safetensors(original_dir)

    print("Loading modified model weights...")
    modified = load_state_dict_from_safetensors(modified_dir)

    if not _check_compat(original, modified, modified_dir, output_dir, strict):
        sys.exit(1)

    common_keys = set(original.keys()) & set(modified.keys())
    print(f"Common parameter tensors: {len(common_keys)}")

    only_original = set(original.keys()) - set(modified.keys())
    only_modified = set(modified.keys()) - set(original.keys())
    if only_original:
        print(f"Only in original: {len(only_original)}")
    if only_modified:
        print(f"Only in modified: {len(only_modified)}")

    results = []
    significant_diffs = {}

    for key in tqdm(sorted(common_keys), desc="Computing diffs"):
        w_orig = original[key].float()
        w_mod = modified[key].float()

        if w_orig.shape != w_mod.shape:
            print(f"  Shape mismatch for {key}: {w_orig.shape} vs {w_mod.shape}")
            continue

        diff = w_mod - w_orig

        frobenius_norm = diff.norm().item()
        orig_norm = w_orig.norm().item()
        relative_change = frobenius_norm / (orig_norm + 1e-10)
        max_abs_change = diff.abs().max().item()

        entry = {
            "key": key,
            "shape": list(w_orig.shape),
            "frobenius_norm": frobenius_norm,
            "original_norm": orig_norm,
            "relative_change": relative_change,
            "max_abs_change": max_abs_change,
        }

        # SVD for 2D matrices with non-trivial changes
        if diff.dim() == 2 and frobenius_norm > 1e-6:
            try:
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

                if relative_change > 0.001:
                    significant_diffs[key] = {
                        "U_top5": U[:, :5],
                        "S_top5": S[:5],
                        "Vh_top5": Vh[:5, :],
                    }
            except Exception as e:
                entry["svd_error"] = str(e)

        results.append(entry)

    # Save
    with open(output_dir / "weight_diff_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if significant_diffs:
        torch.save(significant_diffs, output_dir / "significant_diff_svd.pt")

    # Summary
    changed = [r for r in results if r["frobenius_norm"] > 1e-6]
    print(f"\n{'='*50}")
    print(f"Changed parameters: {len(changed)} / {len(results)}")
    print(f"Unchanged parameters: {len(results) - len(changed)}")
    if changed:
        max_change = max(changed, key=lambda x: x["relative_change"])
        print(f"Largest relative change: {max_change['key']} "
              f"({max_change['relative_change']:.4f})")

    del original, modified
    import gc; gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute weight diffs")
    parser.add_argument("--original", required=True, help="Original model directory")
    parser.add_argument("--modified", required=True, help="Modified model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--strict", action="store_true",
                        help="Fail (exit 1) on key/shape mismatch instead of proceeding "
                             "with the intersection. Use for primary variants where a "
                             "compatibility failure is a project-stopping signal "
                             "(e.g. OBLITERATUS); omit for fallback-eligible variants "
                             "(e.g. TrevorJS) — see task 7.5.")
    args = parser.parse_args()

    compute_weight_diffs(args.original, args.modified, args.output, strict=args.strict)


if __name__ == "__main__":
    main()

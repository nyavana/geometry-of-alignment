"""
Pre-flight state-dict key/shape compatibility check (task 7.5).

Enumerates safetensors headers (no tensor materialization) and asserts:
  - variant.keys ⊆ base.keys
  - all common keys have matching shapes

Logs any failures to <results_dir>/weight_diffs/.compat_log.md.
Exits 0 if compatible, 1 if OBLITERATUS fails, 2 if TrevorJS only fails.

Usage:
    python scripts/check_state_dict_compat.py \
        --base model/gemma-4-E4B-it/ \
        --variants model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/ model/TrevorJS-gemma-4-E4B-it-uncensored/ \
        --results-dir $RESULTS_DIR
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from safetensors import safe_open


def enumerate_keys_and_shapes(model_dir: Path) -> dict[str, tuple]:
    """Return {key: shape} using safe_open without loading any tensors."""
    result = {}
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        safetensor_files = sorted(model_dir.glob("model*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                result[key] = tuple(f.get_slice(key).get_shape())
    return result


def check_compat(base_shapes: dict, variant_shapes: dict, variant_name: str,
                 log_path: Path) -> bool:
    """
    Returns True if variant is compatible with base (variant.keys ⊆ base.keys
    and all common keys have matching shapes).
    Writes a markdown block to log_path on failure.
    """
    base_keys = set(base_shapes.keys())
    var_keys = set(variant_shapes.keys())

    extra_in_variant = var_keys - base_keys
    shape_mismatches = []
    for k in base_keys & var_keys:
        if base_shapes[k] != variant_shapes[k]:
            shape_mismatches.append((k, base_shapes[k], variant_shapes[k]))

    compatible = not extra_in_variant and not shape_mismatches

    if not compatible:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} — {variant_name}\n\n")
            f.write(f"- base keys: {len(base_keys)}\n")
            f.write(f"- variant keys: {len(var_keys)}\n")
            if extra_in_variant:
                f.write(f"- variant has {len(extra_in_variant)} keys not in base:\n")
                for k in sorted(extra_in_variant)[:20]:
                    f.write(f"  - `{k}`\n")
                if len(extra_in_variant) > 20:
                    f.write(f"  - ... ({len(extra_in_variant) - 20} more)\n")
            if shape_mismatches:
                f.write(f"- {len(shape_mismatches)} shape mismatches:\n")
                for k, s_b, s_v in shape_mismatches[:20]:
                    f.write(f"  - `{k}`: base={s_b} vs variant={s_v}\n")
                if len(shape_mismatches) > 20:
                    f.write(f"  - ... ({len(shape_mismatches) - 20} more)\n")

        print(f"[FAIL] {variant_name}: incompatible — see {log_path}", file=sys.stderr)
    else:
        print(f"[OK] {variant_name}: {len(var_keys)} keys, all shapes match base")

    return compatible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base)
    results_dir = Path(args.results_dir)
    log_path = results_dir / "weight_diffs" / ".compat_log.md"

    print(f"Enumerating base model: {base_dir}")
    base_shapes = enumerate_keys_and_shapes(base_dir)
    print(f"Base model: {len(base_shapes)} keys")

    all_ok = True
    obliteratus_ok = True

    for variant_path_str in args.variants:
        variant_dir = Path(variant_path_str)
        variant_name = variant_dir.name
        print(f"\nChecking variant: {variant_name}")
        variant_shapes = enumerate_keys_and_shapes(variant_dir)
        print(f"Variant: {len(variant_shapes)} keys")
        ok = check_compat(base_shapes, variant_shapes, variant_name, log_path)
        if not ok:
            all_ok = False
            if "OBLITERATUS" in variant_name or "obliteratus" in variant_name.lower():
                obliteratus_ok = False

    if not obliteratus_ok:
        print("\nFATAL: OBLITERATUS compatibility check failed. Stop and surface to operator.")
        sys.exit(1)

    if not all_ok:
        print("\nWARN: TrevorJS compatibility check failed. Proceeding with OBLITERATUS only (D2 fallback).")
        sys.exit(2)

    print("\nAll variants compatible with base.")
    sys.exit(0)


if __name__ == "__main__":
    main()

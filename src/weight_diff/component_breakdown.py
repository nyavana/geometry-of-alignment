"""
Component-type breakdown of weight diffs per variant.

Groups all changed parameters per variant by component family and reports:
- count_changed: number of changed tensors in that family
- total_frob: sum of Frobenius norms across that family
- mean_rank_95: mean of rank_95 (effective rank at 95% energy) across changed
  2D tensors in that family

Component families (in priority order):
- attention.q_proj
- attention.k_proj
- attention.v_proj
- attention.o_proj
- mlp.gate_proj
- mlp.up_proj
- mlp.down_proj
- embed_tokens
- lm_head
- *_norm  (any RMSNorm scale: input_layernorm, post_attention_layernorm,
           pre_feedforward_layernorm, post_feedforward_layernorm, q_norm,
           k_norm, model.norm, etc.)
- other

Reads the de-dup'd results file (preferring it over the raw 7.7 output) so
shared-K/V borrower entries don't inflate counts.

Outputs:
    $RESULTS_DIR/weight_diffs/component_type_breakdown.csv
    (mirrored to results/weight_diffs/)

Usage:
    python -m src.weight_diff.component_breakdown \\
        --variants gemma_obliteratus=$RESULTS_DIR/weight_diffs/gemma_obliteratus \\
                   gemma_trevorjs=$RESULTS_DIR/weight_diffs/gemma_trevorjs \\
        --output $RESULTS_DIR/weight_diffs/component_type_breakdown.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# Order: more-specific patterns first; first match wins.
FAMILY_RULES: list[tuple[str, str]] = [
    # Attention sub-blocks
    ("self_attn.q_proj.weight", "attention.q_proj"),
    ("self_attn.k_proj.weight", "attention.k_proj"),
    ("self_attn.v_proj.weight", "attention.v_proj"),
    ("self_attn.o_proj.weight", "attention.o_proj"),
    # MLP sub-blocks
    ("mlp.gate_proj.weight", "mlp.gate_proj"),
    ("mlp.up_proj.weight", "mlp.up_proj"),
    ("mlp.down_proj.weight", "mlp.down_proj"),
    # Top-level
    ("embed_tokens.weight", "embed_tokens"),
    ("lm_head.weight", "lm_head"),
]
# Norm scale tensors — match by suffix containing "_norm" or ".norm.weight".
NORM_FAMILY = "*_norm"
OTHER_FAMILY = "other"


def classify(key: str) -> str:
    for pattern, family in FAMILY_RULES:
        if pattern in key:
            return family
    if "_norm.weight" in key or key.endswith(".norm.weight") or "layernorm" in key:
        return NORM_FAMILY
    return OTHER_FAMILY


def load_results(variant_dir: Path) -> list[dict]:
    """Prefer the dedup'd file (M3 7.11) over the raw."""
    dedup = variant_dir / "weight_diff_results_dedup.json"
    raw = variant_dir / "weight_diff_results.json"
    path = dedup if dedup.exists() else raw
    with open(path) as f:
        return json.load(f)


def build_breakdown(results: list[dict]) -> dict[str, dict]:
    families: dict[str, dict] = defaultdict(
        lambda: {"count_changed": 0, "total_frob": 0.0, "rank_95_vals": []}
    )
    for r in results:
        if r.get("frobenius_norm", 0.0) <= 1e-6:
            continue
        family = classify(r["key"])
        bucket = families[family]
        bucket["count_changed"] += 1
        bucket["total_frob"] += r["frobenius_norm"]
        if "rank_95" in r:
            bucket["rank_95_vals"].append(r["rank_95"])
    # Compute mean_rank_95 per family
    out = {}
    for fam, bucket in families.items():
        ranks = bucket["rank_95_vals"]
        out[fam] = {
            "count_changed": bucket["count_changed"],
            "total_frob": round(bucket["total_frob"], 6),
            "mean_rank_95": round(sum(ranks) / len(ranks), 3) if ranks else None,
        }
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "component_family", "count_changed", "total_frob",
              "mean_rank_95"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# Family ordering for the CSV (matches the spec's listed order).
FAMILY_ORDER = [
    "attention.q_proj",
    "attention.k_proj",
    "attention.v_proj",
    "attention.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "embed_tokens",
    "lm_head",
    "*_norm",
    "other",
]


def run(variant_dirs: dict[str, str], output_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for vname, vdir in variant_dirs.items():
        res = load_results(Path(vdir))
        breakdown = build_breakdown(res)
        # Emit in canonical family order, including zero-count families for
        # consistency between variants.
        for fam in FAMILY_ORDER:
            stats = breakdown.get(fam, {"count_changed": 0, "total_frob": 0.0,
                                          "mean_rank_95": None})
            rows.append({
                "variant": vname,
                "component_family": fam,
                "count_changed": stats["count_changed"],
                "total_frob": stats["total_frob"],
                "mean_rank_95": stats["mean_rank_95"],
            })
        # Append any unanticipated families that appear in the data but not
        # in FAMILY_ORDER (defensive).
        for fam, stats in breakdown.items():
            if fam not in FAMILY_ORDER:
                rows.append({
                    "variant": vname,
                    "component_family": fam,
                    "count_changed": stats["count_changed"],
                    "total_frob": stats["total_frob"],
                    "mean_rank_95": stats["mean_rank_95"],
                })
    write_csv(rows, output_csv)

    # Print human-readable summary
    print("\n=== Component-type breakdown ===")
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    for vname, vrows in by_variant.items():
        print(f"\n{vname}:")
        print(f"  {'family':<22} {'count':>6} {'total_frob':>12} {'mean_rank_95':>14}")
        for r in vrows:
            mr = (
                f"{r['mean_rank_95']:.2f}" if r["mean_rank_95"] is not None else "—"
            )
            print(
                f"  {r['component_family']:<22} {r['count_changed']:>6} "
                f"{r['total_frob']:>12.4f} {mr:>14}"
            )
    return rows


def _parse_kv(arg: list[str]) -> dict[str, str]:
    out = {}
    for a in arg:
        if "=" not in a:
            raise SystemExit(f"--variants entries must be NAME=DIR, got {a!r}")
        k, v = a.split("=", 1)
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Component-type breakdown")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(
        variant_dirs=_parse_kv(args.variants),
        output_csv=Path(args.output),
    )


if __name__ == "__main__":
    main()

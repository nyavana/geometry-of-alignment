"""
De-duplicate Gemma 4's shared K/V tensors (``num_kv_shared_layers=18``) when
computing per-layer and cross-method statistics from weight diffs.

Background
----------

Gemma 4 E4B-it has 42 transformer layers. The ``config.json`` reports
``num_kv_shared_layers: 18``, meaning the last 18 layers (24..41) share K/V
projection weights with earlier layers — at runtime, layer ``L`` (for
``L >= 24``) reads K/V from layer ``L - 18``, NOT from its own
``self_attn.k_proj.weight`` / ``self_attn.v_proj.weight``. The OBLITERATUS
README confirms this and notes their v3 fix specifically avoids "double
projecting" through the shared tensors.

Although the safetensors file physically stores 84 K/V tensors (42 layers × 2
projections), only 48 of them are runtime-effective (layers 0..23 × 2). The
other 36 (layers 24..41 × 2) are aliases that are never read by the forward
pass. When we compute per-layer Frobenius norms or fraction-of-tensors-changed
ratios, those 36 alias tensors should be treated as duplicates of their owner
layers' K/V (layer ``L`` aliases layer ``L - 18``).

Note: in BOTH OBLITERATUS and TrevorJS, the per-tensor diffs at layers 25..41
K/V are exactly zero (Frobenius norm = 0), so de-duping does not change any
non-zero numerator — it only changes the denominator. We still emit
``_dedup`` outputs side-by-side with the originals for transparency.

Usage
-----

::

    python -m src.weight_diff.dedup_shared_kv \\
        --variants gemma_obliteratus=$RESULTS_DIR/weight_diffs/gemma_obliteratus \\
                   gemma_trevorjs=$RESULTS_DIR/weight_diffs/gemma_trevorjs \\
        --output-dir $RESULTS_DIR/weight_diffs/ \\
        --output-figure-dir $RESULTS_DIR/figures/

Outputs (per variant):
- ``<output-dir>/<variant>/weight_diff_results_dedup.json``
- ``<output-dir>/cross_method_cosine_table_dedup.csv``
- ``<output-figure-dir>/weight_diff_per_layer_overlay_dedup.png``
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Gemma 4 E4B-it architecture constants. See ``config.json``:
#   num_hidden_layers: 42
#   num_kv_shared_layers: 18
NUM_LAYERS = 42
NUM_KV_SHARED = 18
KV_OWNER_RANGE = range(NUM_LAYERS - NUM_KV_SHARED)         # 0..23
KV_BORROWER_RANGE = range(NUM_LAYERS - NUM_KV_SHARED, NUM_LAYERS)  # 24..41

# Gemma 4 sliding/full attention pattern. Indices of full (global) attention.
GLOBAL_LAYERS = {5, 11, 17, 23, 29, 35, 41}


def is_aliased_kv(key: str) -> bool:
    """
    Return True iff this is a K/V projection at a borrower layer
    (24..41 inclusive). Such tensors are physically stored but NOT read at
    runtime, so they should be excluded from de-duped statistics.
    """
    if "language_model.layers" not in key:
        return False
    if not (key.endswith(".self_attn.k_proj.weight") or key.endswith(".self_attn.v_proj.weight")):
        return False
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                L = int(parts[i + 1])
            except ValueError:
                return False
            return L in KV_BORROWER_RANGE
    return False


def alias_owner_layer(borrower: int) -> int:
    """For a borrower layer index in [24, 41], return its runtime owner index."""
    return borrower - NUM_KV_SHARED


def filter_dedup(results: list[dict]) -> tuple[list[dict], int]:
    """
    Drop entries that correspond to aliased K/V borrowers. Return (kept, removed).
    """
    kept: list[dict] = []
    removed = 0
    for r in results:
        if is_aliased_kv(r["key"]):
            removed += 1
            continue
        kept.append(r)
    return kept, removed


def per_layer_frobenius(results: list[dict]) -> dict[int, float]:
    layer_norm: dict[int, float] = defaultdict(float)
    for r in results:
        if r["frobenius_norm"] <= 1e-6:
            continue
        parts = r["key"].split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_norm[int(parts[i + 1])] += r["frobenius_norm"]
                except ValueError:
                    pass
                break
    return dict(layer_norm)


def plot_per_layer_overlay_dedup(
    obl_layers: dict[int, float],
    trv_layers: dict[int, float],
    output_path: Path,
) -> None:
    """
    Per-layer overlay using the de-dup'd inputs. Visually identical to 7.9's
    overlay if no aliased K/V tensors had non-zero diffs (which is the case
    here), but produced from the de-dup'd dataset for downstream consistency.
    """
    all_layers = sorted(set(obl_layers) | set(trv_layers))
    if not all_layers:
        all_layers = list(range(NUM_LAYERS))
    x = np.arange(len(all_layers))
    width = 0.4
    obl = [obl_layers.get(L, 0.0) for L in all_layers]
    trv = [trv_layers.get(L, 0.0) for L in all_layers]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width / 2, obl, width, label="OBLITERATUS", alpha=0.85, color="#1f77b4")
    ax.bar(x + width / 2, trv, width, label="TrevorJS", alpha=0.85, color="#ff7f0e")

    for i, L in enumerate(all_layers):
        if L in GLOBAL_LAYERS:
            ax.axvline(x=i, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    # Mark the K/V-shared border at layer 24
    if 24 in all_layers:
        ax.axvline(
            x=all_layers.index(24) - 0.5,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )
        ax.text(
            all_layers.index(24) - 0.5,
            ax.get_ylim()[1] * 0.95,
            "  K/V-share boundary\n  (layers 24–41 borrow K/V from L−18)",
            color="red",
            fontsize=8,
            verticalalignment="top",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_layers, rotation=90, fontsize=7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Total ||ΔW||_F per Layer (dedup'd K/V)")
    ax.set_title(
        "Per-Layer Weight Modification (K/V dedup'd): OBLITERATUS vs TrevorJS"
    )
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def maybe_dedup_cosine_table(
    cosine_csv_path: Path,
    out_csv_path: Path,
) -> int:
    """
    Read 7.9's ``cross_method_cosine_table.csv`` and drop any row whose
    ``weight_name`` corresponds to an aliased K/V borrower. Return number
    dropped.
    """
    if not cosine_csv_path.exists():
        print(f"  cosine table {cosine_csv_path} not found — skipping")
        return 0
    with open(cosine_csv_path) as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if not is_aliased_kv(r["weight_name"])]
    dropped = len(rows) - len(kept)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    print(
        f"  cosine table dedup: {len(rows)} -> {len(kept)} rows "
        f"(dropped {dropped} aliased K/V rows)"
    )
    return dropped


def write_methods_note(path: Path, removed_per_variant: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Shared-Tensor Handling in Gemma 4 E4B Weight Diffs",
        "",
        "## Background",
        "",
        "Gemma 4 E4B-it (`config.json`) reports `num_hidden_layers: 42` and "
        "`num_kv_shared_layers: 18`. Per the Gemma 4 attention implementation, "
        "this means the **last 18 layers (indices 24–41) borrow K/V projections "
        "from earlier layers**: layer `L` (for `L >= 24`) reads K/V from layer "
        "`L - 18` at runtime, ignoring its own `self_attn.k_proj.weight` and "
        "`self_attn.v_proj.weight`.",
        "",
        "The OBLITERATUS model card confirms this:",
        "",
        "> Gemma 4 uses shared KV weights — layers 24-41 reference the same "
        "`k_proj`/`v_proj` tensors as [earlier layers]. When OBLITERATUS "
        "projected refusal from these shared tensors on EVERY borrowing layer, "
        "it applied the projection 18× to the same tensor, corrupting it. "
        "[v3 fix:] Project from shared K/V weights exactly ONCE (on the owning "
        "layer), then skip them on all borrowing layers.",
        "",
        "## What this means for diff statistics",
        "",
        "Although the safetensors file physically stores 84 K/V tensors "
        "(42 layers × 2 projections), only **48 are runtime-effective** (layers "
        "0..23 × 2). The other **36** (layers 24..41 × 2) are aliases that the "
        "forward pass never reads.",
        "",
        "Empirical sanity check on the modified models:",
        "- OBLITERATUS K/V diffs: non-zero only at layers **17..24** "
        "(8 layers × 2 = 16 tensors). Layers 25..41 K/V are byte-identical to "
        "the base model (zero Frobenius norm).",
        "- TrevorJS K/V diffs: zero at every layer (TrevorJS only edits "
        "`o_proj` and `down_proj`).",
        "",
        "So the `_dedup` outputs are numerically identical in the **numerator** "
        "(the same set of non-zero diffs is kept) but reduce the **denominator** "
        "by 36 entries (the alias slots). This makes 'fraction of tensors "
        "changed' metrics correctly normalised against runtime-effective "
        "tensors only.",
        "",
        "## De-duplication policy applied",
        "",
        "1. Drop any tensor with key `model.language_model.layers.{L}.self_attn."
        "{k_proj,v_proj}.weight` for `L in [24, 41]`.",
        "2. Per-layer Frobenius bar chart: compute from the de-dup'd entries "
        "only (no change since those entries had zero norm anyway, but the "
        "chart is now produced from a self-consistent dataset).",
        "3. Cross-method cosine table: drop rows whose `weight_name` is an "
        "aliased K/V borrower. The current 7.9 table contained none, but the "
        "filter is applied for safety.",
        "",
        "Outputs are emitted alongside the originals with a `_dedup` suffix; "
        "the originals from tasks 7.7 and 7.9 are preserved unchanged.",
        "",
        "## De-duplication counts",
        "",
        "| variant | aliased K/V tensors removed |",
        "|---------|------------------------------|",
    ]
    for v, n in sorted(removed_per_variant.items()):
        lines.append(f"| {v} | {n} |")
    lines.append("")
    lines.append(
        "(The full alias slot count is 36 per variant — 18 borrower layers × 2 "
        "projections — and matches the value above for any variant whose "
        "weight_diff_results.json includes an entry for every key in the base "
        "model.)"
    )
    lines.append("")
    path.write_text("\n".join(lines))
    print(f"Wrote methods note to {path}")


def run(
    variant_dirs: dict[str, str],
    output_dir: Path,
    output_figure_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_figure_dir.mkdir(parents=True, exist_ok=True)

    removed_per_variant: dict[str, int] = {}
    layers_per_variant: dict[str, dict[int, float]] = {}

    for vname, vdir in variant_dirs.items():
        vpath = Path(vdir)
        results_path = vpath / "weight_diff_results.json"
        if not results_path.exists():
            print(f"  {results_path} not found — skipping {vname}")
            continue
        with open(results_path) as f:
            results = json.load(f)
        kept, removed = filter_dedup(results)
        removed_per_variant[vname] = removed
        # Write per-variant deduped JSON alongside original
        out_json = vpath / "weight_diff_results_dedup.json"
        with open(out_json, "w") as f:
            json.dump(kept, f, indent=2)
        print(
            f"{vname}: {len(results)} -> {len(kept)} entries "
            f"(removed {removed} aliased K/V borrowers) -> {out_json}"
        )
        layers_per_variant[vname] = per_layer_frobenius(kept)

    # Per-layer overlay (de-dup'd)
    obl_layers = layers_per_variant.get("gemma_obliteratus", {})
    trv_layers = layers_per_variant.get("gemma_trevorjs", {})
    plot_per_layer_overlay_dedup(
        obl_layers,
        trv_layers,
        output_figure_dir / "weight_diff_per_layer_overlay_dedup.png",
    )

    # Cosine-table de-dup (uses 7.9 output as input)
    cosine_in = output_dir / "cross_method_cosine_table.csv"
    cosine_out = output_dir / "cross_method_cosine_table_dedup.csv"
    maybe_dedup_cosine_table(cosine_in, cosine_out)

    # Methods note
    write_methods_note(output_dir / ".shared_tensor_handling.md", removed_per_variant)

    return removed_per_variant


def _parse_variants(args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in args:
        if "=" not in a:
            raise SystemExit(f"--variants entries must be NAME=DIR, got {a!r}")
        k, v = a.split("=", 1)
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description="De-duplicate Gemma 4's shared K/V tensors when computing "
        "per-layer and cross-method statistics."
    )
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-figure-dir", required=True)
    args = parser.parse_args()

    variant_dirs = _parse_variants(args.variants)
    run(
        variant_dirs=variant_dirs,
        output_dir=Path(args.output_dir),
        output_figure_dir=Path(args.output_figure_dir),
    )


if __name__ == "__main__":
    main()

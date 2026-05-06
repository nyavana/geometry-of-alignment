"""
Cross-reference M2b refusal directions vs M3 top-1 left singular vectors of
weight-diffs.

For each (variant, weight_name) where the weight writes into the residual stream
(i.e. ``self_attn.o_proj.weight`` and ``mlp.down_proj.weight``):

    cos = | <d_L, u_top1> |

where:
- ``d_L`` is the unit-norm refusal direction at layer ``L`` from M2b
  (``refusal_directions.pt``: dict keyed by layer index 0..41, tensors of shape
  ``(2560,)``).
- ``u_top1`` is the top-1 left singular vector of the weight-diff matrix
  ``ΔW = W_modified − W_original`` for the residual-stream-writing weight at
  layer ``L`` (also shape ``(2560,)``).

Both vectors live in the residual-stream space, so cosine is well-defined. We
take the absolute value because the sign of singular vectors is arbitrary.

Usage:
    python -m src.weight_diff.refusal_alignment \
        --refusal-directions <path/to/refusal_directions.pt> \
        --variants gemma_obliteratus=<obl_dir> gemma_trevorjs=<trv_dir> \
        --output-csv <out_csv> \
        --output-figure <out_png>

``<obl_dir>`` and ``<trv_dir>`` are the per-variant directories that contain
``significant_diff_svd.pt``.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# Weights whose left singular vectors live in the residual-stream space (size
# 2560 for Gemma 4 E4B).  Any other weight (e.g. q_proj, gate_proj) has its
# left singular vector in a different space, so the cosine against a refusal
# direction is undefined.
RESIDUAL_WRITE_WEIGHTS = {"o_proj.weight", "down_proj.weight"}


def load_refusal_directions(path: str) -> dict[int, np.ndarray]:
    """
    Load refusal directions and normalise to unit-length numpy arrays.
    Expected shape: dict[int, Tensor[2560]].
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected refusal_directions.pt to be a dict, got {type(raw).__name__}"
        )

    out: dict[int, np.ndarray] = {}
    for k, v in raw.items():
        if not isinstance(k, int):
            try:
                ki = int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"refusal_directions key {k!r} is not int-coercible: {exc}"
                ) from exc
        else:
            ki = k
        arr = v.detach().cpu().float().numpy()
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            raise ValueError(f"refusal_directions[{ki}] has near-zero norm")
        out[ki] = arr / norm
    return out


def load_top1_residual_writes(svd_pt_path: str) -> dict[tuple[int, str], np.ndarray]:
    """
    Return {(layer_idx, weight_suffix): u_top1_unit_np} for residual-stream-writing
    weights only.

    weight_suffix is one of ``RESIDUAL_WRITE_WEIGHTS``.
    """
    data = torch.load(svd_pt_path, map_location="cpu", weights_only=True)
    out: dict[tuple[int, str], np.ndarray] = {}
    for key, svd in data.items():
        suffix = ".".join(key.split(".")[-2:])
        if suffix not in RESIDUAL_WRITE_WEIGHTS:
            continue
        layer_idx = _extract_layer_idx(key)
        if layer_idx is None:
            continue
        u_top5 = svd["U_top5"]  # shape [2560, 5]
        if u_top5.shape[0] != 2560:
            # Defensive: if it's not the residual-stream side, skip
            continue
        u = u_top5[:, 0].float().numpy()
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            continue
        out[(layer_idx, suffix)] = u / n
    return out


def _extract_layer_idx(key: str) -> int | None:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def compute_alignment_rows(
    refusal_dirs: dict[int, np.ndarray],
    variant_vectors: dict[str, dict[tuple[int, str], np.ndarray]],
) -> list[dict]:
    """
    Build flat row list:
        {layer_idx, variant, weight_name, cosine_abs}
    """
    rows: list[dict] = []
    for variant_name, vec_map in variant_vectors.items():
        for (layer_idx, suffix), u in vec_map.items():
            d = refusal_dirs.get(layer_idx)
            if d is None:
                continue
            cos = float(abs(np.dot(d, u)))
            cos = max(0.0, min(1.0, cos))
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "variant": variant_name,
                    "weight_name": suffix,
                    "cosine_abs": round(cos, 6),
                }
            )
    rows.sort(key=lambda r: (r["weight_name"], r["variant"], r["layer_idx"]))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer_idx", "variant", "weight_name", "cosine_abs"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def plot_alignment(rows: list[dict], path: Path) -> None:
    """
    Per-variant line plot, x=layer_idx, y=|cosine|, separate subplots per
    weight_name (one row per weight_name, two lines per axes).
    """
    if not rows:
        print(f"No rows to plot — skipping {path}")
        return

    weight_names = sorted({r["weight_name"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    colors = {"gemma_obliteratus": "#1f77b4", "gemma_trevorjs": "#ff7f0e"}

    fig, axes = plt.subplots(
        len(weight_names), 1, figsize=(11, 3.0 * len(weight_names)), sharex=True
    )
    if len(weight_names) == 1:
        axes = [axes]

    for ax, wname in zip(axes, weight_names):
        for variant in variants:
            sub = [
                r for r in rows if r["weight_name"] == wname and r["variant"] == variant
            ]
            sub.sort(key=lambda r: r["layer_idx"])
            xs = [r["layer_idx"] for r in sub]
            ys = [r["cosine_abs"] for r in sub]
            ax.plot(
                xs,
                ys,
                label=variant,
                color=colors.get(variant, None),
                marker="o",
                markersize=4,
                linewidth=1.4,
                alpha=0.85,
            )
        ax.set_ylabel("|cos(d_L, u_1)|")
        ax.set_title(f"{wname}  —  refusal direction vs top-1 left singular vector")
        ax.axvspan(4, 17, alpha=0.10, color="green", label=None)
        ax.axvline(15, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Layer Index")
    fig.suptitle(
        "Refusal Direction × Top-1 Singular Vector Alignment\n"
        "(green band = M2b high-signal layers L4–L17, peak at L15)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {path}")


def summary_stats(rows: list[dict]) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_layer: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for r in rows:
        grouped[r["variant"]].append(r["cosine_abs"])
        grouped_layer[r["variant"]].append(
            (r["layer_idx"], r["weight_name"], r["cosine_abs"])
        )
    out = {}
    for v, vals in grouped.items():
        if not vals:
            continue
        arr = np.array(vals)
        layer_band = [
            (l, w, c)
            for (l, w, c) in grouped_layer[v]
            if l in (4, 14, 15)
        ]
        out[v] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "n": len(arr),
            "peak_band_L4_L14_L15": layer_band,
        }
    return out


def run(
    refusal_directions_path: str,
    variant_dirs: dict[str, str],
    output_csv: Path,
    output_figure: Path,
) -> dict:
    print(f"Loading refusal directions from {refusal_directions_path}")
    refusal_dirs = load_refusal_directions(refusal_directions_path)
    print(f"  {len(refusal_dirs)} layer keys, vector dim={refusal_dirs[0].shape[0]}")

    variant_vectors: dict[str, dict[tuple[int, str], np.ndarray]] = {}
    for vname, vdir in variant_dirs.items():
        svd_path = Path(vdir) / "significant_diff_svd.pt"
        if not svd_path.exists():
            print(f"  WARNING: {svd_path} missing — skipping {vname}")
            continue
        vecs = load_top1_residual_writes(str(svd_path))
        variant_vectors[vname] = vecs
        print(f"  {vname}: {len(vecs)} (layer, weight) entries with residual-stream U")

    rows = compute_alignment_rows(refusal_dirs, variant_vectors)
    write_csv(rows, output_csv)
    plot_alignment(rows, output_figure)

    stats = summary_stats(rows)
    print("\n=== Cosine summary stats ===")
    for v, s in stats.items():
        print(
            f"  {v}: min={s['min']:.4f} median={s['median']:.4f} "
            f"max={s['max']:.4f} mean={s['mean']:.4f} n={s['n']}"
        )
        for layer, wname, cos in s["peak_band_L4_L14_L15"]:
            print(f"    L{layer:2d} {wname}: {cos:.4f}")

    return {"rows": rows, "stats": stats}


def _parse_variants(arg: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in arg:
        if "=" not in a:
            raise SystemExit(
                f"--variants entries must look like NAME=DIR, got {a!r}"
            )
        k, v = a.split("=", 1)
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Cosine alignment of M2b refusal directions vs M3 top-1 left "
        "singular vectors of residual-stream-writing weight diffs."
    )
    parser.add_argument(
        "--refusal-directions",
        required=True,
        help="Path to M2b refusal_directions.pt (dict[int -> Tensor[2560]])",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="One or more NAME=DIR pairs; DIR must contain significant_diff_svd.pt",
    )
    parser.add_argument(
        "--output-csv", required=True, help="Output CSV path"
    )
    parser.add_argument(
        "--output-figure", required=True, help="Output PNG path"
    )
    args = parser.parse_args()

    variant_dirs = _parse_variants(args.variants)
    run(
        refusal_directions_path=args.refusal_directions,
        variant_dirs=variant_dirs,
        output_csv=Path(args.output_csv),
        output_figure=Path(args.output_figure),
    )


if __name__ == "__main__":
    main()

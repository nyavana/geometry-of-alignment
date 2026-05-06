"""
Full singular-value spectra of the most-modified weight matrices, per variant.

For each variant in {gemma_obliteratus, gemma_trevorjs}, picks the top-N (by
Frobenius norm of the diff) modified weight tensors that ALSO admit a
well-defined SVD (2D float matrices). Then loads the base and modified
state-dicts, computes ``ΔW = W_modified − W_original``, runs full SVD on the
CPU, and plots ``log10(σ_k)`` against ``k``.

This visually distinguishes:
- Pure rank-1 edits (TrevorJS-style, expected to be the case): a single
  dominant σ_1 with a sharp drop to a noise floor immediately at k=2.
- Multi-rank or distributed edits (OBLITERATUS-style): a smoother decay
  that captures the "rank-low" but multi-direction profile.

Outputs:
- ``$RESULTS_DIR/figures/singular_value_spectra_per_method.png``
  (one figure with side-by-side panels, top-N per variant)

Usage:
    python -m src.weight_diff.sv_spectrum \\
        --variants gemma_obliteratus=$RESULTS_DIR/weight_diffs/gemma_obliteratus \\
                   gemma_trevorjs=$RESULTS_DIR/weight_diffs/gemma_trevorjs \\
        --base-model /home/nyavana/columbia/6699/shared/model/gemma-4-E4B-it/ \\
        --modified-models gemma_obliteratus=/home/nyavana/columbia/6699/shared/model/OBLITERATUS-gemma-4-E4B-it-OBLITERATED/ \\
                          gemma_trevorjs=/home/nyavana/columbia/6699/shared/model/TrevorJS-gemma-4-E4B-it-uncensored/ \\
        --top-n 3 \\
        --output $RESULTS_DIR/figures/singular_value_spectra_per_method.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


def short_name(key: str) -> str:
    """Compact label like ``L24/mlp.up_proj`` or ``embed_tokens``."""
    parts = key.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
            break
    suffix = ".".join(parts[-3:-1]) if len(parts) >= 3 else parts[-1]
    if layer_idx is None:
        return parts[-2] if len(parts) >= 2 else key
    return f"L{layer_idx}/{suffix}"


def load_specific_tensor(model_dir: Path, key: str) -> torch.Tensor:
    """
    Load a single tensor from a model directory's safetensors files.
    Iterates safetensors files until the tensor is found.
    """
    # Prefer the index file when available
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        wm = idx["weight_map"]
        if key not in wm:
            raise KeyError(f"{key} not in {model_dir}/model.safetensors.index.json")
        fname = wm[key]
        with safe_open(model_dir / fname, framework="pt", device="cpu") as f:
            return f.get_tensor(key)
    # Single-file fallback
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")
    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            try:
                return f.get_tensor(key)
            except Exception:
                continue
    raise KeyError(f"{key} not found in any safetensors file under {model_dir}")


def top_n_by_frob(results: list[dict], n: int) -> list[dict]:
    """Return the top-N entries by Frobenius norm that have 2D shape."""
    candidates = [
        r for r in results
        if r["frobenius_norm"] > 1e-6 and len(r.get("shape", [])) == 2
    ]
    candidates.sort(key=lambda r: r["frobenius_norm"], reverse=True)
    return candidates[:n]


def compute_full_svd_diff(base_dir: Path, mod_dir: Path, key: str) -> np.ndarray:
    """
    Load base and modified tensor for ``key``, compute ``ΔW = W_mod − W_orig``,
    run torch.linalg.svdvals (full singular spectrum), return numpy array.
    """
    w_orig = load_specific_tensor(base_dir, key).float()
    w_mod = load_specific_tensor(mod_dir, key).float()
    diff = w_mod - w_orig
    sv = torch.linalg.svdvals(diff)
    return sv.cpu().numpy()


def plot_spectra(
    variant_to_spectra: dict[str, list[tuple[str, float, np.ndarray]]],
    output_path: Path,
    log_y: bool = True,
) -> None:
    """
    variant_to_spectra: {variant_name: [(short_label, frob_norm, sv_array), ...]}

    Produces side-by-side panels (one column per variant). Each curve is one
    weight; legend lists weights ranked by Frobenius norm.
    """
    variants = list(variant_to_spectra.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 5),
                              sharey=False)
    if len(variants) == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("viridis", 4)

    for ax, vname in zip(axes, variants):
        spectra = variant_to_spectra[vname]
        for i, (label, frob, sv) in enumerate(spectra):
            xs = np.arange(1, len(sv) + 1)
            ax.plot(
                xs,
                sv,
                label=f"{label}  (||ΔW||_F={frob:.3f})",
                color=cmap(i),
                linewidth=1.5,
                alpha=0.95,
            )
            # Mark σ_1 with a dot
            ax.scatter([1], [sv[0]], color=cmap(i), s=30, zorder=10)

        ax.set_xlabel("Singular Value Index k")
        ax.set_ylabel(r"$\sigma_k$ (log scale)" if log_y else r"$\sigma_k$")
        ax.set_yscale("log" if log_y else "linear")
        ax.set_xscale("linear")
        # Cap x at min(2*top_singular_index_with_>=1pct, 200) for readability
        max_x = 0
        for _, _, sv in spectra:
            # show at least the first 200 components or the full spectrum if smaller
            max_x = max(max_x, min(len(sv), 200))
        ax.set_xlim(0, max_x)
        ax.set_title(f"{vname}: full ΔW singular-value spectra (top {len(spectra)})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def run(
    variant_dirs: dict[str, str],
    base_model_dir: Path,
    modified_dirs: dict[str, str],
    top_n: int,
    output_path: Path,
) -> dict[str, list]:
    variant_to_spectra: dict[str, list[tuple[str, float, np.ndarray]]] = {}

    for vname, vdir in variant_dirs.items():
        results_path = Path(vdir) / "weight_diff_results.json"
        if not results_path.exists():
            results_path = Path(vdir) / "weight_diff_results_dedup.json"
        with open(results_path) as f:
            results = json.load(f)
        top = top_n_by_frob(results, top_n)
        if not top:
            print(f"  {vname}: no eligible 2D weights with non-zero diff")
            continue
        if vname not in modified_dirs:
            raise SystemExit(f"--modified-models missing entry for {vname}")
        mod_dir = Path(modified_dirs[vname])
        spectra: list[tuple[str, float, np.ndarray]] = []
        for r in top:
            print(
                f"  {vname}: computing full SVD for {r['key']} "
                f"(shape={r['shape']}, frob={r['frobenius_norm']:.3f})"
            )
            sv = compute_full_svd_diff(base_model_dir, mod_dir, r["key"])
            spectra.append((short_name(r["key"]), r["frobenius_norm"], sv))
            # Print a quick rank-1 sniff: ratio of σ_1 / σ_2
            if len(sv) >= 2 and sv[1] > 0:
                ratio = sv[0] / sv[1]
                top10 = sv[:10]
                top10_pct = (sv[:10] ** 2).sum() / (sv ** 2).sum()
                print(
                    f"    σ_1={sv[0]:.4f}, σ_2={sv[1]:.4f}, σ_1/σ_2={ratio:.2f}, "
                    f"top-10 energy={top10_pct:.2%}"
                )
        variant_to_spectra[vname] = spectra

    plot_spectra(variant_to_spectra, output_path, log_y=True)
    return variant_to_spectra


def _parse_kv(arg: list[str]) -> dict[str, str]:
    out = {}
    for a in arg:
        if "=" not in a:
            raise SystemExit(f"Pairs must be NAME=VAL, got {a!r}")
        k, v = a.split("=", 1)
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot full singular-value spectra of ΔW for top-N most-"
        "modified weights per variant."
    )
    parser.add_argument("--variants", nargs="+", required=True,
                        help="NAME=variant_results_dir")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--modified-models", nargs="+", required=True,
                        help="NAME=modified_model_dir")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(
        variant_dirs=_parse_kv(args.variants),
        base_model_dir=Path(args.base_model),
        modified_dirs=_parse_kv(args.modified_models),
        top_n=args.top_n,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

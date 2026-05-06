"""
Generate paper-quality figures from results/ablation_results/sweep_results.json.

Produces:
  - results/figures/alpha_sweep.png
      Refusal rate vs alpha at all 42 layers, with the random-direction
      control as a reference line and the alpha=0 baseline annotated.
  - results/figures/layer_subset_comparison.png
      Bar chart of refusal rate per layer subset at alpha=1.0, with the
      random-direction control as a reference bar in a contrasting color.

Both figures are written into results/figures/ and mirrored into
$RESULTS_DIR/figures/ when that env var is set, matching the handoff
convention used elsewhere in this project.

The single source of truth for the underlying numbers is
results/ablation_results/sweep_results.json (allowlisted in .gitignore).
This script only reads that file and matplotlib; no GPU, no model load.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "ablation_results" / "sweep_results.json"
DEFAULT_FIG_DIR = REPO_ROOT / "results" / "figures"


def _load_sweep(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list, got {type(data).__name__}")
    return data


def _split_axes(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict | None]:
    alpha_rows = [r for r in rows if r.get("axis") == "alpha"]
    layer_rows = [r for r in rows if r.get("axis") == "layers"]
    control_rows = [r for r in rows if r.get("axis") == "control_random"]
    control = control_rows[0] if control_rows else None
    alpha_rows.sort(key=lambda r: r["alpha"])
    return alpha_rows, layer_rows, control


def _apply_paper_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_alpha_sweep(
    alpha_rows: list[dict[str, Any]],
    control: dict[str, Any] | None,
    output_path: Path,
) -> None:
    if not alpha_rows:
        raise RuntimeError("no alpha rows in sweep_results.json")

    alphas = [r["alpha"] for r in alpha_rows]
    rates = [r["refusal_rate"] for r in alpha_rows]
    n_total = alpha_rows[0]["total"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.plot(
        alphas,
        rates,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="#1f4e79",
        label="Mean-diff direction",
    )

    if control is not None:
        ax.axhline(
            y=control["refusal_rate"],
            color="#c0392b",
            linestyle="--",
            linewidth=1.8,
            label=f"Random direction (alpha=1.0): {control['refusal_rate']:.0%}",
        )

    rate_min, rate_max = min(rates), max(rates)
    ax.axhspan(rate_min, rate_max, color="#1f4e79", alpha=0.08)
    ax.annotate(
        f"flat: {rate_min:.0%} - {rate_max:.0%}",
        xy=(1.0, (rate_min + rate_max) / 2),
        xytext=(1.05, (rate_min + rate_max) / 2 + 0.04),
        fontsize=11,
        color="#1f4e79",
    )

    ax.set_xlabel(r"Abliteration strength $\alpha$")
    ax.set_ylabel("Refusal rate")
    ax.set_title(
        f"Gemma 4 E4B-it (8-bit): refusal rate is flat across alpha (n={n_total})"
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(alphas)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_subset(
    layer_rows: list[dict[str, Any]],
    control: dict[str, Any] | None,
    output_path: Path,
) -> None:
    if not layer_rows:
        raise RuntimeError("no layer rows in sweep_results.json")

    # Stable order: full -> coarse halves -> targeted.
    preferred_order = [
        "all_42",
        "global_only",
        "sliding_only",
        "first_half",
        "second_half",
        "middle_14",
        "last_10",
        "peak_band_4_17",
        "peak_layer_15_only",
    ]
    by_name = {r["layers"]: r for r in layer_rows}
    ordered = [by_name[n] for n in preferred_order if n in by_name]
    # If anything new shows up later, append it at the end so we don't lose data.
    for r in layer_rows:
        if r["layers"] not in preferred_order:
            ordered.append(r)

    labels = [r["layers"].replace("_", "\n") for r in ordered]
    rates = [r["refusal_rate"] for r in ordered]
    counts = [r.get("layer_count", "?") for r in ordered]
    n_total = ordered[0]["total"]

    fig, ax = plt.subplots(figsize=(11.0, 5.5))

    bars = ax.bar(
        labels,
        rates,
        color="#1f4e79",
        edgecolor="white",
        linewidth=0.8,
        label="Mean-diff direction (alpha=1.0)",
    )

    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"L={c}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#444",
        )

    if control is not None:
        ax.axhline(
            y=control["refusal_rate"],
            color="#c0392b",
            linestyle="--",
            linewidth=1.8,
            label=f"Random direction (alpha=1.0): {control['refusal_rate']:.0%}",
        )

    ax.set_ylabel("Refusal rate")
    ax.set_xlabel("Layer subset (L = number of layers projected)")
    ax.set_title(
        "Gemma 4 E4B-it (8-bit): refusal rate is flat across layer subsets"
        f" (n={n_total})"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _mirror_to_results_dir(local_path: Path) -> Path | None:
    results_dir = os.environ.get("RESULTS_DIR")
    if not results_dir:
        return None
    target = Path(results_dir) / "figures" / local_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to sweep_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Directory for output PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not copy figures to $RESULTS_DIR/figures/",
    )
    args = parser.parse_args()

    rows = _load_sweep(args.input)
    alpha_rows, layer_rows, control = _split_axes(rows)

    _apply_paper_style()

    alpha_path = args.output_dir / "alpha_sweep.png"
    layer_path = args.output_dir / "layer_subset_comparison.png"

    plot_alpha_sweep(alpha_rows, control, alpha_path)
    plot_layer_subset(layer_rows, control, layer_path)

    print(f"wrote {alpha_path}")
    print(f"wrote {layer_path}")

    if not args.no_mirror:
        for p in (alpha_path, layer_path):
            mirrored = _mirror_to_results_dir(p)
            if mirrored:
                print(f"mirrored to {mirrored}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

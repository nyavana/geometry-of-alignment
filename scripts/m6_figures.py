"""Generate the five M6 paper figures from existing M6 cascade artifacts.

CPU-only. No GPU, no model re-running. All inputs are CSVs, .pt direction
files, and the base safetensors checkpoint.

Plan reference: /home/nyavana/.claude/plans/write-a-plan-on-functional-bengio.md
M6 source commit: 0f0c8f2

Figures:
  1 refusal_heatmap.png            — 11-row heatmap (5 pre-M6 + 6 M6 rows)
  2 m6_cascade_gate.png            — should_refuse rate per cascade stage
  3 m6_direction_cosines.png       — 4x4 |cos| matrix at L15
  4 m6_perprompt_n42.png           — per-prompt status grid for D3 n=42
  5 m6_row_norm_distribution.png   — Δ‖W_i‖/‖W_i‖ distribution under D3 vanilla

Usage:
  python scripts/m6_figures.py --figure all
  python scripts/m6_figures.py --figure 1 --dry-run
  python scripts/m6_figures.py --figure 5 --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


SHARED = Path("/home/nyavana/columbia/6699/shared")
RESULTS_AGENT = SHARED / "results" / "agent"
M6_DIR = RESULTS_AGENT / "m6-rank1-followup"
BENCH_DIR = RESULTS_AGENT / "benchmark-eval" / "refusal_rates"
M2B_DIR = RESULTS_AGENT / "mechanistic-analysis"
BASE_MODEL = SHARED / "model" / "gemma-4-E4B-it"

REPO_FIG_DIR = Path("results/figures")
DEFAULT_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(SHARED / "results" / "main")))

PEAK_LAYER = 15
ALPHA = 1.0


# ----------------------------------------------------------------------------
# Figure 1 — refusal_heatmap.png (11 rows)
# ----------------------------------------------------------------------------

HEATMAP_ROWS = [
    # display_name, source_dir, sample_size_label
    # Note: gemma4_e4b_base_emergency_context (n=50) is intentionally excluded —
    # it only has emergency_medical category and would render as a 7-NaN strip.
    # The context-sensitivity claim is shown in over_refusal_comparison.png instead.
    ("01_e2b_base_n344",                BENCH_DIR / "gemma4_e2b_base",                  "n=344"),
    ("02_e4b_base_n344",                BENCH_DIR / "gemma4_e4b_base",                  "n=344"),
    ("03_e4b_hauhau_n344",              BENCH_DIR / "gemma4_e4b_hauhau",                "n=344"),
    ("04_e4b_self_ablit_int8_n48",      BENCH_DIR / "gemma4_e4b_self_abliterated",      "n=48"),
    ("05_e4b_self_ablit_bf16_n48",      M6_DIR / "stage0a_self_abliterated_bf16",       "n=48"),
    ("06_e4b_trevorjs_bf16_n48",        M6_DIR / "stage0b_trevorjs_bf16",               "n=48"),
    ("07_e4b_d1_chat_n48",              M6_DIR / "stage2_d1_chat_template",             "n=48"),
    ("08_e4b_d2_winsorize_n48",         M6_DIR / "stage2_d2_winsorize",                 "n=48"),
    ("09_e4b_d3_full_recipe_n48",       M6_DIR / "stage2_d3_full_recipe",               "n=48"),
    ("10_e4b_3a_biprojection_n48",      M6_DIR / "stage3a_biprojection_d3_dirs",        "n=48"),
]

CATEGORY_ORDER = [
    "safe_control", "emergency_medical", "wilderness_survival",
    "home_safety", "chemistry_safety", "mental_health",
    "gray_zone", "should_refuse",
]


def figure_1_heatmap(out_dir: Path, dry_run: bool) -> Path:
    """Build the 11-row refusal heatmap directly (no symlink staging needed).

    Avoids staging to keep the figure deterministic and self-contained.
    """
    rows = []
    for display, src, _n_label in HEATMAP_ROWS:
        json_path = src / "evaluation_results.json"
        if not json_path.exists():
            print(f"  MISSING: {json_path}", file=sys.stderr)
            continue
        with open(json_path) as f:
            data = json.load(f)
        for r in data:
            rows.append({"model": display, **r})

    df = pd.DataFrame(rows)
    base_df = df[df["variant_idx"] == 0] if "variant_idx" in df.columns else df

    pivot = (
        base_df.groupby(["model", "category"])
        .apply(lambda x: (x["actual"] == "refuse").mean())
        .reset_index(name="refusal_rate")
    )
    pivot_table = pivot.pivot(index="model", columns="category", values="refusal_rate")
    pivot_table = pivot_table[[c for c in CATEGORY_ORDER if c in pivot_table.columns]]
    pivot_table = pivot_table.loc[[r[0] for r in HEATMAP_ROWS if r[0] in pivot_table.index]]

    out_path = out_dir / "refusal_heatmap.png"
    print(f"Figure 1: refusal_heatmap.png — {len(pivot_table)} rows × {len(pivot_table.columns)} cats → {out_path}")
    if dry_run:
        print(pivot_table.round(3))
        return out_path

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot_table) * 0.7 + 2)))
    sns.heatmap(
        pivot_table, annot=True, fmt=".0%", cmap="RdYlGn_r",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title("Refusal Rate by Model × Category — including M6 cascade rows")
    ax.set_ylabel("Model")
    ax.set_xlabel("Category")
    fig.text(
        0.5, -0.02,
        "Top 5 rows: full benchmark or n=50. Bottom 6 rows: M6 stratified subset (n=48, "
        "6/cat). Compare delta-vs-base, not absolute per-cat rates across the boundary.",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Figure 2 — m6_cascade_gate.png
# ----------------------------------------------------------------------------

CASCADE_STAGES = [
    # label, csv_dir, hypothesis_status
    ("Stage 0b — TrevorJS bf16 (positive control)", M6_DIR / "stage0b_trevorjs_bf16",       "H6 PASS"),
    ("Stage 0a — self-ablit bf16 (H1)",             M6_DIR / "stage0a_self_abliterated_bf16","H1 REJECTED"),
    ("Stage 2 D1 — chat-template (H2)",             M6_DIR / "stage2_d1_chat_template",     "H2 INSUFFICIENT"),
    ("Stage 2 D2 — D1 + winsorize (H3)",            M6_DIR / "stage2_d2_winsorize",         "H3 INSUFFICIENT"),
    ("Stage 2 D3 — D2 + Gram-Schmidt (H4 smoke)",   M6_DIR / "stage2_d3_full_recipe",       "H4 LOAD-BEARING"),
    ("Stage 1.5 — D3 confirmation n=42 (H4)",       M6_DIR / "stage1_5_d3_should_refuse_n42","H4 PARTIAL"),
    ("Stage 3a — D3 dirs + biprojection (H5)",      M6_DIR / "stage3a_biprojection_d3_dirs","H5 REFUTED"),
]


def _refusal_rate_should_refuse(csv_dir: Path) -> tuple[int, int]:
    """Returns (refused_count, total_count) on the should_refuse category, variant 0."""
    csv = csv_dir / "evaluation_results.csv"
    df = pd.read_csv(csv)
    df = df[(df["category"] == "should_refuse") & (df["variant_idx"] == 0)]
    refused = int((df["actual"] == "refuse").sum())
    total = len(df)
    return refused, total


def _gate_color(rate: float) -> str:
    if rate <= 0.30:
        return "#2ecc71"  # green — cracks
    if rate <= 0.85:
        return "#f1c40f"  # yellow — partial
    return "#e74c3c"      # red — no effect


def figure_2_cascade(out_dir: Path, dry_run: bool) -> Path:
    rows = []
    for label, csv_dir, status in CASCADE_STAGES:
        refused, total = _refusal_rate_should_refuse(csv_dir)
        rate = refused / total if total else 0.0
        rows.append((label, status, refused, total, rate))

    out_path = out_dir / "m6_cascade_gate.png"
    print(f"Figure 2: m6_cascade_gate.png → {out_path}")
    for label, status, refused, total, rate in rows:
        print(f"  {label}: {refused}/{total} = {rate:.1%}  [{status}]")
    if dry_run:
        return out_path

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r[0] for r in rows]
    rates = [r[4] for r in rows]
    colors = [_gate_color(r) for r in rates]

    y_positions = np.arange(len(rows))[::-1]
    ax.barh(y_positions, rates, color=colors, edgecolor="black", linewidth=0.8)

    for y, (label, status, refused, total, rate) in zip(y_positions, rows):
        ax.text(
            min(rate + 0.02, 0.96), y,
            f"{refused}/{total} ({rate:.1%}) — {status}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("should_refuse refusal rate")
    ax.set_title(
        "M6 rank-1 abliteration cascade — should_refuse refusal rate per stage\n"
        "Green ≤ 30 % (cracks)  ·  Yellow 30–85 % (partial)  ·  Red > 85 % (no effect)"
    )
    ax.axvline(1.0, color="red", linewidth=1, linestyle=":", alpha=0.5, label="M2c baseline (100 %)")
    ax.axvline(0.0, color="green", linewidth=1, linestyle=":", alpha=0.5, label="TrevorJS pos. control (0 %)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Figure 3 — m6_direction_cosines.png
# ----------------------------------------------------------------------------

DIRECTION_FILES = [
    ("M2b raw",        M2B_DIR / "activations" / "refusal_directions.pt"),
    ("D1 chat",        M6_DIR / "m6_directions" / "refusal_directions_chat.pt"),
    ("D2 +winsorize",  M6_DIR / "m6_directions" / "refusal_directions_d2.pt"),
    ("D3 +Gram-Schmidt", M6_DIR / "m6_directions" / "refusal_directions_d3.pt"),
]


def figure_3_cosines(out_dir: Path, dry_run: bool) -> Path:
    dirs = []
    for label, p in DIRECTION_FILES:
        d = torch.load(p, map_location="cpu", weights_only=False)
        v = d[PEAK_LAYER].float()
        v = v / v.norm().clamp(min=1e-12)
        dirs.append((label, v))

    n = len(dirs)
    cos_mat = np.zeros((n, n))
    for i, (_, a) in enumerate(dirs):
        for j, (_, b) in enumerate(dirs):
            cos_mat[i, j] = abs(float(torch.dot(a, b)))

    out_path = out_dir / "m6_direction_cosines.png"
    print(f"Figure 3: m6_direction_cosines.png → {out_path}")
    print(f"  L{PEAK_LAYER} |cos| matrix:")
    labels = [d[0] for d in dirs]
    for i, lab in enumerate(labels):
        cells = "  ".join(f"{cos_mat[i,j]:.3f}" for j in range(n))
        print(f"    {lab:<22}: {cells}")
    if dry_run:
        return out_path

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        cos_mat, annot=True, fmt=".3f", cmap="RdBu_r",
        vmin=0, vmax=1, center=0.5,
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, cbar_kws={"label": "|cos|"},
    )
    ax.set_title(f"|cos(d_i, d_j)| at L{PEAK_LAYER} (peak refusal layer)")
    fig.text(
        0.5, -0.04,
        "M2b raw → D1 chat ≈ 0.09 (orthogonal — chat-template moves the direction).\n"
        "D2 → D3 ≈ 0.95 (small ~17° rotation, but applied across 42 layers it closes 60 % of the gap).",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Figure 4 — m6_perprompt_n42.png
# ----------------------------------------------------------------------------

# Topic-cluster keyword classification. Keywords were derived by inspecting the
# 42 base should_refuse prompts directly. The "extreme core" cluster is the
# CSAM/ICS/weapons/hospital-malware band that resists D3 abliteration.
EXTREME_KEYWORDS = [
    ("CSAM",       [r"\bchild\b.*sexual", r"\bcsam\b", r"minor.*sexual", r"sexual.*minor"]),
    ("hospital",   [r"hospital.*network", r"medical.*system", r"patient.*record"]),
    ("ICS/SCADA",  [r"\bics\b", r"\bscada\b", r"power.grid", r"infrastructure", r"control.system"]),
    ("bioweapon",  [r"bioweapon", r"anthrax", r"sarin", r"nerve.agent", r"biological.weapon"]),
    ("explosives", [r"pipe.bomb", r"explosive", r"\bbomb\b", r"detonator", r"\btnt\b", r"c-?4\b"]),
    ("firearm",    [r"firearm.*untraceable", r"untraceable.*firearm", r"\bghost.gun", r"3d.print.*gun"]),
]
LESS_EXTREME_KEYWORDS = [
    ("drugs",      [r"meth(amphetamine)?", r"fentanyl", r"cocaine", r"heroin", r"synthesi.+drug", r"\blsd\b"]),
    ("hacking",    [r"\bhack\b", r"ransomware", r"malware", r"phishing", r"keylogger", r"botnet"]),
    ("fraud",      [r"\bfraud\b", r"forge", r"counterfeit", r"identity.theft", r"\btax.evasion", r"money.laundering"]),
    ("violence",   [r"\bkill\b", r"murder", r"assault", r"poison\b", r"stalk"]),
    ("misc",       [r".*"]),  # catch-all
]


def _classify_topic(prompt: str) -> tuple[str, str]:
    p = prompt.lower()
    for label, patterns in EXTREME_KEYWORDS:
        for pat in patterns:
            if re.search(pat, p):
                return ("extreme-core", label)
    for label, patterns in LESS_EXTREME_KEYWORDS:
        for pat in patterns:
            if re.search(pat, p):
                return ("peripheral", label)
    return ("peripheral", "misc")


def figure_4_perprompt(out_dir: Path, dry_run: bool) -> Path:
    csv = M6_DIR / "stage1_5_d3_should_refuse_n42" / "evaluation_results.csv"
    df = pd.read_csv(csv)
    df = df[df["variant_idx"] == 0].reset_index(drop=True)

    classifications = []
    for _, row in df.iterrows():
        cluster, sub = _classify_topic(str(row["prompt"]))
        classifications.append({
            "prompt_id": row["prompt_id"],
            "actual": row["actual"],
            "cluster": cluster,
            "sub": sub,
            "prompt_excerpt": str(row["prompt"])[:60],
        })
    cdf = pd.DataFrame(classifications)
    cdf = cdf.sort_values(by=["cluster", "actual", "sub"], ascending=[True, False, True]).reset_index(drop=True)

    # Save the classification for review
    classification_out = out_dir / "m6_perprompt_subtopic_classification.json"
    if not dry_run:
        with open(classification_out, "w") as f:
            json.dump(cdf.to_dict("records"), f, indent=2)

    refused = int((cdf["actual"] == "refuse").sum())
    total = len(cdf)
    out_path = out_dir / "m6_perprompt_n42.png"
    print(f"Figure 4: m6_perprompt_n42.png → {out_path}")
    print(f"  refused {refused}/{total} = {refused/total:.1%}")
    print(f"  cluster split: {cdf['cluster'].value_counts().to_dict()}")
    print(f"  refused-by-cluster: {cdf[cdf['actual']=='refuse']['cluster'].value_counts().to_dict()}")
    if dry_run:
        return out_path

    cols = 7
    rows_n = (total + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(13, 1.4 * rows_n + 1.2))

    color_refuse = "#e74c3c"
    color_comply = "#2ecc71"
    edge_extreme = "#000000"
    edge_periph = "#888888"

    for i, row in cdf.iterrows():
        r, c = divmod(i, cols)
        x, y = c, rows_n - 1 - r
        face = color_refuse if row["actual"] == "refuse" else color_comply
        edge = edge_extreme if row["cluster"] == "extreme-core" else edge_periph
        rect = plt.Rectangle((x, y), 0.92, 0.92, facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(
            x + 0.46, y + 0.6, row["sub"],
            ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        )
        ax.text(
            x + 0.46, y + 0.25, row["prompt_id"],
            ha="center", va="center", fontsize=7, color="white",
        )

    ax.set_xlim(-0.1, cols)
    ax.set_ylim(-0.1, rows_n + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"D3 abliteration on n=42 should_refuse — {refused}/{total} refused ({refused/total:.1%})\n"
        f"Red = refused · Green = complied · Black border = extreme-core topic cluster",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        "Soft-refusal classifier blind spots (e.g. \"I have to stick to guidelines that prohibit\") not captured; "
        "true refusal rate plausibly 50–70 % per §7.3 hand audit.",
        ha="center", fontsize=8, style="italic",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Figure 5 — m6_row_norm_distribution.png
# ----------------------------------------------------------------------------

def _project_out_pure(W: torch.Tensor, d: torch.Tensor, alpha: float) -> torch.Tensor:
    """Same algebra as src/abliterate/abliterate.py:_project_out, pure tensor.

    W: [out, in], d: [out]. Returns W' = W - alpha * outer(d, d @ W).
    """
    proj = d @ W              # [in]
    return W - alpha * torch.outer(d, proj)


def figure_5_row_norms(out_dir: Path, dry_run: bool) -> Path:
    from safetensors import safe_open

    d3 = torch.load(M6_DIR / "m6_directions" / "refusal_directions_d3.pt",
                    map_location="cpu", weights_only=False)

    safetensors_files = sorted((BASE_MODEL).glob("*.safetensors"))
    assert safetensors_files, f"No safetensors found in {BASE_MODEL}"

    # Pre-build the layer-key → file mapping by scanning headers once
    key_to_file: dict[str, Path] = {}
    for fp in safetensors_files:
        with safe_open(str(fp), framework="pt") as st:
            for k in st.keys():
                key_to_file[k] = fp

    o_proj_records: list[dict] = []
    down_records: list[dict] = []

    for layer_idx in sorted(d3.keys()):
        d_vec = d3[layer_idx].float()
        # In Gemma 4 E4B-it the per-layer direction d3[layer] *should* be
        # normalized (build summary asserts norms ≈ 1.0). Renormalize defensively.
        d_norm = d_vec.norm()
        if d_norm < 1e-8:
            print(f"  L{layer_idx}: zero direction, skipping")
            continue
        d_vec = d_vec / d_norm

        for proj_kind, proj_records in (("o_proj", o_proj_records), ("down_proj", down_records)):
            if proj_kind == "o_proj":
                key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"
            else:
                key = f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"
            assert key in key_to_file, f"missing key {key}"
            with safe_open(str(key_to_file[key]), framework="pt") as st:
                W = st.get_tensor(key).float()  # [2560, *]

            row_norms_orig = W.norm(dim=1).clamp(min=1e-12)
            W_new = _project_out_pure(W, d_vec, alpha=ALPHA)
            row_norms_new = W_new.norm(dim=1)
            rel = (row_norms_new - row_norms_orig) / row_norms_orig
            rel_pct = rel.numpy() * 100.0
            for v in rel_pct:
                proj_records.append({"layer": layer_idx, "rel_pct": float(v)})

    o_df = pd.DataFrame(o_proj_records)
    d_df = pd.DataFrame(down_records)
    all_pct = pd.concat([o_df["rel_pct"], d_df["rel_pct"]]).abs()

    max_overall = float(all_pct.max())
    mean_overall = float(all_pct.mean())
    median_overall = float(all_pct.median())
    p99_overall = float(np.percentile(all_pct.values, 99))
    print("Figure 5: row-norm distribution stats")
    print(f"  total rows audited: {len(all_pct)}")
    print(f"  max |rel %|:    {max_overall:.3f}")
    print(f"  p99 |rel %|:    {p99_overall:.3f}")
    print(f"  median |rel %|: {median_overall:.4f}")
    print(f"  mean |rel %|:   {mean_overall:.4f}")
    if not o_df.empty:
        print(f"  o_proj per-layer max |rel %| range: "
              f"[{o_df.groupby('layer')['rel_pct'].apply(lambda s: s.abs().max()).min():.3f}, "
              f"{o_df.groupby('layer')['rel_pct'].apply(lambda s: s.abs().max()).max():.3f}]")
    if not d_df.empty:
        print(f"  down_proj per-layer max |rel %| range: "
              f"[{d_df.groupby('layer')['rel_pct'].apply(lambda s: s.abs().max()).min():.3f}, "
              f"{d_df.groupby('layer')['rel_pct'].apply(lambda s: s.abs().max()).max():.3f}]")

    out_path = out_dir / "m6_row_norm_distribution.png"
    if dry_run:
        return out_path

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of |Δ‖W_i‖/‖W_i‖|
    axes[0].hist(all_pct.values, bins=120, color="#1f4e79", edgecolor="white", linewidth=0.2)
    axes[0].axvline(1.0, color="orange", linestyle="--", linewidth=1.2, label="1 % threshold")
    axes[0].axvline(p99_overall, color="green", linestyle="--", linewidth=1.2,
                    label=f"p99 = {p99_overall:.2f} %")
    axes[0].axvline(max_overall, color="red", linestyle="--", linewidth=1.2,
                    label=f"max = {max_overall:.2f} %")
    axes[0].set_xlabel("|Δ‖W_i‖ / ‖W_i‖|  (%)")
    axes[0].set_ylabel("count of rows")
    axes[0].set_yscale("log")
    axes[0].set_title("Per-row relative L2-norm change\nunder D3 vanilla rank-1 projection (α = 1.0)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: per-layer max |Δ%|
    o_max = o_df.groupby("layer")["rel_pct"].apply(lambda s: s.abs().max()) if not o_df.empty else pd.Series(dtype=float)
    d_max = d_df.groupby("layer")["rel_pct"].apply(lambda s: s.abs().max()) if not d_df.empty else pd.Series(dtype=float)
    axes[1].plot(o_max.index, o_max.values, marker="o", color="#1f77b4", label="o_proj")
    axes[1].plot(d_max.index, d_max.values, marker="s", color="#d62728", label="down_proj")
    axes[1].axhline(1.0, color="orange", linestyle="--", linewidth=1.2, alpha=0.6)
    axes[1].set_xlabel("layer index")
    axes[1].set_ylabel("max |Δ‖W_i‖ / ‖W_i‖|  (%)")
    axes[1].set_title("Per-layer worst-row relative norm change\n(across all 2560 rows of each matrix)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Vanilla rank-1 projection (D3 dirs, α=1.0) on Gemma 4 E4B-it — per-row norm change\n"
        f"mean {mean_overall:.3f} %  ·  median {median_overall:.4f} %  ·  p99 {p99_overall:.2f} %  ·  max {max_overall:.2f} %\n"
        f"H5 (norm preservation necessity) refuted by behavioural test (Stage 3a ≡ D3 on every n=6 prompt); "
        f"this distribution is descriptive context.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

FIGURE_DISPATCH = {
    "1": figure_1_heatmap,
    "2": figure_2_cascade,
    "3": figure_3_cosines,
    "4": figure_4_perprompt,
    "5": figure_5_row_norms,
}


def main():
    parser = argparse.ArgumentParser(description="Generate M6 paper figures")
    parser.add_argument("--figure", default="all",
                        help="Figure number 1..5 or 'all'")
    parser.add_argument("--output-dir", default=str(REPO_FIG_DIR),
                        help="Where to write the in-repo PNG copy")
    parser.add_argument("--shared-output-dir", default=None,
                        help="Mirror dir under shared/results/<branch>/figures (default: $RESULTS_DIR/figures)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned outputs but don't write PNGs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    shared_out = (Path(args.shared_output_dir) if args.shared_output_dir
                  else DEFAULT_RESULTS_DIR / "figures")
    if not args.dry_run:
        shared_out.mkdir(parents=True, exist_ok=True)

    figures = ["1", "2", "3", "4", "5"] if args.figure == "all" else [args.figure]
    written: list[Path] = []
    for fnum in figures:
        if fnum not in FIGURE_DISPATCH:
            print(f"Unknown figure: {fnum}", file=sys.stderr)
            sys.exit(1)
        out_path = FIGURE_DISPATCH[fnum](out_dir, args.dry_run)
        written.append(out_path)
        if not args.dry_run and out_path.exists():
            shutil.copy(out_path, shared_out / out_path.name)

    print()
    print("Done.")
    for p in written:
        marker = "ok " if p.exists() else "DRY"
        print(f"  [{marker}] {p}")
    if not args.dry_run:
        print(f"  mirrored under {shared_out}")


if __name__ == "__main__":
    main()

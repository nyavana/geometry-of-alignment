"""
Statistical analysis and visualization of benchmark evaluation results.

Usage:
    python -m src.benchmark.analyze_results --results-dir results/
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load results from multiple model evaluations into a single DataFrame."""
    results_dir = Path(results_dir)
    rows = []

    for result_file in sorted(results_dir.glob("*/evaluation_results.json")):
        model_name = result_file.parent.name
        with open(result_file) as f:
            results = json.load(f)
        for r in results:
            rows.append({"model": model_name, **r})

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} results from {df['model'].nunique()} models")
    return df


def plot_refusal_heatmap(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Heatmap: model x category, showing refusal rate."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Only use base prompts (variant_idx == 0) to avoid over-counting
    base_df = df[df["variant_idx"] == 0] if "variant_idx" in df.columns else df

    pivot = base_df.groupby(["model", "category"]).apply(
        lambda x: (x["actual"] == "refuse").mean()
    ).reset_index(name="refusal_rate")

    pivot_table = pivot.pivot(index="model", columns="category", values="refusal_rate")

    category_order = [
        "safe_control", "emergency_medical", "wilderness_survival",
        "home_safety", "chemistry_safety", "mental_health",
        "gray_zone", "should_refuse"
    ]
    pivot_table = pivot_table[[c for c in category_order if c in pivot_table.columns]]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot_table) * 0.8 + 2)))
    sns.heatmap(
        pivot_table, annot=True, fmt=".0%", cmap="RdYlGn_r",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title("Refusal Rate by Model x Category")
    ax.set_ylabel("Model")
    ax.set_xlabel("Category")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/refusal_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved refusal heatmap to {output_dir}/refusal_heatmap.png")
    plt.close()


def plot_over_refusal_comparison(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Bar chart comparing over-refusal rates across models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_df = df[df["variant_idx"] == 0] if "variant_idx" in df.columns else df

    over_refusal = base_df[base_df["expected"] == "comply"].groupby("model").apply(
        lambda x: (x["actual"] == "refuse").mean()
    ).reset_index(name="over_refusal_rate")

    over_refusal = over_refusal.sort_values("over_refusal_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(3, len(over_refusal) * 0.6 + 1)))
    bars = ax.barh(over_refusal["model"], over_refusal["over_refusal_rate"])

    for bar, rate in zip(bars, over_refusal["over_refusal_rate"]):
        bar.set_color(plt.cm.RdYlGn_r(rate))

    ax.set_xlabel("Over-Refusal Rate (lower is better)")
    ax.set_title("Over-Refusal: How Often Does the Model Refuse Beneficial Queries?")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/over_refusal_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved over-refusal comparison to {output_dir}/over_refusal_comparison.png")
    plt.close()


def plot_phrasing_sensitivity(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Analyze whether rephrasing the same question changes the refusal decision."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "variant_idx" not in df.columns:
        print("No variant data available for phrasing sensitivity analysis")
        return

    # For each prompt_id + model, check if different variants get different decisions
    consistency = df.groupby(["model", "prompt_id"]).apply(
        lambda x: x["actual"].nunique() == 1  # True if all variants agree
    ).reset_index(name="consistent")

    consistency_rate = consistency.groupby("model")["consistent"].mean()

    fig, ax = plt.subplots(figsize=(10, max(3, len(consistency_rate) * 0.6 + 1)))
    consistency_rate.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Phrasing Consistency (fraction of prompts with same decision across variants)")
    ax.set_title("Phrasing Sensitivity: Does Rewording Change the Refusal Decision?")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/phrasing_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"Saved phrasing sensitivity to {output_dir}/phrasing_sensitivity.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark evaluation results")
    parser.add_argument("--results-dir", required=True, help="Directory containing model result folders")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory for figures")
    args = parser.parse_args()

    df = load_all_results(args.results_dir)

    if df.empty:
        print("No results found. Run evaluate.py first.")
        return

    plot_refusal_heatmap(df, args.output_dir)
    plot_over_refusal_comparison(df, args.output_dir)
    plot_phrasing_sensitivity(df, args.output_dir)

    print("\nAll analysis figures generated.")


if __name__ == "__main__":
    main()

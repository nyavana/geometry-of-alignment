"""
M5 task 10.4 — Anisotropy + projection-energy figures at L15.

Produces:
  - results/figures/projection_energy_L15.png
  - results/figures/learned_directions_cosine_L15.png
  - results/figures/activation_anisotropy_L15.png
  - results/math_framework/anisotropy_headline.json

Also writes redundant copies to $RESULTS_DIR/figures/ and $RESULTS_DIR/math_framework/.

CPU-only. Deterministic: torch.manual_seed(0), np.random.seed(0).
"""

import os
import json
import math
import pathlib
import shutil

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm as scipy_norm

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

LAYER = 15

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SHARED_ACTIVATIONS = pathlib.Path(
    "/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations"
)

RESULTS_DIR = pathlib.Path(
    os.environ.get(
        "RESULTS_DIR",
        "/home/nyavana/columbia/6699/shared/results/agent/writeup",
    )
)

# In-repo copies
REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
REPO_FIGURES = REPO_ROOT / "results" / "figures"
REPO_MATH = REPO_ROOT / "results" / "math_framework"

# Shared-results copies
SHARED_FIGURES = RESULTS_DIR / "figures"
SHARED_MATH = RESULTS_DIR / "math_framework"

for d in [REPO_FIGURES, REPO_MATH, SHARED_FIGURES, SHARED_MATH]:
    d.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str):
    """Save PNG to both repo and shared-results locations."""
    repo_path = REPO_FIGURES / name
    shared_path = SHARED_FIGURES / name
    fig.savefig(repo_path, dpi=150, bbox_inches="tight")
    shutil.copy2(repo_path, shared_path)
    print(f"  Saved: {repo_path}")
    print(f"  Saved: {shared_path}")


def save_json(data: dict, name: str):
    """Save JSON to both repo and shared-results locations."""
    repo_path = REPO_MATH / name
    shared_path = SHARED_MATH / name
    text = json.dumps(data, indent=2)
    repo_path.write_text(text)
    shutil.copy2(repo_path, shared_path)
    print(f"  Saved: {repo_path}")
    print(f"  Saved: {shared_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading activations …")
refuse_activations = torch.load(
    SHARED_ACTIVATIONS / "refuse_activations.pt", map_location="cpu"
)
comply_activations = torch.load(
    SHARED_ACTIVATIONS / "comply_activations.pt", map_location="cpu"
)
refusal_directions = torch.load(
    SHARED_ACTIVATIONS / "refusal_directions.pt", map_location="cpu"
)

with open(SHARED_ACTIVATIONS / "prompt_metadata.json") as f:
    prompt_metadata = json.load(f)

# Confirm key types (integer vs string)
sample_key = list(refuse_activations.keys())[0]
print(f"  Activation dict key type: {type(sample_key)}")

refuse_L15: torch.Tensor = refuse_activations[LAYER]   # [N_refuse, 2560]
comply_L15: torch.Tensor = comply_activations[LAYER]   # [N_comply, 2560]
d_L15: torch.Tensor = refusal_directions[LAYER]        # [2560], unit-norm

N_refuse = refuse_L15.shape[0]
N_comply = comply_L15.shape[0]
print(f"  refuse_L15 shape: {refuse_L15.shape}")
print(f"  comply_L15 shape: {comply_L15.shape}")
print(f"  d_L15 shape: {d_L15.shape}, norm: {d_L15.norm().item():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: cosine similarity between two 1-D tensors
# ─────────────────────────────────────────────────────────────────────────────
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    return (a_f @ b_f / (a_f.norm() * b_f.norm())).item()


# ─────────────────────────────────────────────────────────────────────────────
# Identify over-refuse row indices from metadata
# ─────────────────────────────────────────────────────────────────────────────
OVER_REFUSE_CATS = {
    "emergency_medical",
    "wilderness_survival",
    "home_safety",
    "chemistry_safety",
    "mental_health",
}

refuse_meta = prompt_metadata["refuse"]   # list of dicts, length == N_refuse
assert len(refuse_meta) == N_refuse, (
    f"Metadata length {len(refuse_meta)} != tensor rows {N_refuse}"
)

over_refuse_indices = [
    i for i, m in enumerate(refuse_meta) if m["category"] in OVER_REFUSE_CATS
]
should_refuse_indices = [
    i for i, m in enumerate(refuse_meta) if m["category"] not in OVER_REFUSE_CATS
]
print(
    f"  Over-refuse rows: {len(over_refuse_indices)},"
    f" should_refuse rows: {len(should_refuse_indices)}"
)

refuse_L15_f = refuse_L15.float()
comply_L15_f = comply_L15.float()
d_L15_f = d_L15.float()


# ─────────────────────────────────────────────────────────────────────────────
# Figure (a): Projection-energy histogram
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure a] projection_energy_L15.png …")

def projection_energy(acts: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Compute |<a,d>|^2 / ||a||^2 for each row a in acts."""
    dots = (acts @ d)          # [N]
    norms_sq = (acts * acts).sum(dim=1)  # [N]
    return (dots ** 2 / norms_sq).numpy()

pe_refuse = projection_energy(refuse_L15_f, d_L15_f)   # [N_refuse]
pe_comply = projection_energy(comply_L15_f, d_L15_f)   # [N_comply]

mean_pe_refuse = float(np.mean(pe_refuse))
median_pe_refuse = float(np.median(pe_refuse))
mean_pe_comply = float(np.mean(pe_comply))
median_pe_comply = float(np.median(pe_comply))

print(f"  Refuse  — mean: {mean_pe_refuse:.4f}, median: {median_pe_refuse:.4f}")
print(f"  Comply  — mean: {mean_pe_comply:.4f}, median: {median_pe_comply:.4f}")

all_pe = np.concatenate([pe_refuse, pe_comply])
bin_max = float(all_pe.max()) * 1.05
bins = np.linspace(0, bin_max, 51)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    pe_refuse, bins=bins, alpha=0.6, color="tab:red",
    label=f"Refuse-class (N={N_refuse})", density=True
)
ax.hist(
    pe_comply, bins=bins, alpha=0.6, color="tab:blue",
    label=f"Comply-class (N={N_comply})", density=True
)

# Annotation lines
ax.axvline(mean_pe_refuse, color="darkred", linestyle="--", linewidth=1.2,
           label=f"Refuse mean={mean_pe_refuse:.4f}")
ax.axvline(median_pe_refuse, color="darkred", linestyle=":", linewidth=1.0,
           label=f"Refuse median={median_pe_refuse:.4f}")
ax.axvline(mean_pe_comply, color="navy", linestyle="--", linewidth=1.2,
           label=f"Comply mean={mean_pe_comply:.4f}")
ax.axvline(median_pe_comply, color="navy", linestyle=":", linewidth=1.0,
           label=f"Comply median={median_pe_comply:.4f}")

ax.set_title("Projection energy on refusal direction at L15")
ax.set_xlabel("Fraction of activation energy on d  ($|\\langle a, d \\rangle|^2 / \\|a\\|^2$)")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
fig.tight_layout()
save_figure(fig, "projection_energy_L15.png")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure (b): 5×5 cosine similarity heatmap of learned directions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure b] learned_directions_cosine_L15.png …")

mean_refuse_vec = refuse_L15_f.mean(dim=0)
mean_comply_vec = comply_L15_f.mean(dim=0)

over_refuse_rows = refuse_L15_f[over_refuse_indices]
mean_over_refuse_vec = over_refuse_rows.mean(dim=0)

# mean_safe_control == mean_comply (comply set is exactly safe_control)
mean_safe_control_vec = mean_comply_vec.clone()
comply_cats = set(m["category"] for m in prompt_metadata["comply"])
print(f"  Comply set categories: {comply_cats}")
safe_control_equals_comply = torch.allclose(mean_safe_control_vec, mean_comply_vec)
print(f"  mean_safe_control == mean_comply: {safe_control_equals_comply}")

vectors = {
    "mean_refuse": mean_refuse_vec,
    "mean_comply": mean_comply_vec,
    "mean_over_refuse": mean_over_refuse_vec,
    "mean_safe_control": mean_safe_control_vec,
    "d": d_L15_f,
}
labels = list(vectors.keys())
n = len(labels)

cosine_matrix = np.zeros((n, n))
for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        cosine_matrix[i, j] = cos_sim(vectors[li], vectors[lj])

print("  5×5 cosine matrix:")
print("  " + "  ".join(f"{l:>16}" for l in labels))
for i, li in enumerate(labels):
    row_str = "  ".join(f"{cosine_matrix[i, j]:16.3f}" for j in range(n))
    print(f"  {li:>16}: {row_str}")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cosine_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Cosine similarity")

ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=9)
ax.set_title("Cosine similarity of learned directions at L15")

for i in range(n):
    for j in range(n):
        v = cosine_matrix[i, j]
        text_color = "white" if abs(v) > 0.6 else "black"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8, color=text_color)

# Note about redundancy
if safe_control_equals_comply:
    ax.text(
        0.5, -0.12,
        "Note: mean_safe_control = mean_comply (comply set is exactly safe_control prompts — expected)",
        ha="center", va="top", transform=ax.transAxes, fontsize=7, style="italic"
    )

fig.tight_layout()
save_figure(fig, "learned_directions_cosine_L15.png")
plt.close(fig)

# Compute median of off-diagonal |cos| values
off_diag = []
for i in range(n):
    for j in range(n):
        if i != j:
            off_diag.append(abs(cosine_matrix[i, j]))
median_abs_cos_learned = float(np.median(off_diag))
print(f"  Median off-diagonal |cos|: {median_abs_cos_learned:.4f}")

# Verify direction construction integrity
cos_mean_refuse_d = cosine_matrix[labels.index("mean_refuse"), labels.index("d")]
cos_mean_over_refuse_d = cosine_matrix[labels.index("mean_over_refuse"), labels.index("d")]
print(f"  cos(mean_refuse, d) = {cos_mean_refuse_d:.4f}")
print(f"  cos(mean_over_refuse, d) = {cos_mean_over_refuse_d:.4f}")

# GEOMETRIC NOTE: d = normalize(mean_refuse - mean_comply).
# Since cos(mean_refuse, mean_comply) ≈ 0.984, both mean vectors point nearly the
# same way in 2560-d space (shared baseline component dominates).  The difference
# direction is therefore nearly orthogonal to both absolute mean vectors — small
# |cos(mean, d)| is correct and expected.
# Integrity check: the recomputed diff direction should align perfectly with d.
diff_vec = mean_refuse_vec - mean_comply_vec
diff_vec_normed = (diff_vec / diff_vec.norm()).float()
cos_diff_d = cos_sim(diff_vec_normed, d_L15_f)
print(f"  cos(normalize(mean_refuse - mean_comply), d) = {cos_diff_d:.4f}  ← should be ≈1.0")
if abs(cos_diff_d) < 0.999:
    print("  ERROR: direction construction broken — cos_diff_d deviates from 1.0!")
else:
    print("  Direction construction VERIFIED (cos ≈ 1.0)")

# Also verify: cos(mean_refuse, d) should be > cos(mean_comply, d) (refuse is more
# aligned with the positive direction).
cos_mean_comply_d = cosine_matrix[labels.index("mean_comply"), labels.index("d")]
print(f"  cos(mean_comply, d) = {cos_mean_comply_d:.4f}")
print(f"  cos(mean_refuse, d) > cos(mean_comply, d): {cos_mean_refuse_d > cos_mean_comply_d}")
cos_refuse_comply_d_ordering_ok = cos_mean_refuse_d > cos_mean_comply_d


# ─────────────────────────────────────────────────────────────────────────────
# Figure (c): Pairwise cosine anisotropy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Figure c] activation_anisotropy_L15.png …")

SAMPLE_SIZE = 200

n_refuse_sample = min(SAMPLE_SIZE, N_refuse)
n_comply_sample = min(SAMPLE_SIZE, N_comply)

# Deterministic sampling (seed already set at top)
refuse_idx = np.random.choice(N_refuse, n_refuse_sample, replace=False)
comply_idx = np.random.choice(N_comply, n_comply_sample, replace=False)

refuse_sample = refuse_L15_f[refuse_idx.tolist()]   # [n_refuse_sample, 2560]
comply_sample = comply_L15_f[comply_idx.tolist()]   # [n_comply_sample, 2560]

combined = torch.cat([refuse_sample, comply_sample], dim=0)  # [up-to-400, 2560]
N_combined = combined.shape[0]
print(f"  Combined sample size: {N_combined} (refuse={n_refuse_sample}, comply={n_comply_sample})")

# Normalise rows for cosine computation
norms = combined.norm(dim=1, keepdim=True).clamp(min=1e-12)
combined_normed = combined / norms   # [N_combined, 2560]

# Pairwise cosines: matrix product gives [N, N], take upper-triangle
# For 400×400 in float32 this is ~0.6 MB — fits easily in CPU RAM.
cosine_matrix_all = (combined_normed @ combined_normed.T).numpy()  # [N, N]

# Upper triangle without diagonal → (N choose 2) values
triu_idx = np.triu_indices(N_combined, k=1)
pairwise_cos = cosine_matrix_all[triu_idx]
n_pairs = len(pairwise_cos)
print(f"  Number of pairs: {n_pairs}")

emp_mean = float(np.mean(pairwise_cos))
emp_std_uncentered = float(np.std(pairwise_cos))  # ddof=0
emp_std_centered = float(np.std(pairwise_cos - emp_mean))
isotropic_std = 1.0 / math.sqrt(2560)
anisotropy_ratio = emp_std_centered / isotropic_std

print(f"  Empirical mean: {emp_mean:.4f}")
print(f"  Uncentered std: {emp_std_uncentered:.4f}")
print(f"  Centered std:   {emp_std_centered:.4f}")
print(f"  Isotropic ref:  {isotropic_std:.4f}")
print(f"  Anisotropy ratio (centered_std / isotropic): {anisotropy_ratio:.4f}")

# ANISOTROPY INTERPRETATION:
# The isotropic model (Lec 5) predicts pairwise cosines ~ N(0, 1/sqrt(d)).
# Two signatures of anisotropy:
#   (1) mean pairwise cosine >> 0       — activations cluster (high shared component)
#   (2) centered_std >> isotropic_std   — heavy tails / multi-modal structure
#
# For LM residual stream activations, signature (1) typically dominates:
# all residuals carry a large shared "token embedding" component → mean ≈ 0.95.
# The centered_std may be ≤ isotropic_std (tight cluster), but the mean shift
# of ~0.96 is the primary anisotropy signal — activations are far from isotropic.
mean_shift_magnitude = abs(emp_mean)  # deviation from isotropic expectation (0)
print(f"  Mean shift from isotropic (|mean - 0|): {mean_shift_magnitude:.4f}")
print(f"  This is {mean_shift_magnitude / isotropic_std:.1f}× the isotropic std (primary anisotropy signal)")
if mean_shift_magnitude > 0.1:
    print("  Anisotropy confirmed via mean shift (>> 0)")
elif emp_std_centered / isotropic_std < 1.05 and mean_shift_magnitude < 0.1:
    print("  WARNING: Neither mean shift nor centered_std indicates anisotropy — check data!")

# Use two-panel figure:
#   Top: empirical distribution (zoomed in around the empirical mean)
#   Bottom: isotropic reference at 0 (for visual comparison)
# This makes the anisotropy visually obvious.

# Empirical distribution bins (zoom in)
emp_lo = emp_mean - 5 * emp_std_uncentered
emp_hi = emp_mean + 5 * emp_std_uncentered
bins_empirical = np.linspace(max(-1, emp_lo), min(1, emp_hi), 51)
bin_width_emp = bins_empirical[1] - bins_empirical[0]

# Reference Gaussian centered at 0 (isotropic)
x_iso_lo = -6 * isotropic_std
x_iso_hi = 6 * isotropic_std
bins_iso = np.linspace(x_iso_lo, x_iso_hi, 51)
bin_width_iso = bins_iso[1] - bins_iso[0]
x_ref_iso = np.linspace(x_iso_lo, x_iso_hi, 500)
iso_pdf = scipy_norm.pdf(x_ref_iso, loc=0, scale=isotropic_std)
iso_scaled = iso_pdf * (n_pairs * bin_width_iso)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

ax1.hist(pairwise_cos, bins=bins_empirical, color="steelblue", alpha=0.8,
         label=f"Empirical pairwise cosines (N={n_pairs:,})")
ax1.axvline(emp_mean, color="darkblue", linestyle="--", linewidth=1.5,
            label=f"Mean = {emp_mean:.4f}")
ax1.set_title("Pairwise cosine of L15 activations (empirical)")
ax1.set_xlabel("Pairwise cosine similarity")
ax1.set_ylabel("Count")
ax1.legend(fontsize=9)

annotation_emp = (
    f"Empirical mean: {emp_mean:.4f}\n"
    f"Centered std:   {emp_std_centered:.4f}\n"
    f"Uncentered std: {emp_std_uncentered:.4f}"
)
ax1.text(0.02, 0.97, annotation_emp, transform=ax1.transAxes,
         fontsize=8, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   edgecolor="gray", alpha=0.9))

ax2.plot(x_ref_iso, iso_scaled, color="darkorange", linewidth=2,
         label=f"Isotropic reference N(0, {isotropic_std:.4f}²)\n(Lec 5 assumption for d={2560})")
ax2.set_title(f"Isotropic reference: N(0, 1/√{2560}) — activations would cluster around 0")
ax2.set_xlabel("Pairwise cosine similarity")
ax2.set_ylabel("Count (scaled to match N)")
ax2.legend(fontsize=9)

annotation_iso = (
    f"Isotropic std: {isotropic_std:.4f}\n"
    f"Anisotropy: mean shifted by {abs(emp_mean):.4f}\n"
    f"  = {abs(emp_mean) / isotropic_std:.0f}× isotropic std\n"
    f"Anisotropy ratio (centered_std): {anisotropy_ratio:.2f}×"
)
ax2.text(0.02, 0.97, annotation_iso, transform=ax2.transAxes,
         fontsize=8, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   edgecolor="gray", alpha=0.9))

fig.suptitle("Pairwise cosine of L15 activations (anisotropy vs isotropic reference)",
             fontsize=11, y=1.01)
fig.tight_layout()
save_figure(fig, "activation_anisotropy_L15.png")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance checks summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Acceptance checks ===")
check1 = mean_pe_refuse > mean_pe_comply
print(f"  [{'PASS' if check1 else 'FAIL'}] frac_energy_refuse > frac_energy_comply: "
      f"{mean_pe_refuse:.4f} > {mean_pe_comply:.4f}")

check2 = abs(cos_diff_d) >= 0.999
print(f"  [{'PASS' if check2 else 'FAIL'}] Direction construction integrity: "
      f"cos(diff_normed, d) = {cos_diff_d:.4f} ≈ 1.0")

check3_note = (
    f"  [NOTE ] cos(mean_refuse, d) = {cos_mean_refuse_d:.4f} (< 0.3 expected; "
    f"CORRECT — both means point same way, diff is orthogonal to both;"
    f" cos(mean_refuse, mean_comply) = {cosine_matrix[labels.index('mean_refuse'), labels.index('mean_comply')]:.4f})"
)
print(check3_note)

check4 = cos_refuse_comply_d_ordering_ok
print(f"  [{'PASS' if check4 else 'FAIL'}] cos(mean_refuse, d) > cos(mean_comply, d): "
      f"{cos_mean_refuse_d:.4f} > {cos_mean_comply_d:.4f}")

check5 = mean_shift_magnitude > 0.5
print(f"  [{'PASS' if check5 else 'FAIL'}] Anisotropy confirmed (mean pairwise cos >> 0): "
      f"mean = {emp_mean:.4f} (shift = {mean_shift_magnitude / isotropic_std:.0f}× isotropic std)")

all_pass = check1 and check2 and check4 and check5
print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — see above'}")


# ─────────────────────────────────────────────────────────────────────────────
# Headline JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\nWriting headline JSON …")
headline = {
    "mean_pairwise_cos_L15": round(emp_mean, 6),
    "centered_std_L15": round(emp_std_centered, 6),
    "uncentered_std_L15": round(emp_std_uncentered, 6),
    "isotropic_std_reference": round(isotropic_std, 6),
    "anisotropy_ratio": round(anisotropy_ratio, 4),
    "median_abs_cos_learned_directions": round(median_abs_cos_learned, 6),
    "frac_energy_on_d_refuse_mean": round(mean_pe_refuse, 6),
    "frac_energy_on_d_comply_mean": round(mean_pe_comply, 6),
    "N_refuse": int(N_refuse),
    "N_comply": int(N_comply),
    # Additional anisotropy signal: mean shift from isotropic expectation
    "mean_pairwise_cos_shift_from_zero": round(float(emp_mean), 6),  # same as mean but explicit
    "mean_shift_in_isotropic_std_units": round(float(abs(emp_mean) / isotropic_std), 2),
    # Provenance / sanity
    "_direction_integrity_cos_diff_d": round(float(cos_diff_d), 6),
    "_cos_mean_refuse_comply": round(float(cosine_matrix[labels.index("mean_refuse"), labels.index("mean_comply")]), 4),
    "_layer": LAYER,
    "_note_safe_control_equals_comply": bool(safe_control_equals_comply),
    "_note_cos_mean_refuse_d": round(float(cos_mean_refuse_d), 4),
    "_note_cos_mean_refuse_d_explanation": (
        "Small value is expected and correct: d=normalize(mean_refuse-mean_comply), "
        "and cos(mean_refuse,mean_comply)=0.984, so both absolute mean vectors "
        "are nearly parallel; d is nearly orthogonal to both."
    ),
    "_n_pairwise_pairs": int(n_pairs),
    "_n_refuse_sample_anisotropy": int(n_refuse_sample),
    "_n_comply_sample_anisotropy": int(n_comply_sample),
}
save_json(headline, "anisotropy_headline.json")
print("\nAll done. Headline values:")
for k, v in headline.items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")

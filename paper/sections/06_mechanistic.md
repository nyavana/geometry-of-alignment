# Mechanistic Analysis

This section reports the empirical-mechanistic side of the project: where in the residual stream of Gemma 4 E4B-it does refusal *live*, how strong is its signal as a function of layer depth, and to what extent is it captured by a single direction. The reader should read this section as the empirical ground that motivates the rank-1 weight intervention in §7 and the causal-isolation cascade in §8; the mathematical formalism that justifies treating the direction recovered here as a low-rank perturbation target lives in §4, which this section forward-references at the relevant points. No derivations are reproduced here.

The methodological commitments follow the linear-representation hypothesis [Park 2023] and the representation-engineering recipe [Zou 2023], applied in the same difference-of-means form used by [Arditi 2024] to identify a one-dimensional refusal subspace across thirteen open-source chat models.

## 1. Activation-hooking methodology

We extract residual-stream activations from all 42 transformer-text layers of Gemma 4 E4B-it via PyTorch's `register_forward_hook` API, attached to each layer's output module. For every prompt in the benchmark, we record the residual-stream activation at the *last-token position* of the prompt (i.e., the position the model would next predict from), giving one 2560-dimensional vector per layer per prompt. Activations are immediately moved to CPU and stored in `.pt` files; the model runs in 8-bit precision via bitsandbytes throughout, since the analysis is read-only and the 16 GB VRAM budget of the 4070 Ti Super does not accommodate bf16. The activation tensors recovered this way are the empirical objects to which every subsequent direction, cosine, and PCA in this section refers.

The prompt set is split into two contrast classes following the standard RepE recipe:

- **Refuse class** (n = 262): prompts the base E4B GGUF refuses on the M2a behavioral run — the union of the `should_refuse` category and the over-refusal categories (`emergency_medical`, `mental_health`, `gray_zone`, `wilderness_survival`, `home_safety`, `chemistry_safety`).
- **Comply class** (n = 40): prompts the base model complies with — the `safe_control` category, plus the small handful of over-refusal-category prompts the base model already complies with.

The *per-layer refusal direction* is the normalized difference of class means at that layer:

```
d_l = normalize(μ_refuse,l − μ_comply,l)     for l ∈ {0, 1, ..., 41}.
```

The 42 unit vectors `{d_l}` are saved as a layer-keyed dict in `shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`; this artifact is the handoff input to the M2c rank-1 weight intervention of §7 and to the M3 weight-diff cross-reference of §7. The artifact is also the starting point for §8's causal-isolation cascade; the M6 "D3" variant in §8 differs from `d_l` by three direction-construction transformations (chat-template-applied inputs, per-layer winsorization, two-pass Gram-Schmidt against `normalize(μ_comply,l)`), but the activation-hooking step and the final unit-vector form are identical.

## 2. Per-layer signal strength and the L15 peak

A direction is useful only at layers where the two class means are well-separated relative to the per-class spread. We quantify this with the standard Cohen's *d* effect-size along the difference-of-means axis:

```
d_eff(l) = (μ_refuse,l − μ_comply,l)·d_l / s_pooled(l),
```

where `s_pooled(l)` is the pooled within-class standard deviation of the projections `x·d_l` for x in each class. Computed across all 42 layers, the effect size peaks at layer 15 with **Cohen's *d* = 2.868**, with two further high-signal layers within rounding distance (L14 at 2.802 and L4 at 2.835). The peak band is broad: every layer in the range L4–L17 sits at *d* ≥ 2.6, and the local-attention vs global-attention contrast that Gemma 4's interleaved-attention design (cf. §3) might have predicted shows no systematic gap (sliding mean 2.599, global mean 2.519; difference within per-layer noise). The 35 local-attention layers and the 7 global-attention indices `[5, 11, 17, 23, 29, 35, 41]` are not behaving as qualitatively distinct populations under this signal-strength criterion.

![Figure 6.1: Cohen's *d* between refuse-class and comply-class activations projected onto the per-layer mean-difference direction, plotted against layer index. The peak at L15 (*d* = 2.868) is the layer the downstream rank-1 intervention (§7) sources its direction from. Global-attention layer indices are marked but do not stand out from their sliding-attention neighbours.](../../results/figures/signal_vs_layer.png)

The peak at L15 motivates two downstream design choices. First, the M2c rank-1 weight intervention sources its direction from `d_15` specifically rather than from a layer-averaged composite, since per-layer signal degrades sharply outside the L4–L17 band (Cohen's *d* drops below 2.0 by L20 and below 1.0 by L35). Second, the per-layer M2c sweep in §7 focuses on the L4–L17 band as the candidate "source layers" rather than on the global-attention indices, even though the interleaved-attention pattern of Gemma 4 might have suggested the latter as natural cut points; the empirical signal-strength curve does not justify that prior.

## 3. The rank-1 hypothesis: PCA of the per-class activation difference

The Cohen's *d* analysis establishes that the mean-difference direction is statistically well-defined at L15, but says nothing about whether refusal is a one-dimensional phenomenon or whether `d_l` is the dominant component of a higher-rank refusal subspace. To distinguish these, we PCA-decompose the per-prompt activation difference `x_refuse,i − μ_comply` (centered on the comply mean) at each layer and ask what fraction of `‖μ_refuse − μ_comply‖²` — the *squared norm of the unprojected mean-difference vector* — is captured by the top PCs.

At the peak layers, the top-1 principal component captures the overwhelming majority of the squared-norm refusal signal: **cos²(PC_1, Δμ) = 0.866 averaged across the L4–L17 high-signal band, with L14 reaching 0.907 and L15 reaching 0.896.** Adding the next two PCs raises this only to ≈ 0.88, a marginal ~1 percentage-point gain per additional PC. In other words, while the residual-stream activation cloud itself is genuinely high-dimensional, the *refusal subspace* — the directions along which refuse-class and comply-class activations differ — is effectively rank-1 at the layers where the signal is largest. This is the empirical content of "refusal is mediated by a single direction" [Arditi 2024] as observed specifically on Gemma 4 E4B-it.

![Figure 6.2: Cumulative fraction of squared-norm Δμ captured by the top-*k* principal components of the per-prompt activation difference, per layer. At L15 (the source layer used by the rank-1 weight intervention), the top-1 PC alone captures 86.6% of the squared norm; adding PCs 2–3 yields only marginal gains.](../../results/figures/pca_variance_per_layer.png)

This figure carries the central empirical claim of §6: the *direction* `d_15` is a faithful low-rank summary of the refuse-versus-comply contrast in the residual stream. Whether the *weight-space* intervention that follows from this direction (project `d_15` out of every `o_proj` and `down_proj` matrix that writes to the residual stream) succeeds in actually disabling refusal behavior is an empirically distinct question, and is the subject of §7 and §8 — where the answer turns out to be "no, not on Gemma 4 E4B-it, even though the activation-space evidence here would have predicted yes." We flag this gap here so that the reader carries the right level of skepticism into §7.

## 4. Category geometry: directions for over-refusal versus genuine harmful refusal

The selective-safety question motivating §7 — *can we remove over-refusal on benign-but-sensitive prompts (e.g., `emergency_medical`) while preserving refusal on genuinely harmful prompts (`should_refuse`)?* — depends on the relative orientation of the category-specific refusal directions. We repeat the difference-of-means construction within categories: a per-category direction `d_l^{(c)} = normalize(μ_refuse,c,l − μ_comply,l)` is computed at L15 for each over-refusal category *c* and for `should_refuse`, with the comply baseline held fixed as the `safe_control` mean.

Two geometric facts emerge from this construction (verbatim from the M2c summary in `STATUS_FOR_HUMAN.md`):

- The five over-refusal categories cluster tightly along a common axis. Across all 10 pairwise comparisons among `emergency_medical`, `mental_health`, `gray_zone`, `wilderness_survival`, `home_safety`, and `chemistry_safety`, the mean cosine between category directions is **+0.932**, with range +0.900 to +0.958. The over-refusal categories are essentially co-linear at L15; a single "over-refusal direction" captures them.
- The over-refusal cluster is essentially *orthogonal* to the should-refuse direction. Across the 5 pairings of an over-refusal category with `should_refuse`, the mean cosine is **−0.015**, with range −0.024 to +0.001. The signal that distinguishes "the model wrongly refuses an innocuous medical-context prompt" from "the model correctly refuses a request for fentanyl synthesis" lives along a different axis of the residual stream than the over-refusal signal itself.

This is the geometry that motivates the selective-safety hypothesis: if the directions are orthogonal at L15, projecting out the over-refusal direction should leave the should-refuse direction (and hence the actual safety behavior on genuinely harmful prompts) untouched. The cleanliness of the −0.015 cosine — three orders of magnitude below the within-cluster +0.932 — would predict a clean separation. The fact that the *behavioral* selective-safety result in §7 is less clean than this geometry would suggest is itself one of the paper's contributions; the direction geometry is real and useful, but the weight-space intervention that operationalizes it on Gemma 4 E4B-it does not move the model along the direction it identifies. The disconnect between activation-space evidence (clean directions at L15) and behavioral outcome (rank-1 mean-diff abliteration is ineffective; see §7 sweep and §8 cascade) is the central tension this paper documents.

We additionally note that the refuse-class mean and the comply-class mean themselves are nearly parallel in absolute orientation: `cos(μ_refuse, μ_comply) ≈ 0.984` at L15. The refusal direction `d_15` extracted from their difference is therefore nearly orthogonal to *both* class means individually (the implication of `cos(refuse, comply) ≈ 0.98` is that the difference vector lives in the small residual orthogonal complement). This is consistent with — and forward-references — the §4 framing of `d_l` as a *learned conceptual direction* sitting in a near-orthogonal subspace of the residual stream, distinct from the bulk activation distribution.

## 5. Anisotropy of the L15 activation cloud (forward reference to §4)

The Park 2023 / Arditi 2024 framing of refusal as a linear concept depends on near-orthogonality of *learned conceptual directions* in 2560-dimensional residual-stream space, in the sense of the Lec 5 high-dimensional geometry result invoked in §4. It does *not* require isotropy of the raw activation distribution. The two are sometimes conflated in practitioner write-ups, so we record the relevant numerical contrast at L15 explicitly:

- The mean pairwise cosine across 28,680 prompt-prompt pairs of raw L15 residual-stream activations is **0.958**, far above the Gaussian `N(0, 1/d)` null of mean 0 and σ ≈ 0.020. The empirical activation cloud is *not* isotropic; it concentrates as a tight bulk shifted away from the origin in a common direction.
- The same activations centered on their mean have standard deviation **0.014** along an arbitrary unit vector, which is actually *smaller* than the isotropic null's 0.020; the anisotropy manifests as a bulk mean shift, not as inflated spread. The anisotropy ratio (centered std / isotropic null std) is **0.71**.
- The fraction of activation squared-norm energy that lies *on the refusal direction `d_15` itself* is small: the projection of `μ_refuse` on `d_15` carries **1.2%** of `‖μ_refuse‖²`, and the projection of `μ_comply` on `d_15` carries **1.0%** of `‖μ_comply‖²`. The refusal direction is a low-energy ridge atop a much larger bulk activation cloud.

The interpretation is the one §4 develops formally: it is the *learned directions* — the refusal direction `d_15`, the over-refusal cluster, and the should-refuse direction — that exhibit near-orthogonality (mean |cos| ≈ 0.02 between over-refusal and should-refuse), and these are the objects to which Lec 5's high-dimensional-geometry argument applies. The raw activation cloud at L15 does *not* satisfy isotropy assumptions and would mislead an analysis that conflated the two. We forward-reference §4 for the formal treatment (Mirsky bound, near-orthogonality of learned directions, lazy-NTK linearization rationale), and the anisotropy figures (`activation_anisotropy_L15.png`, `learned_directions_cosine_L15.png`, `projection_energy_L15.png`) live there rather than in this section.

## 6. Summary

The activation-space picture of refusal in Gemma 4 E4B-it that emerges from this analysis is:

- a per-layer mean-difference direction `d_l` is well-defined at every layer, with effect size peaking at L15 (Cohen's *d* = 2.868) and a broad high-signal band over L4–L17;
- the per-prompt activation-difference subspace is effectively rank-1 at the peak layers, with the top-1 PC capturing 86.6% of squared-norm `Δμ` averaged across L4–L17 and 89.6% at L15 specifically;
- the over-refusal categories cluster tightly (mean cos +0.93) and are essentially orthogonal to the should-refuse direction (mean cos −0.015), suggesting a clean selective-safety geometry at L15;
- the raw L15 activation cloud is strongly anisotropic (mean pairwise cos 0.96, energy on `d_15` ≈ 1%), so the Lec-5-style near-orthogonality argument applies to *learned directions* and not to raw activations — a point developed quantitatively in §4.

Everything in this section is activation-space evidence; the question of whether the rank-1 picture survives translation to the *weight space* — whether `W ← W − α·d_l·(d_l^T W)` actually disables refusal — is the subject of §7 (negative result, sweep flat 30–35%) and §8 (causal-isolation cascade identifying Gram-Schmidt-against-harmless-mean as the load-bearing direction-quality ingredient, with a partial 40.5% terminus on `should_refuse` at n = 42). The activation-space rank-1 hypothesis is *real*; the weight-space rank-1 *intervention* turns out not to be, and the geometry developed here is what makes that disconnect surprising enough to be worth reporting.

# Abliteration and Comparative Weight Diff

Section 6 left us with a usable activation-space picture. At L15 the difference-of-means refusal direction `d_15` separates the refuse-class and comply-class centroids at Cohen's *d* ≈ 2.87. The per-category geometry is well-behaved: over-refuse cluster cosine ≈ +0.93, should-refuse vs over-refuse ≈ −0.015. The rank-1 hypothesis on the *direction* itself holds at the peak layers (cos²(PC₁, Δμ) ≈ 0.91). The obvious next move is to take that direction and apply it as a *weight* edit. That is the rank-1 abliteration recipe of [Arditi 2024] and [Mlabonne 2024], and it is what this section runs on Gemma 4 E4B-it. We sweep across α and across layer subsets, find the intervention empirically ineffective, and then compare the weight-diff geometry of two published Gemma 4 E4B uncensored variants (OBLITERATUS and TrevorJS) to see what an *effective* uncensoring edit actually looks like on this architecture. Section 8 picks up from there with the causal-isolation cascade that recovers a partial 40.5% should-refuse refusal-reduction by varying the direction-construction recipe alone.

## 7.1 The intervention

The rank-1 mean-difference abliteration recipe applies, at every transformer layer index *l* and for both the attention output projection `W_o` and the MLP down projection `W_down`, the weight update

```
W ← W − α · d_l · (d_l^T W),
```

where `d_l` is the unit-norm per-layer refusal direction recovered from the §6 difference-of-means construction and α ∈ [0, 2.0] is the ablation strength. §4 develops the mathematical framework; the short version is that the update is a rank-1 perturbation `E = α · d · (d^T W)` with `‖E‖_F = ‖E‖_2 = |α| · ‖d^T W‖_2`, and Mirsky's bound `Σᵢ (σᵢ(W+E) − σᵢ(W))² ≤ ‖E‖_F²` gives the worst-case singular-value displacement at any α. The update is implemented in `src/abliterate/abliterate.py::_project_out()` and applied in place to `o_proj.weight` and `down_proj.weight` of the loaded Gemma 4 E4B-it checkpoint. Activation extraction runs in 8-bit precision via bitsandbytes for VRAM reasons, and the weight edit is applied to those 8-bit weights in place. (The bf16 control for this choice is examined in §8 and turns out not to be the cause of the negative finding reported below.)

## 7.2 The M2c sweep result

We swept α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0} at full-depth coverage (all 42 layers), and swept layer subsets at α = 1.0 across nine partitions (all_42, global_only, sliding_only, first_half, second_half, last_10, middle_14, peak_band_4_17, peak_layer_15_only). Each cell was evaluated on a stratified 20-prompt subsample of `data/benchmark_prompts.json` with `max_new_tokens = 128`. This subset was the trade-off committed in [STATUS, M2c 5.7] to fit the 19-cell ensemble into ≈ 2.8 h of GPU time. A random-direction control at α = 1.0 was run as a sanity check.

The headline numbers, taken verbatim from `results/ablation_results/sweep_results.json`:

- Alpha sweep: refusal rate flat at 30–35% across α ∈ [0.0, 2.0]. Concretely, 30% at α = 0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, and 35% at α = 0.3, 1.2.
- Layer-subset sweep: refusal rate flat at 25–35% across the nine subsets. Global-only and sliding-only both 30%, second_half 25%, the rest 30–35%.
- Random-direction control: refusal 30%, indistinguishable from the α = 0 baseline.

![Figure 7.1: Alpha sweep of standard rank-1 mean-difference abliteration on Gemma 4 E4B-it (8-bit, all 42 layers). Refusal rate stays in the 30–35% band across α ∈ [0, 2.0] on the stratified n = 20 subsample; the curve is statistically indistinguishable from the α = 0 baseline and from a random-direction control. The rank-1 mean-difference intervention does not move the model along the direction the activation-space analysis identifies.](../../results/figures/alpha_sweep.png)

![Figure 7.2: Layer-subset sweep at α = 1.0 across nine partitions (all_42, global_only, sliding_only, first_half, second_half, last_10, middle_14, peak_band_4_17, peak_layer_15_only). Refusal rate is flat in the 25–35% band across all subsets; concentrating the projection on the L4–L17 peak band identified in §6 does not produce a stronger effect than spreading it across all 42 layers, and neither subset crosses the threshold the random-direction control fails to cross either.](../../results/figures/layer_subset_comparison.png)

The headline finding: standard rank-1 mean-difference abliteration is empirically ineffective on Gemma 4 E4B-it with the raw `d_l` direction artifact. The sweep is flat, no α or layer subset crosses the random-direction control, and a direct n = 48 behavioral re-test against the load-bearing `should_refuse` category (M2c-followup, [STATUS, (b)]) registered 6/6 = 100% refusal, identical to base. This is *not* a direction-quality failure at the activation level. §6 documented that the per-category geometry is well-separated: over-refuse cluster cosine +0.93, should-refuse vs over-refuse −0.015, three orders of magnitude apart. The directions exist; the rank-1 mean-difference fix does not move the model along them.

## 7.3 Forward pointer to §8

The follow-up is a causal-isolation cascade that holds the rank-1 weight update fixed and varies the direction-construction recipe instead. §8 reports the full five-hypothesis cascade: bnb int8 edit path (H1), chat-template-applied direction extraction (H2), per-layer winsorization at 99.5% (H3), two-pass Gram-Schmidt against the harmless mean (H4), norm-preserving biprojection (H5). The cascade's terminal result is that swapping the M2b raw direction for the D3 stacked variant (H2 + H3 + H4) raises should_refuse refusal-reduction from 0 to 17/42 = 40.5% partial.

The §8 claim is phrased in the language the M5 spec requires for stacked Stage 2 variants. D3 (two-pass Gram-Schmidt against the harmless mean) is necessary in combination with prior ingredients to produce the partial 40.5% result; we do not claim sufficient — Stage 2.5 unstacked isolation was not run, so we cannot rule out the possibility that the chat-template step (D1) or the winsorization step (D2) is doing the real work on its own and the Gram-Schmidt layer is decorative. Section 8 develops the distinction in detail. For the present section the point is narrower: the rank-1 *intervention* is not the failure mode. The *direction* fed into it is.

## 7.4 Comparative weight-diff geometry: OBLITERATUS and TrevorJS

To see what an effective uncensoring edit looks like on Gemma 4 E4B-it, we compute the element-wise weight diff between the published base checkpoint and two published abliterated variants:

- OBLITERATUS [elder-plinius 2025], abliteration of Gemma 4 E4B-it produced by the OBLITERATUS toolkit; available as bf16 safetensors and Q8_0 GGUF.
- TrevorJS [TrevorS 2026], Gemma 4 E4B-it uncensored via norm-preserving biprojection [grimjim 2025] plus Efficient Gradient Ablation; available as bf16 safetensors.

The diff pipeline (`src/weight_diff/compute_diff.py`) loads both checkpoints in fp32 on CPU, computes `Δ = W_modified − W_original` per tensor, and reports the Frobenius norm, relative change `‖Δ‖_F / ‖W_original‖_F`, max absolute change, the top-10 singular values of Δ, the effective ranks `rank_95` and `rank_99` (smallest *k* such that the top-*k* singular values explain ≥ 95% or 99% of `‖Δ‖_F²`), and `top1_energy_fraction = σ_1² / ‖Δ‖_F²`. A K/V-borrower-alias deduplication step (described in §7.6) removes 36 tensors that appear as edits but are alias-image artifacts of Gemma 4's shared K/V architecture.

The dedup results are reported in `results/weight_diffs/gemma_obliteratus/weight_diff_results_dedup.json` and `results/weight_diffs/gemma_trevorjs/weight_diff_results_dedup.json`. The headline contrast:

| | OBLITERATUS | TrevorJS |
|---|---|---|
| Changed weight tensors (post-dedup) | ~201 (200 with rank_95 computed) | 84 |
| Median `rank_95` | 6 | 1 |
| Modification character | multi-rank across q/k/v/o, gate/up/down, embed | pure rank-1 norm-preserving biprojection |
| Param-type spread (LM-only, frob > 1e-4) | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, embed | o_proj, down_proj only |

![Figure 7.3: OBLITERATUS effective-rank distribution per modified weight tensor. The median `rank_95` is 6, so the published successful abliteration is *not* rank-1 on Gemma 4. The distribution has a long tail extending into the high-dozens for some MLP layers, consistent with a low-dimensional but multi-modal weight footprint.](../../results/figures/obliteratus_modification_ranks.png)

![Figure 7.4: Cross-method top-1 left singular vectors of the per-layer weight diff. The histogram of cosine similarities between OBLITERATUS's and TrevorJS's top-1 modification directions, taken at the matching layer/parameter pair, has median −0.08 (n = 48 layer × {o_proj, down_proj} pairs; min −0.35, max +0.38). The two methods' top-1 modification directions live in nearly orthogonal subspaces; they are not approximating the same underlying rank-1 fix.](../../results/figures/cross_method_singular_vectors.png)

The interpretation is straightforward. Even the published successful abliteration is not rank-1 on Gemma 4. OBLITERATUS achieves uncensoring through median-rank-6 edits spread across seven parameter types (q/k/v/o, gate/up/down) and the embedding matrix. TrevorJS achieves it through pure rank-1 norm-preserving biprojection on o_proj and down_proj only. Two methodologically incompatible recipes both succeed where standard rank-1 mean-difference abliteration fails, *and* their top-1 modification directions are nearly orthogonal across the matched layer/parameter pairs (cross-method top-1 cosine median −0.08). The implication is that refusal on Gemma 4 admits a continuum of effective low-rank fixes rather than one canonical safety basis. The full per-layer overlay (`weight_diff_per_layer_overlay_dedup.png`) and the singular-value spectra (`singular_value_spectra_per_method.png`, `obliteratus_per_layer_weight_change.png`, `obliteratus_param_type_changes.png`) document the layer-by-layer distribution and component-type breakdown referenced in the M5 spec scenario "Section 7 narrative shape."

## 7.5 Refusal-direction × singular-vector cross-reference

An obvious diagnostic is to ask whether either method's weight-edit direction *coincides* with the activation-space refusal direction `d_l` recovered in §6. For each modified `o_proj` or `down_proj` matrix in each method's diff, we compute the cosine similarity between the top-1 left singular vector of Δ and the corresponding-layer M2b refusal direction. The table lives in `results/weight_diffs/refusal_direction_vs_singular_vector.csv`.

The result (n = 132 rows; 66 OBLITERATUS + 66 TrevorJS) is median |cos| ≈ 0.04 for both methods: OBLITERATUS median 0.0425, TrevorJS median 0.0423. The mean is 0.05. The per-row distribution is concentrated near zero, with a thin tail reaching ~0.20.

![Figure 7.5: Cosine similarity between the M2b activation-space refusal direction (§6, per-layer `d_l`) and the top-1 left singular vector of the per-layer weight diff, for OBLITERATUS and TrevorJS separately. Both methods land at median |cos| ≈ 0.04. Neither method's top-1 weight-edit direction aligns with the activation-space refusal direction the §6 mechanistic analysis identifies.](../../results/figures/refusal_direction_vs_singular_vector.png)

The implication is one of the paper's non-trivial geometric findings. The M2b refusal direction is not the same vector as either method's top-1 weight-diff singular vector. The activation-level direction and the weight-edit direction diverge on Gemma 4. This is the same disconnect §6.4 flagged at the activation-vs-behavior level, now surfaced at the weight-space level: the activation-space rank-1 picture is real (cos²(PC₁, Δμ) ≈ 0.91 at L15), but it is not the direction that uncensoring methods actually edit along. Standard rank-1 mean-difference abliteration takes the activation-space direction and applies it directly to the weights, and that is the recipe the M2c sweep showed to be ineffective. The published successful methods do not target that direction.

## 7.6 Gemma 4 architectural quirks

Three Gemma 4-specific architectural features bear on how the weight-diff numbers are read. They are subtle enough to bias the analysis if ignored.

*Shared K/V borrower aliases.* Gemma 4 E4B-it has 42 layers (35 sliding-attention plus 7 global-attention at indices [5, 11, 17, 23, 29, 35, 41]). The sliding-attention layers do not have their own `k_proj` and `v_proj` weight tensors. They alias to the nearest preceding global-attention layer's K/V. When safetensors are written, the alias is materialized as a separate tensor pointing to the same parameter buffer. A naive element-wise diff over the safetensors manifest therefore sees a change in *both* the global K/V layer and every sliding K/V layer that borrows from it: 36 alias-image edits per K/V change. We deduplicate by detecting shared-buffer tensors and reporting each edit once at the borrower-source (global) layer. The procedure is described in `results/weight_diffs/.shared_tensor_handling.md`, and it reduces the OBLITERATUS edit count from 2130 to 2094 raw entries, and from 403 to 367 entries with nonzero Frobenius change, of which 200 carry a computed `rank_95` after the LM-only-and-frob-significant filter. The TrevorJS edit set is unaffected because TrevorJS edits only `o_proj` and `down_proj`, neither of which is K/V-aliased.

*RMSNorm row-norm sensitivity.* Gemma 4 wraps each `o_proj` and `down_proj` output in an RMSNorm layer that was trained against a particular per-row magnitude structure. A vanilla rank-1 projection `W ← W − α · d · (d^T W)` changes `‖W_i‖_2` for every row *i* on which `d^T W` has support, and the RMSNorm rescaling can then propagate the perturbation in directions the projection did not intend. This is the H5 hypothesis tested in §8. The TrevorJS variant addresses it by norm-preserving biprojection [grimjim 2025], which decomposes each row into magnitude × direction, projects within the direction subspace, and re-applies the original magnitude, leaving `‖W_i‖_2` exactly unchanged. The §8 full 42-layer row-norm audit (215 040 rows) reports mean Δ‖W_i‖/‖W_i‖ = 0.038%, p99 = 0.34%, and a max of 9.73% at one row of L01 o_proj. Those numbers sit below the 1% threshold at which RMSNorm sensitivity would plausibly drift, and behaviorally Stage 3a (norm-preserving biprojection on the D3 direction artifact) produces identical outputs to D3 vanilla on every n = 6 prompt. H5 is rejected at this scale. The row-norm-penalty argument cited in [Heretic Issue 265] and [grimjim 2025] does not appear to be load-bearing for the failure of standard rank-1 abliteration on E4B at 8-bit edit precision.

*Direction-construction recipe.* The published successful methods differ from the [Arditi 2024] / [Mlabonne 2024] mean-difference recipe in the *direction-construction* step, not in the projection algebra. [TrevorS 2026] uses chat-template-applied inputs, per-layer winsorization at the 99.5th percentile, and two-pass Gram-Schmidt orthogonalization against the harmless-mean activation. [elder-plinius 2025] uses multi-rank descent that effectively produces a low-rank-but-not-rank-1 direction artifact. The §8 cascade isolates Gram-Schmidt against the harmless mean (H4) as necessary in the D3 stack, consistent with the geometric picture in §7.4 and §7.5: the *weight-diff* singular vectors of both successful methods are nearly orthogonal to each other and nearly orthogonal to the activation-space refusal direction. The load-bearing geometric object appears to be not the activation-space `d_l` itself but a leakage-removed variant of it that lives off the harmless-mean axis.

## 7.7 Summary

Three findings carry forward from this section to §8 and the discussion.

First, standard rank-1 mean-difference abliteration on Gemma 4 E4B-it is empirically ineffective: alpha sweep flat 30–35%, layer-subset sweep flat 25–35%, random-direction control 30%, and a head-to-head behavioral test on the load-bearing `should_refuse` category at 6/6 = 100% versus base 100%. The sweep is statistically indistinguishable from doing nothing.

Second, the published successful uncensoring methods on Gemma 4 E4B-it are not rank-1 in the mean-difference sense. OBLITERATUS has median `rank_95` = 6 across ~201 modified weights (200 with `rank_95` computed) spanning seven parameter types and the embedding. TrevorJS is rank-1 but uses norm-preservation rather than vanilla projection, and edits 84 weights restricted to `o_proj` and `down_proj`. The two methods' top-1 modification directions are nearly orthogonal across matched layer/parameter pairs (cross-method top-1 cosine median −0.08), implying a continuum of effective low-rank fixes rather than a canonical safety basis.

Third, the M2b activation-space refusal direction and the weight-space modification direction diverge on Gemma 4. Both methods' top-1 weight-diff singular vectors are essentially uncorrelated with the matching-layer M2b `d_l` (median |cos| ≈ 0.04). The activation-space rank-1 picture of §6 is real, but it is not the direction that operative uncensoring methods edit along. That gap is the architectural anomaly the §8 cascade takes up by varying the direction-construction recipe alone.

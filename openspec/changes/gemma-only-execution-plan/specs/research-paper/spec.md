## ADDED Requirements

### Requirement: Paper structure with survey and experiments
The paper SHALL follow a 10-section structure: (1) Introduction with the hiking emergency motivation, (2) Background on alignment techniques, (3) Related Work survey, (4) **Mathematical Framework** anchoring the rank-1 intervention in EECS 6699 lecture results (Lec 5 high-dim geometry, Lec 6 Mirsky / singular-value perturbation, Lec 7 SVD machinery, Lec 8–9 lazy / NTK linearization), (5) Over-Refusal Analysis, (6) Mechanistic Analysis, (7) **Abliteration and Comparative Weight Diff Across Published Gemma 4 E4B Abliterations** (combined section), (8) **M6 Rank-1 Cascade** (causal isolation of the load-bearing ingredient on Gemma 4), (9) Discussion and Course Connections, (10) Conclusion and Ethics.

#### Scenario: All sections present
- **WHEN** the paper is compiled
- **THEN** it SHALL contain all 10 sections with substantive content in each

#### Scenario: Section 7 narrative shape
- **WHEN** Section 7 is written
- **THEN** it SHALL present the project's own M2c rank-1 abliteration alongside at least one published Gemma 4 E4B uncensored variant (OBLITERATUS) — and ideally a second (TrevorJS) — citing per-layer Frobenius norms, SVD effective ranks, and the cross-reference cosine between weight-diff singular vectors and M2b refusal directions; **it SHALL NOT contain MoE / expert / router content** (the Qwen MoE workstream is removed from the project)

#### Scenario: M6 outcomes integrated into the headline (causal isolation)
- **WHEN** M6 produces a positive Stage 1.5-confirmed result on a single ingredient
- **THEN** the paper headline (Section 1 abstract + Section 7 / 8 framing) SHALL reframe from "rank-1 abliteration is empirically ineffective on Gemma 4 E4B" to a *causally-isolated* finding naming the load-bearing ingredient — e.g., "rank-1 abliteration works on Gemma 4 except when composed with bitsandbytes int8 in-place editing" (H1), "the refusal direction on Gemma 4 is chat-template-sensitive" (H2), or "norm preservation is necessary because RMSNorm penalizes row-norm changes" (H5) — and SHALL cite Stage 4 full-benchmark numbers from `$RESULTS_DIR/stage4_<winner_slug>/evaluation_results.csv`

#### Scenario: M6 outcomes integrated when no stage cracks
- **WHEN** every M6 stage fails to produce a confirmed ≤30% result at n=42
- **THEN** Section 7 or Section 8 SHALL append a systematic-ablation appendix documenting the five hypotheses tested (H1 bnb int8 edit path, H2 chat-template-derived direction, H3 winsorization, H4 Gram-Schmidt, H5 norm preservation) and SHALL phrase the contribution as "we tested five hypotheses about the cause; none individually closed the gap, suggesting either a higher-order composition not captured by single-variable toggles, or an environmental cause (tokenizer / classifier / generation settings)"

#### Scenario: M6 paper claims phrased as "necessary," not "sufficient"
- **WHEN** Section 7 / 8 cites a Stage 2 stacked variant (D1/D2/D3) as the load-bearing ingredient
- **THEN** the prose SHALL phrase the claim as "ingredient X is necessary in combination with prior ingredients" UNLESS Stage 2.5 unstacked isolation has confirmed single-variable causation, in which case the prose MAY phrase the claim as "ingredient X is sufficient on its own"

### Requirement: Survey covers alignment and de-alignment literature
The survey (Sections 2-3) SHALL cover: RLHF, DPO, Constitutional AI as alignment techniques; representation engineering (Zou et al. 2023), abliteration (Arditi et al. 2024), fine-tuning attacks (Qi et al. 2023) as de-alignment techniques; and the over-refusal problem.

#### Scenario: Minimum citation count
- **WHEN** the survey is complete
- **THEN** it SHALL reference at least 15 papers with proper citations

#### Scenario: Coverage of recent Gemma 4 abliteration work
- **WHEN** the survey is complete
- **THEN** Section 3 SHALL reference at least three published Gemma 4-era abliteration approaches (OBLITERATUS / elder-plinius, TrevorJS / norm-preserving biprojection per grimjim, Heretic / p-e-w) and SHALL discuss the Gemma 4 RMSNorm and shared K/V architectural quirks documented in the model cards

### Requirement: Section 4 Mathematical Framework anchors EECS 6699 lectures
**Section 4 SHALL be a Mathematical Framework** that formalizes the rank-1 abliteration intervention `W ← W − α·d·dᵀW` (with `W` rectangular — `o_proj` and `down_proj`, so eigenvalue-based bounds do not apply directly) and invokes EECS 6699 lecture results *before* the empirical sections (§5–§8) cite them. The section SHALL cite the canonical lecture-to-module map at `docs/findings/course-material-mapping.md`. The section SHALL invoke (a) **Lec 5** — near-orthogonality of random vectors in 2560-dim Gaussian space and volume concentration on the thin shell, applied to *learned conceptual directions* (not raw activations); (b) **Lec 6 — Mirsky's singular-value perturbation theorem** in the form `Σᵢ (σᵢ(W+E) − σᵢ(W))² ≤ ‖E‖_F²`, with the rank-1 specialization that for unit `d`, `E = α·d·(dᵀW)` is rank-1 and `‖E‖_F = ‖E‖_2 = |α|·‖dᵀW‖_2` (the singular-value form of Hoffman-Wielandt, since `W` is rectangular); (c) **Lec 7** — SVD / Frobenius / spectral norms as the machinery used downstream in §7 and §8; (d) **Lec 8–9** — lazy training / NTK linearization: weights stay near initialization, so the model's behavioral output under a small rank-1 edit is approximately a *linear* functional of the perturbation.

#### Scenario: Section 4 invokes Mirsky, not eigenvalue Hoffman-Wielandt
- **WHEN** Section 4 states the perturbation bound on `W ← W − α·d·dᵀW`
- **THEN** the bound SHALL be stated in terms of singular values `σᵢ` (Mirsky's theorem), NOT eigenvalues `λᵢ`, and the prose SHALL note that the standard Hoffman-Wielandt eigenvalue formulation does not apply directly to the rectangular `o_proj` / `down_proj` matrices; the section SHALL also note that for rank-1 `E`, the Frobenius and spectral norms coincide (`‖E‖_F = ‖E‖_2`)

#### Scenario: Section 4 numerics embedded
- **WHEN** Section 4 is compiled
- **THEN** it SHALL embed the per-layer Mirsky-style **relative perturbation** heatmaps `‖E‖_F / ‖W‖_F` from `results/math_framework/mirsky_bound_per_layer.csv` for **both** the D3 directions (`shared/results/agent/m6-rank1-followup/m6_directions/refusal_directions_d3.pt`, the M6 winner) and the raw M2b directions (`shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt`, the negative baseline), and SHALL embed the three anisotropy figures: `results/figures/projection_energy_L15.png`, `results/figures/learned_directions_cosine_L15.png`, and `results/figures/activation_anisotropy_L15.png`

#### Scenario: Anisotropy caveat for Lec 5 framing
- **WHEN** Section 4 cites Lec 5 near-orthogonality
- **THEN** it SHALL explicitly distinguish between near-orthogonality of *random* vectors in 2560-dim Gaussian space (which the Lec 5 result establishes and which applies to *learned conceptual directions* such as the refusal direction and category means) and isotropy of *raw residual-stream activations* (which the `activation_anisotropy_L15.png` figure shows does NOT hold — the empirical activation cloud at L15 has mean pairwise cosine ≈ 0.96, far from the Gaussian N(0, 1/d) null of mean 0 and σ ≈ 0.02; the anisotropy manifests as a bulk mean shift rather than inflated spread, since empirical centered std ≈ 0.014 actually sits slightly below the isotropic null); the load-bearing claim SHALL be framed as near-orthogonality of *learned directions*, not isotropy of activations

### Requirement: Section 9 course connections (Discussion)
Section 9 SHALL re-cite the Section 4 framework with experimental numbers plugged in (over-parametrization framing; quantitative tie from the Mirsky bound to the M6 cascade's partial-effect result), with a higher density of lecture citations than §4 alone.

#### Scenario: Five lecture connections, at least one quantitative
- **WHEN** Section 9 is written
- **THEN** it SHALL contain **at least 5** explicit connections to specific EECS 6699 course lectures or named theorems, with mathematical notation; **at least one connection SHALL be quantitative** (a numeric Mirsky-style bound computed for the D3 directions, tied to Section 8's 40.5% partial-refusal finding by showing that the per-layer relative perturbation `‖E‖_F / ‖W‖_F` is small enough to leave most of the spectrum intact, consistent with a partial behavioral change); **at least one connection SHALL come from each of** {Lec 5 high-dim geometry}, {Lec 6 perturbation theory, Lec 7 spectral norms}, and {Lec 8 lazy training, Lec 9 NTK / Rademacher}

#### Scenario: Lazy-NTK framing soft, not predictive
- **WHEN** Section 9 ties Lectures 8–9 (lazy training / NTK) to Section 8's partial 40.5% refusal-reduction result
- **THEN** the prose SHALL phrase the connection as "the lazy / linearization lens makes a partial first-order response plausible" or "consistent with a linear first-order behavioral response under a small rank-1 perturbation," NOT as "NTK predicts the 40.5% partial result"; the residual ~60% is to be framed as evidence that the refusal behavior is not fully one-dimensional, not as something the NTK framework predicts on its own

### Requirement: Figures from experimental results
The paper SHALL include publication-quality figures generated by the experimental pipelines.

#### Scenario: Minimum figure set
- **WHEN** all experiments are complete
- **THEN** the paper SHALL include at minimum: (a) refusal rate heatmap (M2a + M2c-followup), (b) layer-wise signal strength chart (M2b), (c) UMAP activation visualization (M2b), (d) alpha sweep curve (M2c), (e) **per-variant per-layer weight-diff chart and the per-layer overlay across variants** (M3), (f) **cross-method cosine table or figure** (M3), (g) **refusal-direction × singular-vector cross-reference figure** (M3), (h) selective safety results table (M2c), (i) **Mirsky-bound relative-perturbation `‖E‖_F / ‖W‖_F` heatmap per layer, for both D3 and M2b directions** (§4 numerics from `results/math_framework/mirsky_bound_per_layer.csv`), (j) **projection-energy histogram at L15** (`results/figures/projection_energy_L15.png`), (k) **learned-directions cosine matrix at L15** (`results/figures/learned_directions_cosine_L15.png`), and (l) **activation-anisotropy figure at L15** with Gaussian N(0, 1/d) reference overlay (`results/figures/activation_anisotropy_L15.png`)

#### Scenario: M6 ablation table when a stage cracks
- **WHEN** any M6 stage produces a Stage 1.5-confirmed result
- **THEN** the paper SHALL include a M6 ablation table whose rows are the M6 hypothesis stack (Stage 0a bf16 edit path, Stage 2 D1/D2/D3, Stage 3a/3b) and whose columns are `should_refuse` rate at n=6, n=42 (if reached), and n=344 (Stage 4 winner only), so the reader can see which incremental ingredient closed the gate

### Requirement: Ethics discussion
Section 10 SHALL include an ethics discussion addressing the dual-use nature of abliteration research, the over-refusal harm motivation, and the selective safety contribution.

#### Scenario: Ethics section present
- **WHEN** Section 10 is written
- **THEN** it SHALL acknowledge that abliteration techniques can be misused, explain why understanding alignment fragility is necessary for improving it, and highlight the selective safety experiment as the constructive application

### Requirement: Presentation slides
The slide deck and paper writeup SHALL be produced by an agent (or small agent team) working on the `agent/writeup` branch, AFTER the human verification gate (M4) has passed, and SHALL cite only experimental numbers that exist in files under `results/` with a verifiable file path and commit hash.

#### Scenario: Slide deck complete
- **WHEN** the presentation is prepared
- **THEN** it SHALL contain the hiking emergency scenario as the opening, at least 3 key result figures, a dedicated **Mathematical Framework** slide referencing Section 4 (Mirsky-style bound on Gemma 4 weights + the anisotropy caveat for Lec 5), and a course-connections section (Section 9 recap with ≥5 lecture connections)

#### Scenario: Writeup gated by human verification
- **WHEN** the writeup agent begins drafting paper prose or slide bullets
- **THEN** `STATUS_FOR_HUMAN.md` SHALL already exist on the `agent/writeup` branch AND contain the operator's green-light sentence, otherwise the writeup agent SHALL stop and report the gate as unresolved

#### Scenario: All numeric claims traceable
- **WHEN** any paper section or slide quotes a refusal rate, signal strength, alpha value, or Frobenius norm
- **THEN** the same number SHALL appear in a file under `results/`, and the writeup agent SHALL record the source path and commit hash either inline or in a companion `paper/sources.md` file

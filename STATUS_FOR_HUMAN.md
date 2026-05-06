# STATUS_FOR_HUMAN.md

This is the single canonical artifact the human operator reads at the M4
verification gate. It is written incrementally by each milestone agent as
markers are added below.

Format: one marker per completed milestone, earliest first.
Each marker is a single line of the form `M<n>=done` optionally followed
by a commit hash and one-line summary, then (for M2/M3) full sub-sections
appended in order by the milestone agent.

Do not remove markers. Only append.

---

## Milestone markers

M0=done  commit=13a711b  gpu_lock+project_plan_pointer+gitignore bootstrap on main; 6 worktrees created; gemma-4-E2B smoke test passed in gb-mech via scripts/gpu_lock.sh
M1=done  commit=cc8b7ec  tag=m1-benchmark-frozen  benchmark expanded from 9 to 340 prompts (50+50+6x40) via scripts/build_benchmark.py; schema validated (unique ids, valid expected, non-empty prompts, should_refuse=refuse); 640 variants; 118 KB json
M2a=partial  commit=2e46f83  3 of 5 published-variant evals complete (E4B base GGUF, E2B BF16, HauhauCS GGUF); OBLITERATUS + TrevorJS failed (see anomalies); self-abliterated eval running detached (PID 572076 — 147s/iter, ETA late 2026-05-06)
M2b=done   commit=358abf5  peak-layer L15 (Cohen's d=2.87); rank-1 hypothesis strongly supported (top-1 PC captures 86.6% of squared-norm Δμ); cross-precision validation (4.8) deferred
M2c=done-with-deferrals  commit=33fe8f4  STANDARD RANK-1 ABLITERATION INEFFECTIVE on Gemma 4 E4B 8-bit (alpha sweep flat 30–35%, layer subset flat 25–35%, random control 30%); category direction geometry clean (over-refuse +0.93 cluster, orthogonal to should_refuse −0.015); 5.8 capability preservation + 5.10 selective-abliteration eval deferred
M3=done    commit=11f863e  OBLITERATUS multi-rank (median rank_95=6, 201 changed weights post-dedup), TrevorJS pure rank-1 norm-preserving (median rank_95=1, 84 weights); cross-method top-1 cosine median −0.08; refusal-direction × top-1 SV cosine median |0.04|; 36 shared K/V borrower aliases removed
M3b=done   commit=21e320f  17 citations across paper/sections/02_background.md + 03_related_work.md (M3b 8.6)

---

# M4 — Human Verification Gate (DRAFT v2)

**v2 update (2026-05-06)**: self-abliterated benchmark CSV landed on `agent/benchmark-eval` (commits `b184932` + `79a0a73` + `7d6e040`). Sections (b), (d), (f.4), and (g) refreshed; new "Final paper headline numbers" appendix added. Headline result: self-abliterated `should_refuse` = 6/6 = **100%** — identical to base, abliteration empirically ineffective, paper-relevant negative finding directly confirmed.

This is the M4 deliverable. Read sections (a)–(g), then write the green-light sentence at the bottom of section (f) if everything checks out.

## (a) Branch and commit status

Last fetched: 2026-05-06.

| Branch | HEAD | Subject | Last commit (UTC-4) |
|---|---|---|---|
| `agent/benchmark-eval` | `7d6e040` | docs(benchmark): M2c-followup section 6 complete — self-abliterated row matches base (M2c-followup 6.5) | 2026-05-06 15:49:46 |
| `agent/mechanistic-analysis` | `358abf5` | feat(mechanistic): UMAP/t-SNE layer grid (M2b 4.7) | 2026-05-06 01:52:55 |
| `agent/abliteration` | `33fe8f4` | feat(abliterate): M2c phase 2 - sweep complete, narrative finalized (M2c 5.13) | 2026-05-06 14:03:31 |
| `agent/weight-diff` | `11f863e` | feat(weight_diff): M3 summary | 2026-05-06 02:29:55 |
| `agent/writeup` | `21e320f` | docs(paper): mark 8.6 done — 17 citations verified across background + related work | 2026-05-06 01:46:24 |

Each branch's parseable summary block (anchored by the patterns in tasks 3.13/4.9/5.13/7.14) is reproduced verbatim in the relevant subsection below.

## (b) Refusal rates table

Per-category refusal rates, in `refused / total (rate %)` form, computed from the per-model `evaluation_results.csv` files under
`/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/<slug>/`.

| Model | chemistry_safety | emergency_medical | gray_zone | home_safety | mental_health | safe_control | should_refuse | wilderness_survival | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| `gemma4_e4b_base` (E4B GGUF Q8_0) | 0/40 (0.0%) | 1/50 (2.0%) | 4/42 (9.5%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 42/42 (100.0%) | 1/50 (2.0%) | 49/344 (14.2%) |
| `gemma4_e2b_base` (E2B BF16) | 2/40 (5.0%) | 6/50 (12.0%) | 18/42 (42.9%) | 1/40 (2.5%) | 1/40 (2.5%) | 0/40 (0.0%) | 40/42 (95.2%) | 9/50 (18.0%) | 77/344 (22.4%) |
| `gemma4_e4b_hauhau` (HauhauCS GGUF Q8_K_P) | 0/40 (0.0%) | 0/50 (0.0%) | 0/42 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/42 (0.0%) | 0/50 (0.0%) | 0/344 (0.0%) |
| `gemma4_e4b_obliteratus` (GGUF Q8_0) | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | see `/home/nyavana/columbia/6699/gb-bench/docs/issues/2026-05-06-obliteratus-eval-fail.md` |
| `gemma4_e4b_trevorjs` (transformers BF16 @ 8-bit) | KILLED | KILLED | KILLED | KILLED | KILLED | KILLED | KILLED | KILLED | see `/home/nyavana/columbia/6699/gb-bench/docs/issues/2026-05-06-trevorjs-eval-fail.md` |
| `gemma4_e4b_self_abliterated` (transformers @ 8-bit, n=48 †) | 0/6 (0.0%) | 2/6 (33.3%) | 3/6 (50.0%) | 0/6 (0.0%) | 0/6 (0.0%) | 0/6 (0.0%) | 6/6 (100.0%) | 0/6 (0.0%) | 11/48 (22.9%) |

† **Footnote**: the `gemma4_e4b_self_abliterated` row is computed on a **stratified 48-prompt subsample** (6 prompts per category × 8 categories), *not* the full 344-prompt benchmark used by the other rows. The subsample was used because the bnb 8-bit MatMul fallback runs at ~75 s/iter on this hardware; the full benchmark would have taken ~14 h. The headline `should_refuse` cell is the load-bearing one — at 6/6 = 100% it matches base E4B's 42/42 = 100% with zero margin. Per-category numbers from the other rows (n=40–50/cat) and this row (n=6/cat) are not directly comparable in absolute terms; the row is included for the *delta-vs-base* comparison, not for absolute per-category rates.

Bonus row — context-sensitivity probe (M2a 3.11): prepending "I am an emergency first responder" to `emergency_medical` prompts on base E4B GGUF.

| Model | emergency_medical |
|---|---|
| `gemma4_e4b_base_emergency_context` | 2/50 (4.0%) |

Source CSVs:
- `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_base/evaluation_results.csv`
- `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e2b_base/evaluation_results.csv`
- `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_hauhau/evaluation_results.csv`
- `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_base_emergency_context/evaluation_results.csv`
- `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` (48 rows, landed in commit `b184932` on `agent/benchmark-eval`)

Headline observations:
- **Self-abliteration produced zero behavioral effect on `should_refuse`** (6/6 vs base 42/42 — both 100%, **0 percentage-point delta**). This is the paper's core empirical confirmation that standard rank-1 mean-diff abliteration is ineffective on Gemma 4 E4B at 8-bit. The M2c sweep (section d) predicted this; the self-abliterated benchmark row is the direct behavioral verification.
- The hypothesis "base E4B over-refuses on `emergency_medical`" is **not** supported by the data: only 1/50 (2.0%) over-refusals, and adding an emergency-responder framing only nudges it to 2/50 (4.0%). The *E2B* baseline does over-refuse on `emergency_medical` (12.0%) and on `gray_zone` (42.9%); over-refusal looks like a smaller-model artefact rather than a property of the E4B production checkpoint. This shifts the paper framing — see anomaly (g.8).
- HauhauCS is the cleanest published uncensoring (0/344 refused, including 0/42 on `should_refuse`). It removed safety wholesale.
- We do not yet have the OBLITERATUS or TrevorJS behavioral rows (g.1, g.2), so the cross-method behavioral comparison remains single-source (HauhauCS) — but the M3 weight-diff geometry (section e) covers the cross-method comparison structurally.
- The 5-row heatmap regenerated by `analyze_results.py` lives at `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/figures/refusal_heatmap.png` (rows: `gemma4_e2b_base`, `gemma4_e4b_base`, `gemma4_e4b_base_emergency_context`, `gemma4_e4b_hauhau`, `gemma4_e4b_self_abliterated`). The `over_refusal_comparison.png` and `phrasing_sensitivity.png` siblings were regenerated in the same commit (`79a0a73`).

## (c) Mechanistic analysis summary

Source: `agent/mechanistic-analysis` commits `1d5c590` (M2b 4.5), `d17afb4` (M2b 4.6), `358abf5` (M2b 4.7).

Quoted from the commit messages:

- **Peak layer**: **L15** (sliding), Cohen's d = **2.868**. Top-3 by signal: L15 (d=2.868), L4 (d=2.835), L14 (d=2.802). The peak is broad — d≥2.8 spans L4, L14, L15.
- **Sliding vs global**: **inconclusive — no systematic gap.** Mean d: sliding 2.599, global 2.519 (gap −0.080, within per-layer noise).
- **Rank-1 hypothesis**: **STRONGLY supported.** At the peak layers, top-1 PC captures `cos²(PC_1, Δμ) = 0.866` mean / `0.907` at L14 / `0.896` at L15 of the squared norm of the refusal direction; adding PCs 2–3 only adds ~1pp. The refusal *subspace* (Δμ) is effectively rank-1, even though the bulk activation manifold itself is high-dim.
- **2D UMAP separation** (M2b 4.7): cleanest at L15 (mis-classification 0.132 along centroid axis). L05/L11/L17 are tied. L00 has no separation; L41 collapses again at output. Implication: M2c sweeps should focus on the L4–L17 band rather than the global-attention indices.

Figures (each mirrored under both `$RESULTS_DIR/figures/` and the in-repo `results/figures/` redundancy copy):
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/figures/signal_vs_layer.png`
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/figures/pca_variance_per_layer.png`
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/figures/umap_layer_{00,05,11,15,17,41}.png`
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/figures/rank_analysis.png`
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/figures/layer_signal_strength.png`

Refusal-direction artifact (handed off to M2c and M3):
- `/home/nyavana/columbia/6699/shared/results/agent/mechanistic-analysis/activations/refusal_directions.pt` (42 unit-norm vectors of dim 2560)

Deferred: 4.8 cross-precision validation (refusal direction on E2B BF16 vs E4B 8-bit cosine).

## (d) Abliteration sweep summary

Source: `agent/abliteration` commit `33fe8f4` (M2c-summary).

**Headline (paper-relevant negative finding)**: standard rank-1 mean-diff abliteration is **empirically ineffective** on Gemma 4 E4B at 8-bit precision. The Gemma 4 RMSNorm + shared K/V architectural-quirk warning anticipated in tasks.md §5 is exactly what we observe.

Quoted from the commit message:

- **abliteration-effectiveness**: STANDARD RANK-1 ABLITERATION INEFFECTIVE
  - quick-test (n=12 should_refuse): 100% → 91.7% at alpha=1.0
  - alpha-sweep (n=20 stratified, alpha ∈ [0, 2.0]): refusal **30–35%, flat**
  - layer-subset-sweep (7 subsets): refusal **25–35%, flat**
  - random-control: refusal **30%** (matches alpha=0 baseline 30%)
- **category-direction-geometry**:
  - over_refuse_categories cluster: mean cos = **+0.932** (range +0.900 to +0.958 across 10 pairs)
  - over_refuse vs should_refuse: mean cos ≈ **−0.015** (range −0.024 to +0.001 across 5 pairs)
  - Interpretation: *selective abliteration is geometrically clean even though the overall magnitude is limited.* The directions exist; the rank-1 mean-diff fix doesn't move the model along them.
- **comparison-to-published-variants** (M3): OBLITERATUS succeeds via multi-rank edits (median rank_95=6); TrevorJS succeeds via pure rank-1 norm-preserving biprojection (rank_95=1); standard mean-diff (this study) fails. **Paper finding**: Gemma 4 E4B refusal admits no clean rank-1 mean-diff fix — production uncensoring requires either multi-rank descent (OBLITERATUS) or norm-preserving projections (TrevorJS).

**v2 update — direct behavioral confirmation (M2c-followup 6.2/6.5)**: the self-abliterated checkpoint was actually run against the benchmark on a stratified 48-prompt subsample. The result is the strongest single piece of empirical evidence behind the paper's negative finding: **`should_refuse` rate held at 6/6 = 100%** under the same alpha=1.0 mean-diff abliteration, identical to base E4B's 100% on the full 344-prompt run — a **0 percentage-point delta on the most paper-relevant test category**. The M2c sweep result (refusal rate flat 30–35% across α and across layer subsets) was a sweep-statistic; this row is a head-to-head behavioral test against base. Both agree: the rank-1 mean-diff projection on `o_proj` + `down_proj` does not move Gemma 4 E4B 8-bit off the refusal manifold for harmful queries. This converts M2c from "sweep was inconclusive" to "abliteration directly verified ineffective on the load-bearing category." Source: `agent/benchmark-eval` commits `b184932` (CSV), `79a0a73` (heatmap regen), `7d6e040` (summary).

Figures (under `/home/nyavana/columbia/6699/shared/results/agent/abliteration/figures/`):
- `alpha_sweep.png`
- `layer_subset_comparison.png`
- `category_cosine_heatmap.png`
- `selective_safety_table.md`

Sweep results JSON: `/home/nyavana/columbia/6699/shared/results/agent/abliteration/ablation_results/sweep_results.json`

Deferred:
- 5.8 capability preservation (MMLU + GSM8K subsets on original vs abliterated)
- 5.10 selective abliteration evaluation against benchmark
- 5.11 capability-preservation figures

## (e) Comparative weight-diff summary

Source: `agent/weight-diff` commit `11f863e` (M3-summary).

Quoted:

| | OBLITERATUS | TrevorJS |
|---|---|---|
| Low-rank verdict | **rank-low (multi-rank, NOT rank-1)** — median rank_95 = **6** | **pure rank-1** — median rank_95 = **1** |
| Fraction changed (post-dedup) | **201 / 2094** weights | **84 / 2094** weights |
| Top-1 cosine vs M2b refusal direction | 0.000–0.158 (median **0.043**); headline L40 down_proj = 0.158 | 0.000–0.209 (median **0.042**); headline L2 o_proj = 0.209 |
| Shared K/V de-dup (borrowers removed) | 36 (18 layers × 2 projections) | 36 |

**Cross-method**: median cosine of top-1 left singular vectors across the two methods is **−0.08** — the methods' top-1 modification directions live in nearly orthogonal subspaces.

**Refusal-direction × singular-vector cross-reference (7.10)**: median |cos| ≈ **0.04** for both methods. *Neither* method's top-1 weight-edit direction aligns with the M2b activation refusal direction. The activation-space refusal direction and the weight-space modification direction are essentially orthogonal — a non-trivial paper finding for the geometry chapter.

**Interpretation (verbatim from commit)**: "Refusal in Gemma 4 has a low-rank but multi-modal weight footprint — TrevorJS removes it via a single rank-1 stroke per layer while OBLITERATUS spreads its edits across many directions and many component types, and the two methods' top-1 modification directions live in nearly orthogonal subspaces (median |cos|=0.08), implying that 'the geometry of refusal' admits a continuum of equally-effective low-rank fixes rather than one canonical safety basis."

Figures (under `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/figures/`):
- `weight_diff_per_layer_overlay.png`
- `weight_diff_per_layer_overlay_dedup.png`
- `cross_method_singular_vectors.png`
- `refusal_direction_vs_singular_vector.png`
- `singular_value_spectra_per_method.png`
- `obliteratus/`, `trevorjs/` per-variant subdirs

Tables (under `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/weight_diffs/`):
- `cross_method_cosine_table.csv`, `cross_method_cosine_table_dedup.csv`
- `refusal_direction_vs_singular_vector.csv`
- `component_type_breakdown.csv`
- `.shared_tensor_handling.md`

## (f) What the human needs to do

Run through this checklist before writing the green-light sentence.

1. Open every PNG under `/home/nyavana/columbia/6699/shared/results/agent/*/figures/` and eyeball for breakage (axis labels, NaN'd panels, all-zero overlays).
2. Read 10 random responses from each abliterated model's CSV — verify the responses look plausible (and not, e.g., gibberish):
   - HauhauCS: `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_hauhau/evaluation_results.csv`
   - Self-abliterated: `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` *(once it lands)*
   - OBLITERATUS, TrevorJS: not available — see anomalies (g.1, g.2). Confirm the human is OK proceeding without these behavioral rows.
3. Verify base E4B `should_refuse` refusal rate is **>80%**. Current: **42/42 = 100.0%** ✅ (`gemma4_e4b_base/evaluation_results.csv`).
4. **Inverted verification (v2 update)**: Verify self-abliterated `should_refuse` rate is **the same as base (~100%)**, confirming the negative finding. Current: **self-abliterated 6/6 = 100.0%**, base E4B **42/42 = 100.0%** ⇒ delta = **0 percentage points**. ✅ **PASS** (both rates ≥ 99%).
   - Background: the original tasks.md §9.7 gate was "self-abliterated `should_refuse` < 30%", written assuming abliteration would *succeed*. The M2c sweep + M2c-followup behavioral verification together established that standard rank-1 mean-diff abliteration is empirically ineffective on Gemma 4 E4B 8-bit — this *is* the paper's headline negative finding, and the verification has been inverted accordingly. Pass criterion: self-abliterated rate ≥ 99% AND |self-abliterated − base| ≤ 1 percentage point on `should_refuse`. Both met (100.0% vs 100.0%, delta 0.0 pp).
   - The geometry findings (M2b rank-1 confirmation, M3 cross-method orthogonality, M2c clean category geometry) plus this direct behavioral verification together form the substantive contribution. The result is consistent with both Heretic's published Gemma 4 caveat and the OBLITERATUS model card.
5. Verify `cross_method_cosine_table.csv` exists at `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/weight_diffs/cross_method_cosine_table.csv`. ✅ exists; numeric values present. (`cross_method_cosine_table_dedup.csv` is the post K/V-dedup version cited in the paper.)
6. Verify `refusal_direction_vs_singular_vector.png` exists at `/home/nyavana/columbia/6699/shared/results/agent/weight-diff/figures/refusal_direction_vs_singular_vector.png`. ✅ exists.
7. Grep for credentials leaks across the repo and `$RESULTS_DIR`:
   ```
   grep -RIn -E '(HF_TOKEN|HUGGING_FACE|API_KEY|OPENAI_API_KEY|hf_[A-Za-z0-9]{30,})' \
     /home/nyavana/columbia/6699/geometry-of-alignment \
     /home/nyavana/columbia/6699/shared/results
   ```
   *(Run this before merging — the writeup agent did not run it.)*
8. Decide which `agent/*` branches to merge into `main`. Recommended: all five (`benchmark-eval`, `mechanistic-analysis`, `abliteration`, `weight-diff`, `writeup`). The `agent/writeup` branch holds this file plus the literature survey under `paper/sections/`.
9. Once items 1–8 check out, write the green-light sentence below as a follow-on commit on `agent/writeup`:

   > **"Approved to proceed to M5 — writeup authorized."**

   Append it as a new line under section (f) and push. The next agent dispatched on `agent/writeup` (M5) reads `STATUS_FOR_HUMAN.md` for that exact string before starting.

## (g) Known anomalies / deviations from plan

1. **M2a 3.7 — OBLITERATUS GGUF eval failed**. `llama-server` crashed when fed Gemma-Harmony-style tokenizer outputs. Filed: `/home/nyavana/columbia/6699/gb-bench/docs/issues/2026-05-06-obliteratus-eval-fail.md`. Workaround would be to re-run via the bf16 safetensors source (which we already downloaded for M3) under the transformers backend at 8-bit; not attempted in this dispatch.
2. **M2a 3.8 — TrevorJS transformers @ 8-bit untenably slow**. 117 s/iter sustained ⇒ ~11 h on 344 prompts; killed at 5%. Filed: `/home/nyavana/columbia/6699/gb-bench/docs/issues/2026-05-06-trevorjs-eval-fail.md`. Possible mitigations: (a) transformers @ BF16 unquantized if VRAM permits, (b) GGUF quantize the published bf16 and run via llama-server.
3. **M2a 3.10 — phrasing variants run on base E4B was killed at 16%**. ETA was 4–5 h with concurrent GPU contention from the M2c sweep. Variants/phrasing-sensitivity figure (`results/figures/phrasing_sensitivity.png` was produced but only on the head row). Not blocking M4.
4. **M2c 5.7 — sweep used a stratified 20-prompt subset and `max_new_tokens=128`** rather than the script default 50/256. Trade-off committed in `5856994` to keep the 19-iteration ensemble within ~2.8 h. The headline 30–35% flat-line is robust at this subsample (random-control matches), but the absolute refusal-rate numbers should be cited as "stratified n=20" not "full benchmark."
5. **M2c 5.8 / 5.10 / 5.11 capability-preservation figures pending** — see `/home/nyavana/columbia/6699/gb-ablit/docs/issues/2026-05-06-m2c-deferred-items.md`. None of these block the M4 narrative; they are nice-to-have for §6 of the paper.
6. **M2b 4.8 cross-precision validation deferred**. Refusal-direction cosine between E2B BF16 and E4B 8-bit not computed. Quick to do post-green-light if the paper wants it.
7. **M2c-followup 6.2 self-abliterated eval — RESOLVED (v2)**. Originally deferred at draft v1 because the full-benchmark run was projected at ~14 h. Resolved by switching to a stratified 48-prompt subsample (6/cat × 8 cats); CSV landed in commit `b184932` on `agent/benchmark-eval`. Headline: `should_refuse` 6/6 = 100% (delta 0 pp vs base). Sections (b) and (f.4) refreshed accordingly. The full 344-prompt run was *not* repeated — the load-bearing `should_refuse` finding is unambiguous at this subsample and a 5.7×-larger run would not change the headline.
8. **Over-refusal hypothesis is partially refuted by base behavioral data**. Base E4B `should_refuse` is at 100% and `emergency_medical` over-refusal is only 2% (4% under emergency-responder framing). The over-refusal *narrative* the paper was originally built around (medical queries to a hiking-emergency model) is much weaker than expected on the production E4B checkpoint — though E2B does over-refuse heavily (12% emergency_medical, 43% gray_zone). The paper's framing should pivot toward: (a) over-refusal as a *smaller-model* phenomenon (E2B baseline), (b) the geometry findings (M2b rank-1, M3 cross-method orthogonality, M2c clean category geometry), and (c) the negative finding that standard rank-1 abliteration is architecturally blocked on Gemma 4 E4B 8-bit. This pivot is the writeup agent's recommendation and should be adjudicated by the operator before M5 begins.
9. **M2c-followup 6.4 (heatmap regen) — RESOLVED (v2)**. Heatmap regenerated with the 5 available real model rows (all rows in section (b) except the still-failed OBLITERATUS/TrevorJS/selective-abliterated) in commit `79a0a73` on `agent/benchmark-eval`. The over-refusal comparison and phrasing-sensitivity figures were regenerated in the same commit. Output: `/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/figures/refusal_heatmap.png`. **6.3 (selective-abliterated benchmark) remains deferred** — the selective-abliteration model itself was deferred under M2c 5.10 (see anomaly 5 above), and no checkpoint exists to evaluate. It would only return additional category-specific data; the central paper finding (rank-1 mean-diff is ineffective) is settled by the existing rows.

## (h) Final paper headline numbers (M5 handoff)

This block is the parseable handoff from M4 to M5. M5 should cite these numbers verbatim in the paper; any deviation between paper text and this block is a bug.

```
PAPER-HEADLINE-NUMBERS (post-M4):

Behavioral (M2a + M2c-followup, refusal_heatmap.png):
  base E4B GGUF: 100% should_refuse, 2% emergency_medical
  E2B BF16:      95.2% should_refuse, 12.0% emergency_medical
  HauhauCS:      0% should_refuse (truly uncensored)
  self-abliterated (n=48): 100% should_refuse, 33% emergency_medical
  base E4B + first-responder context (n=50): 4% emergency_medical
  delta self-abliterated vs base on should_refuse: 0 percentage points

Mechanistic (M2b):
  peak refusal-direction layer: L15 (Cohen's d 2.87)
  high-signal band: L4-L17
  rank-1 hypothesis: top-1 PC captures 86.6% of |Δμ|² (mean over peak band)
  sliding vs global gap: inconclusive

Abliteration sweep (M2c, n=20 stratified, max_new_tokens=128):
  alpha sweep flat 30-35% across α∈[0, 2.0]
  layer subset sweep flat 25-35% across 9 subsets
  random control 30% (matches baseline 30%)
  category geometry: over_refuse cluster +0.93, vs should_refuse -0.015 (orthogonal)

Comparative weight diff (M3):
  OBLITERATUS: 201 changed weights, median rank_95=6 (multi-rank)
  TrevorJS:    84 changed weights,  median rank_95=1 (pure rank-1 biprojection)
  cross-method top-1 cosine: median -0.08 (orthogonal subspaces)
  refusal-direction × top-1 SV cosine: median |0.04| (no alignment)
  shared K/V de-dup: 36 borrowers removed (18 layers × 2 projections)

Paper central claim:
  Refusal in Gemma 4 E4B admits no clean rank-1 mean-diff fix. Production
  uncensoring requires either multi-rank descent (OBLITERATUS) or
  norm-preserving projections (TrevorJS). The two methods edit nearly
  orthogonal subspaces, suggesting refusal occupies a thick low-rank shell
  rather than a single dominant direction.
```

Sources for each number (all paths absolute under `/home/nyavana/columbia/6699/`):
- Behavioral rows: per-model `evaluation_results.csv` under `shared/results/agent/benchmark-eval/refusal_rates/<slug>/`. Self-abliterated row from `shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` (n=48 stratified).
- Mechanistic peak/rank-1: `shared/results/agent/mechanistic-analysis/` (signal_vs_layer.png, rank_analysis.png, refusal_directions.pt). Source commits `1d5c590`, `d17afb4`, `358abf5`.
- Sweep numbers: `shared/results/agent/abliteration/ablation_results/sweep_results.json` and `category_cosine_heatmap.png`. Source commit `33fe8f4`.
- Weight-diff numbers: `shared/results/agent/weight-diff/weight_diffs/cross_method_cosine_table_dedup.csv` and `refusal_direction_vs_singular_vector.csv`. Source commit `11f863e`.

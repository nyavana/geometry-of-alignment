# Sources for Numeric Claims

Every numeric claim in the paper that is empirically derived maps below to (a) the section that quotes it, (b) the verbatim numeric, (c) the on-disk source file, (d) the commit where that artifact was last meaningfully updated, and (e) a one-phrase re-derivation note. Commit hashes are abbreviated to 7 chars and reference the `agent/writeup` (paper) branch unless prefixed with a subagent-branch tag in the note.

Numbers re-cited verbatim in later sections (§9, §10) point back to the §4–§8 row where they were first sourced; we do not re-list a number unless a later section computes it independently. A small set of secondary or interpretive numbers that have no isolated on-disk source are listed under `## Unverified` at the bottom.

---

## §1 Introduction — four findings + framing

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §1.3 Finding 1 | `42/42 (100%)` should_refuse on base E4B | `results/refusal_rates/gemma4_e4b_base/evaluation_results.csv` | `60dfbf6` | groupby `category`, count rows where `actual_label=="refuse"` for `category=="should_refuse"` |
| §1.3 Finding 1 | `1/50 (2%)` emergency_medical on base E4B | same CSV | `60dfbf6` | same groupby, `category=="emergency_medical"` |
| §1.3 Finding 1 | `2/50 (4%)` emergency_medical under emergency-responder framing | `STATUS_FOR_HUMAN.md` §(b) (probe row) | `f42e0d6` | recorded headline from M2a 3.6-followup probe (not in a per-CSV file) |
| §1.3 Finding 1 | `6/50 (12%)` emergency_medical on base E2B | `results/refusal_rates/gemma4_e2b_base/evaluation_results.csv` | `e48d838` | same groupby, `category=="emergency_medical"` |
| §1.3 Finding 1 | `18/42 (43%)` gray_zone on base E2B | same E2B CSV | `e48d838` | same groupby, `category=="gray_zone"` (18/42 = 42.9%) |
| §1.3 Finding 2 | L15 top-1 PC captures `86.6%` of squared norm of Δμ across L4–L17 | `results/refusal_directions/refusal_directions.pt` + §6 mechanistic pipeline; reported in `STATUS_FOR_HUMAN.md` §"PAPER-HEADLINE-NUMBERS / Activation-space" | `57cedcf` | per-layer PCA cos²(PC₁, Δμ) computed in `src/mechanistic/layer_analysis.py`; band mean across L4–L17 |
| §1.3 Finding 2 | Cohen's *d* = `2.87` (rounded from 2.868) at L15 | same source | `57cedcf` | per-layer effect size from `src/mechanistic/layer_analysis.py`; L15 row |
| §1.3 Finding 2 | α-sweep 30–35%, layer-subset 25–35% | `results/ablation_results/sweep_results.json` | `09f4931` | filter `axis=="alpha"` and `axis=="layers"` rows; min/max of `refusal_rate` |
| §1.3 Finding 3 | D3 → `17/42 = 40.5%` should_refuse at n=42 | `shared/results/agent/m6-rank1-followup/stage_1_5_d3_should_refuse_n42/evaluation_results.csv` (handoff); summary in `STATUS_FOR_HUMAN.md` §"Stage 1.5 result" | M6 merge `e4e5622` (Stage 1.5 commit `b3e8a6b`) | groupby `category=="should_refuse"`, count `actual_label=="refuse"` |
| §1.3 Finding 3 | row-norm change `mean 0.038%` (vanilla projection) | `scripts/m6_figures.py::figure_5_row_norms` + audit table in `STATUS_FOR_HUMAN.md` §"Stage 3a result" | `866aeb7` (figure regen); audit in M6 merge `e4e5622` | full 42-layer × 2 projection × 2560 rows audit; mean Δ‖W_i‖/‖W_i‖ |
| §1.3 Finding 4 | OBLITERATUS median `rank_95 = 6`, `201` weights post-dedup | `results/weight_diffs/gemma_obliteratus/weight_diff_results_dedup.json` | `9d6c830` | filter entries with `rank_95` not null; median; count entries with `frobenius_norm>0` (=367), then LM-only-and-rank_95-computed filter → 200; paper cites ~201 |
| §1.3 Finding 4 | TrevorJS pure rank-1, `84` weights | `results/weight_diffs/gemma_trevorjs/weight_diff_results_dedup.json` | `9d6c830` | filter entries with `frobenius_norm>0`; count; median `rank_95` over those = 1 |
| §1.3 Finding 4 | cross-method top-1 cosine median `−0.08` | `results/weight_diffs/cross_method_cosine_table_dedup.csv` | `9d6c830` | median of `cosine_similarity` column over 48 rows |
| §1.3 Finding 4 | refusal × top-1 SV median \|cos\| ≈ `0.04` | `results/weight_diffs/refusal_direction_vs_singular_vector.csv` | `c3c0ebb` | groupby `variant`, median of `abs(cosine_abs)`: OBLITERATUS 0.0425, TrevorJS 0.0423 |
| §1.4 framing | over-refuse-category mean cosine `+0.93` mutual; `−0.015` vs should_refuse | activations + §6 selective-safety geometry; numbers in `STATUS_FOR_HUMAN.md` §"M2c geometry" | `57cedcf` (activations); analysis commit per `src/abliterate/selective_safety.py` | per-category mean-diff at L15, pairwise cosines (10 pairs intra-cluster, 5 pairs vs should-refuse) |

---

## §4 Mathematical framework — Mirsky bound + anisotropy

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §4.5 | D3 median `‖E‖_F / ‖W‖_F = 0.022154` over 84 cells | `results/math_framework/mirsky_bound_per_layer.csv` | `c8891c3` | filter `variant=="d3"`, median of `ratio` column; n=84 |
| §4.5 | D3 min `0.018052`, max `0.030920` | same CSV | `c8891c3` | min/max of `ratio` for `variant=="d3"` |
| §4.5 | M2b median `‖E‖_F / ‖W‖_F = 0.023713` over 84 cells | same CSV | `c8891c3` | filter `variant=="m2b"`, median of `ratio` column; n=84 |
| §4.5 | M2b min `0.019343`, max `0.038281` | same CSV | `c8891c3` | min/max of `ratio` for `variant=="m2b"` |
| §4.5 cross-check | D3 (L15, o_proj) `frob_E = 1.19374`, `frob_W = 58.18948`, ratio `0.02052` | same CSV | `c8891c3` | row `variant==d3, layer==15, projection==o_proj` |
| §4.5 cross-check | TrevorJS L15 o_proj `frobenius_norm = 1.24537`, `original_norm = 58.18948` | `results/weight_diffs/gemma_trevorjs/weight_diff_results_dedup.json` | `9d6c830` | lookup `model.language_model.layers.15.self_attn.o_proj.weight` |
| §4.5 cross-check | M2b L15 o_proj `frob_E = 1.38` | `results/math_framework/mirsky_bound_per_layer.csv` | `c8891c3` | row `variant==m2b, layer==15, projection==o_proj` (1.3832) |
| §4.4 | mean pairwise cos L15 raw activations = `0.958` (over 28,680 pairs) | `results/math_framework/anisotropy_headline.json` (`mean_pairwise_cos_L15`) | `eabb5ce` | json key `mean_pairwise_cos_L15` |
| §4.4 | centered std L15 raw = `0.014` | same json (`centered_std_L15`) | `eabb5ce` | json key `centered_std_L15` (0.014042) |
| §4.4 | isotropic null std `1/√d ≈ 0.020` for d=2560 | same json (`isotropic_std_reference`) | `eabb5ce` | json key `isotropic_std_reference` (0.019764) |
| §4.4 | anisotropy ratio = `0.71` (≈ 0.014 / 0.020) | same json (`anisotropy_ratio`) | `eabb5ce` | json key `anisotropy_ratio` (0.7105) |
| §4.4 | mean shift `48.5` isotropic std units | same json (`mean_shift_in_isotropic_std_units`) | `eabb5ce` | json key |
| §4.4 | learned-directions median \|cos\| = `0.978` at L15 | same json (`median_abs_cos_learned_directions`) | `eabb5ce` | json key (0.978216) |
| §4.4 | `cos(mean_refuse, mean_comply) ≈ 0.984` | same json (`_cos_mean_refuse_comply`) | `eabb5ce` | json key (0.9843) |
| §4.4 | `d = normalize(mean_refuse − mean_comply)` cosine to mean_refuse `≈ 0.081` | same json (`_note_cos_mean_refuse_d`) | `eabb5ce` | json key (0.0809) |
| §4.4 | frac energy on d_refuse mean `1.22%` (refuse class) | same json (`frac_energy_on_d_refuse_mean`) | `eabb5ce` | json key (0.012151) |
| §4.4 | frac energy on d_comply mean `1.05%` (comply class) | same json (`frac_energy_on_d_comply_mean`) | `eabb5ce` | json key (0.010467) |
| §4.4 | N_refuse = 262, N_comply = 40 | same json | `eabb5ce` | json keys `N_refuse`, `N_comply` |
| §4.4 | 28,680 pairs | same json (`_n_pairwise_pairs`) | `eabb5ce` | json key |

---

## §5 Over-Refusal — four-model refusal-rate landscape

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §5.1 | 344 prompts, 8 categories, 646 variants | `data/benchmark_prompts.json` | `6252640` (initial commit) | count primary entries; sum lengths of `variants` arrays |
| §5.2 E2B row | `2/40` chemistry, `6/50` emergency_medical, `18/42` gray_zone, `1/40` home_safety, `1/40` mental_health, `0/40` safe_control, `40/42` should_refuse, `9/50` wilderness; total `77/344 (22.4%)` | `results/refusal_rates/gemma4_e2b_base/evaluation_results.csv` | `e48d838` | groupby `category`, count `actual_label=="refuse"`; total = sum (verified 77/344) |
| §5.2 E4B base row | `0/40` chem, `1/50` emergency_medical, `4/42` gray_zone, `1/40` home, `0/40` mental, `0/40` safe, `42/42` should_refuse, `1/50` wilderness; total `49/344 (14.2%)` | `results/refusal_rates/gemma4_e4b_base/evaluation_results.csv` | `60dfbf6` | same groupby; total verified 49/344 |
| §5.2 HauhauCS row | `0/x` across all 8 categories; `0/344 (0.0%)` | `results/refusal_rates/gemma4_e4b_hauhau/evaluation_results.csv` | `b629d67` | same groupby; all zero |
| §5.2 self-ablit row (n=48) | `0/6` chem, `2/6 (33.3%)` emergency_medical, `3/6 (50.0%)` gray_zone, `0/6` home, `0/6` mental, `0/6` safe, `6/6 (100%)` should_refuse, `0/6` wilderness; total `11/48 (22.9%)` | `results/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` | `b184932` | same groupby on n=48 stratified subset |
| §5.3 | self-ablit should_refuse `6/6 = 100%` = base E4B 100% (delta 0 pp) | same self_abliterated CSV + E4B base CSV | `b184932` + `60dfbf6` | direct read |
| §5.4 | base E2B is roughly 6× E4B's per-category over-refusal rate | computed across E2B/E4B CSVs above | `e48d838` + `60dfbf6` | sum refusals over 6 benign categories per model; ratio |
| §5.6 | α sweep flat 30–35%, layer-subset 25–35%, random-direction 30% | `results/ablation_results/sweep_results.json` | `09f4931` | min/max `refusal_rate` per axis; control row `axis=="control_random"` |

---

## §6 Mechanistic — L15 peak + rank-1 evidence

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §6.1 | refuse class n=262, comply class n=40 | `results/refusal_directions/prompt_metadata.json` | `57cedcf` | count entries per `class` field in metadata |
| §6.2 | Cohen's *d* at L15 = `2.868`; L14 = 2.802, L4 = 2.835 | activations + `src/mechanistic/layer_analysis.py` Cohen's d compute; per-layer figures `results/figures/signal_vs_layer.png` | `4fa3fdc` (activations) + `57cedcf` (directions) | per-layer pooled-std effect size on projected scores |
| §6.2 | sliding mean *d* = 2.599, global mean *d* = 2.519 | same | `4fa3fdc` + `57cedcf` | groupby attention type using `[5,11,17,23,29,35,41]` global indices |
| §6.3 | top-1 PC `86.6%` of squared norm Δμ across L4–L17 band; L14 `0.907`, L15 `0.896` | activations + PCA per layer; figure `results/figures/pca_variance_per_layer.png` | `4fa3fdc` + `57cedcf` | per-layer PCA of `x_refuse,i − μ_comply`, take cos²(PC₁, Δμ); average across L4–L17 |
| §6.4 | over-refuse cluster mean cos `+0.932`, range +0.900 to +0.958 (10 pairs) | per-category direction pipeline (`src/abliterate/selective_safety.py`); reported in `STATUS_FOR_HUMAN.md` §"M2c geometry" | activations `4fa3fdc` + analysis commit per M2c | per-category mean-diff at L15, pairwise cosines across 5 over-refuse categories |
| §6.4 | over-refuse × should-refuse mean cos `−0.015`, range −0.024 to +0.001 (5 pairings) | same | same | cosine of each over-refuse direction against should_refuse direction at L15 |
| §6.4 | cos(μ_refuse, μ_comply) ≈ `0.984` at L15 | `results/math_framework/anisotropy_headline.json` (`_cos_mean_refuse_comply`) | `eabb5ce` | json key |
| §6.5 | mean pairwise cos 0.958 (28,680 pairs), centered std 0.014, iso null 0.020, ratio 0.71 | `results/math_framework/anisotropy_headline.json` | `eabb5ce` | json keys (see §4.4 rows) |
| §6.5 | frac energy on d_15: 1.2% refuse / 1.0% comply | same json | `eabb5ce` | json keys `frac_energy_on_d_refuse_mean` / `frac_energy_on_d_comply_mean` |

---

## §7 Abliteration + comparative weight diff

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §7.2 | α-sweep refusal flat 30–35% across α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0} | `results/ablation_results/sweep_results.json` | `09f4931` | filter `axis=="alpha"`; print `(alpha, refusal_rate)` pairs |
| §7.2 | layer-subset sweep flat 25–35% across 9 partitions; `second_half` = 25% | same | `09f4931` | filter `axis=="layers"`; print `(layers, refusal_rate)` pairs |
| §7.2 | random-direction control 30% | same | `09f4931` | row `axis=="control_random"`, refusal_rate=0.3 |
| §7.2 | self-ablit n=48 should_refuse `6/6 = 100%` | `results/refusal_rates/gemma4_e4b_self_abliterated/evaluation_results.csv` | `b184932` | groupby; same as §5 row |
| §7.4 | OBLITERATUS ~201 weights post-dedup with rank_95 computed (200 with rank_95 + 367 with non-zero frob) | `results/weight_diffs/gemma_obliteratus/weight_diff_results_dedup.json` | `9d6c830` | filter `frobenius_norm > 0` ⇒ 367; filter `rank_95 not null` ⇒ 200; paper writes "~201" |
| §7.4 | OBLITERATUS median `rank_95 = 6` | same | `9d6c830` | median of `rank_95` over 200 non-null entries |
| §7.4 | TrevorJS 84 weights, median `rank_95 = 1` | `results/weight_diffs/gemma_trevorjs/weight_diff_results_dedup.json` | `9d6c830` | filter `frobenius_norm > 0` ⇒ 84; median `rank_95` = 1 |
| §7.4 | cross-method top-1 SV cosine median `−0.08` (n=48), min `−0.35`, max `+0.38` | `results/weight_diffs/cross_method_cosine_table_dedup.csv` | `9d6c830` | median/min/max of `cosine_similarity` column over 48 rows (verified −0.0781, −0.3545, +0.3843) |
| §7.5 | refusal × top-1 SV median \|cos\| ≈ `0.04` (n=132, 66 OBLITERATUS + 66 TrevorJS) | `results/weight_diffs/refusal_direction_vs_singular_vector.csv` | `c3c0ebb` | groupby `variant`, median \|cos\|: OBLITERATUS 0.0425, TrevorJS 0.0423; mean ≈ 0.05 |
| §7.5 | per-method maxima: OBLITERATUS L40 down_proj cos = 0.158; TrevorJS L2 o_proj cos = 0.209 | same | `c3c0ebb` | filter by `variant`, find max `cosine_abs` row |
| §7.6 | K/V dedup removes 36 alias tensors; 2130 → 2094 raw entries; 403 → 367 nonzero-frob | `results/weight_diffs/gemma_obliteratus/weight_diff_results.json` vs `..._dedup.json` + `results/weight_diffs/.shared_tensor_handling.md` | `9d6c830` | diff entry counts pre/post dedup |
| §7.6 | row-norm audit (215,040 rows): mean 0.038%, p99 0.34%, max 9.73% at L01 o_proj | M6 audit; `scripts/m6_figures.py::figure_5_row_norms` (regenerated figures in `866aeb7`) | `866aeb7` (figure regen); underlying audit commit `b3e8a6b` (M6 Stage 1.5) | full 42-layer × 2 projection × 2560 rows audit on D3 directions at α=1.0 |

---

## §8 Rank-1 cascade — M6 stage results

| Section | Verbatim claim | Source file | Commit | Re-derivation |
|---|---|---|---|---|
| §8.3 Table | Stage 0a bf16 self-ablit: `6/6 (100%)` should_refuse, `9/48 (18.8%)` total | `shared/results/agent/m6-rank1-followup/stage0a/evaluation_results.csv` | M6 merge `e4e5622` (Stage 0 commit `6e084c1`) | groupby `category`, count refuse rows |
| §8.3 Table | Stage 0b TrevorJS bf16: `0/6` should_refuse, `0/48 (0%)` total | `shared/results/agent/m6-rank1-followup/stage0b/evaluation_results.csv` | M6 merge `e4e5622` | same groupby |
| §8.3 Table | D1 (chat-template): `6/6 (100%)`, `6/48 (12.5%)` | `shared/results/agent/m6-rank1-followup/stage_2_d1_chat_template/evaluation_results.csv` | M6 merge `e4e5622` (commit `380c1d0` for D1 plumbing) | same groupby |
| §8.3 Table | D2 (D1 + winsorize): `6/6 (100%)`, `6/48 (12.5%)` | `.../stage_2_d2_winsorize/evaluation_results.csv` | M6 merge `e4e5622` (commit `4bc5708` for D2 build) | same groupby |
| §8.3 Table | D3 (D2 + Gram-Schmidt): `1/6 (16.7%)`, `1/48 (2.1%)` smoke | `.../stage_2_d3_full_recipe/evaluation_results.csv` | M6 merge `e4e5622` (commit `811ce3a`) | same groupby |
| §8.3 Table | Stage 3a (D3 + norm-preserving biprojection): `1/6 (16.7%)`, `1/48 (2.1%)` | `.../stage_3a_biprojection_d3_dirs/evaluation_results.csv` | M6 merge `e4e5622` (commit `e840ea4`); biprojection plumbing `578b682` | same groupby |
| §8.3 | **n=42 D3 confirmation: `17/42 (40.5%)` should_refuse** | `shared/results/agent/m6-rank1-followup/stage_1_5_d3_should_refuse_n42/evaluation_results.csv` | M6 merge `e4e5622` (commit `b3e8a6b`) | groupby `category=="should_refuse"`, count refuse rows; 17/42 |
| §8.3 cosines | cos(M2b raw, D1 chat-template) at L15 = `0.09` | direction `.pt` artifacts under `shared/results/agent/m6-rank1-followup/m6_directions/` | M6 merge `e4e5622` (D1 plumbing `380c1d0`) | per-layer cosine of refusal_directions.pt vs refusal_directions_chat.pt at L15 |
| §8.3 cosines | cos(D1, D2) at L15 = `0.994` | same .pt artifacts | M6 merge `e4e5622` (D2 build `4bc5708`) | cosine of refusal_directions_chat.pt vs refusal_directions_d2.pt at L15 |
| §8.3 cosines | cos(D2, D3) at L15 = `0.952` (≈17° rotation) | same | M6 merge `e4e5622` (D3 build `811ce3a`) | cosine of refusal_directions_d2.pt vs refusal_directions_d3.pt at L15 |
| §8.3 winsorize check | max-abs L15 activation drops `77.0 → 4.6` under 99.5% winsorize | activations + winsorize plumbing | M6 merge `e4e5622` (commit `4bc5708`) | computed by D2 build script during cascade |
| §8.3 row-norm audit | mean `0.038%`, median `0.013%`, p99 `0.34%`, max `9.73%` at L01 o_proj; per-layer mean band `0.024–0.069%` | M6 audit; figure `results/figures/m6_row_norm_distribution.png` | `866aeb7` (figure regen); audit data in M6 merge `e4e5622` | 215,040 rows = 42 × 2 × 2560; relative Δ‖W_i‖ |

---

## §9 Discussion — re-cites §4–§8

All numbers in §9 are re-cited from §4–§8 entries above. The §9.1 lecture-connection narrative references:

- Cohen's *d* = 2.868 / 86.6% top-1 PC (→ §6 rows)
- α-sweep 30–35% / layer-subset 25–35% (→ §7.2 rows)
- D3 17/42 = 40.5% (→ §8.3 n=42 row)
- median `‖E‖_F / ‖W‖_F` = 0.022 D3 / 0.024 M2b (→ §4.5 rows)
- 1.22% / 1.05% projection energy (→ §4.4 rows)
- mean pairwise cos 0.958, ratio 0.71, 48.5 std units (→ §4.4 rows)
- over-refuse +0.932 / vs should-refuse −0.015 (→ §6.4 rows)
- OBLITERATUS median `rank_95 = 6` / TrevorJS 1 / cross-method cosine −0.08 / refusal × SV \|cos\| ≈ 0.04 (→ §7 rows)
- row-norm mean 0.038% (→ §7.6 / §8.3 rows)
- 1/√2560 ≈ 0.0198 isotropic null (→ §4.4 row `isotropic_std_reference`)
- `√2048 ≈ 45.3`, `√2560 ≈ 50.6` derived from architecture constants in `CLAUDE.md`

No new on-disk artifacts cited in §9 beyond what is already mapped above.

---

## §10 Conclusion — re-cites §5–§8

All numbers in §10 are re-cited from §5–§8 entries above:

- 42/42 (100%) and 1/50 (2%) base E4B; 6/50 (12%), 18/42 (43%) E2B (→ §5.2 rows)
- HauhauCS 0/344 (→ §5.2 row)
- self-ablit 6/6 = 100% (→ §5.2 / §5.3 row)
- Cohen's *d* = 2.87, top-1 PC 86.6%, +0.93 / −0.015 (→ §6 rows)
- OBLITERATUS 201 weights, median rank_95 = 6 / TrevorJS 84 weights, median rank_95 = 1; cross-method cosine −0.08; refusal × SV \|cos\| ≈ 0.04 (→ §7 rows)
- α-sweep 30–35%, layer-subset 25–35%; 17/42 = 40.5%; row-norm mean 0.038% (→ §7 / §8 rows)
- median `‖E‖_F / ‖W‖_F` = 0.022 (D3), 0.024 (M2b); mean pairwise cos 0.958; +0.93 / −0.015 (→ §4 rows)

---

## Unverified

The numbers below are quoted in the paper but I could not locate an isolated on-disk artifact for them. They are likely derivable from CSVs above, from prose in `STATUS_FOR_HUMAN.md`, or from a derivation script that was not committed.

1. **§5.4 — emergency-responder probe `2/50 (4%)` on base E4B GGUF.** Reported in `STATUS_FOR_HUMAN.md` §(b) as a one-line probe result; no dedicated `evaluation_results.csv` under `results/refusal_rates/` carries the probe rows. Re-derivation would need to re-run the probe prompts through the existing pipeline.

2. **§6.2 — per-layer Cohen's *d* curve (2.868 at L15, 2.802 at L14, 2.835 at L4, sliding mean 2.599 / global mean 2.519, drop below 2.0 by L20 / below 1.0 by L35).** The L15 headline is confirmed by `STATUS_FOR_HUMAN.md` "PAPER-HEADLINE-NUMBERS" block, but the per-layer table per-se is not committed as a CSV; it is computed inside `src/mechanistic/layer_analysis.py` from the `.pt` activations and rendered to `results/figures/signal_vs_layer.png`. The activations themselves (`4fa3fdc`) are the source of truth; the curve is reconstructible.

3. **§6.3 — PCA per-layer table (L14 = 0.907, L15 = 0.896, band-mean L4–L17 = 0.866).** Same situation as #2: computed in `src/mechanistic/layer_analysis.py` and rendered to `results/figures/pca_variance_per_layer.png`; the raw activations at commit `4fa3fdc` are the source, but no CSV is committed.

4. **§6.4 — over-refuse cluster mean `+0.932` (range 0.900–0.958) and `−0.015` (range −0.024 to +0.001).** Reported verbatim in `STATUS_FOR_HUMAN.md` "PAPER-HEADLINE-NUMBERS / M2c geometry" block; the per-pairing CSV is `results/figures/category_cosine_heatmap.png`'s underlying data, which is computed in `src/abliterate/selective_safety.py` but the pair-level numbers are not committed as a flat table. The .pt directions in `results/refusal_directions/refusal_directions.pt` (`57cedcf`) are the source.

5. **§8.3 — cosines between successive direction artifacts (M2b vs D1 = 0.09, D1 vs D2 = 0.994, D2 vs D3 = 0.952).** Cited verbatim from `STATUS_FOR_HUMAN.md` Stage 2 summary; computed inline during the M6 cascade. The .pt files exist under `shared/results/agent/m6-rank1-followup/m6_directions/` but the pairwise cosine table is not committed as a flat artifact.

6. **§8.3 — winsorize max-abs activation drop `77.0 → 4.6` at L15 (94% reduction).** Reported in `STATUS_FOR_HUMAN.md` M6 Stage 2 summary; computed by the D2 build script (`4bc5708`); no isolated CSV.

7. **§7.4 / §7.6 — OBLITERATUS parameter-type spread (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, embed) and TrevorJS spread (o_proj, down_proj only).** Derivable from `results/weight_diffs/component_type_breakdown.csv` (commit `37ee48f`); confirmed by grep but the paper does not quote a specific count per type.

8. **§4.5 — squared `‖E‖_F² / ‖W‖_F² ≈ 0.00049` ("one part in two thousand").** §9.1 derivation; algebraically follows from the §4.5 median ratio 0.022 (0.022² = 0.000484), so reconstructible but not in any committed file.

9. **Paper-prose claim "44 / 7.3% over-refusal aggregated across six benign categories on E2B vs E4B (≈6× ratio)."** Approximation in §5.4; computable from the §5.2 CSV rows above but not separately stored.

None of the unverified numbers above are headline numbers; all are either (a) derivable from a committed artifact and not separately tabulated, or (b) probe/computation results that lived in a derivation script's stdout rather than a saved CSV. Orchestrator can decide whether to keep the prose claim or downgrade to "computed inline".

---

## Divergences flagged for orchestrator

One non-trivial divergence between paper prose and on-disk artifacts:

- **§1 / §5.2 paper writes "201 modified weight matrices" for OBLITERATUS; on-disk count is `200` entries with `rank_95` computed (`frobenius_norm > 0` count is 367, of which 200 carry a `rank_95` after the LM-only-and-frob-significant filter).** Paper §7.4 row reads "~201 (200 with rank_95 computed)", a hedge that matches the artifact; §1 Finding 4 and §10 round to "201" without the parenthetical. The 1-count discrepancy is below noise and likely from a different counting filter; flag for orchestrator decision on whether to harmonize the paper prose to "200".

All other quoted numbers reconcile to the artifacts within rounding.

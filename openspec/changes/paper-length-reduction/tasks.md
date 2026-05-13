## Operating notes

All paths below are relative to the `gb-paper` worktree root: `/home/nyavana/columbia/6699/gb-paper/`. Run commands from there.

Word budgets are **soft targets** (±15% per spec; ±30% triggers a tightening pass and operator surfacing). The global 10,500–13,500 band on `paper_short.md` is the actual constraint.

Work directly from `paper/sections/*.md` via `Read` and `Edit` against new copies in `paper/short/`. Do not modify the originals.

Each task ends with a commit using `feat(short):`, `refactor(short):`, or `fix(short):` prefixes.

**Headline numerics** to preserve verbatim across all cuts: `42 / 42`, `1 / 50`, `6 / 50`, `18 / 42`, `Cohen's *d* = 2.87`, `86.6 %`, `17 / 42`, `40.5 %`, `median rank_95 = 6`, `~201`, `84`, `−0.08`, `0.04`, `≈ 0.96`, `+0.93`, `−0.015`, `2560`.

**Figure-link convention.** Always write image links as `![alt](../../results/figures/<name>.png)` — markdown-relative from `paper/short/`. Never repo-root-relative `results/figures/<name>.png` (renders in some tools, breaks others).

**Citation truth set.** The canonical bibliography is `paper/references.md`. Use `[Park 2023]` / `[Zou 2023]` / `[Arditi 2024]` — no `[Mikolov 2013]`, no `[Park 2024]`.

## 1. Branch and folder scaffold

- [ ] 1.1 Verify clean working tree on `agent/writeup`: run `git status && git branch --show-current`; expected `clean` and `agent/writeup`. If dirty, stop and surface to the operator.
- [ ] 1.2 Cut feature branch: `git checkout -b feat/paper-short`; expected `Switched to a new branch 'feat/paper-short'`.
- [ ] 1.3 Create the empty target folder: `mkdir -p paper/short`; expected silent success.
- [ ] 1.4 Track the empty folder with a `.gitkeep` and commit: `touch paper/short/.gitkeep && git add paper/short/.gitkeep && git commit -m "feat(short): scaffold paper/short/ folder for length-cut version"`.

## 2. Abstract — verbatim copy

- [ ] 2.1 Read the abstract from `paper/paper.md` (the prose between `## Abstract` and the next `##` heading; approximately lines 9–11).
- [ ] 2.2 Write `paper/short/00_abstract.md` with the heading `## Abstract` plus the abstract prose copied verbatim. Do not edit a word.
- [ ] 2.3 Verify word count: `wc -w paper/short/00_abstract.md` should be 395 ± 5.
- [ ] 2.4 Numerics audit on the copy: `grep -oE "42 / 42|1 / 50|2\\.87|86\\.6|rank_95 = 6|≈ 0\\.04|17 / 42|40\\.5" paper/short/00_abstract.md | sort | uniq -c`; expected nonzero counts for whichever numbers appeared in the source. If any source number is missing, recopy.
- [ ] 2.5 Commit: `git add paper/short/00_abstract.md && git commit -m "feat(short): 00_abstract — verbatim copy of original abstract"`.

## 3. §1 Introduction (~1,100w from 2,107)

- [ ] 3.1 Read `paper/sections/01_introduction.md` (entire file, 47 lines).
- [ ] 3.2 Draft `paper/short/01_introduction.md` with four subsections:
  - `### 1.1 The hiking-emergency scenario` (~250w) — keep the Gemma 4 E2B refusal anecdote and the wilderness-medicine framing; cut the second elaboration paragraph; one sentence on over-refusal benchmarks [Röttger 2024]/[Cui 2024]. Compressed from old §1.1 (lines 3–9, ~360w).
  - `### 1.2 Research question` (~300w) — two questions (where does refusal live; can we remove it selectively); anchor on Arditi 2024 rank-1 finding; one sentence noting that on Gemma 4 the picture turns out partially true. Compressed from old §1.2 (lines 11–17, ~380w); drop the "selective-safety reduces to angle" sentence (§3 carries that).
  - `### 1.3 Contributions` (~400w) — five bullets, each ~70w (compressed from ~120w each), all five contributions retained. From old §1.5 (lines 35–43).
  - `### 1.4 What follows` (~150w) — short bridge paragraph gesturing at §2 background, §3 math tools, §4 four findings F1→F2→F4→F3, §5 discussion+conclusion. Replaces old §1.3 (preview), §1.4 (math framing), §1.6 (roadmap).
- [ ] 3.3 Word-count check: `wc -w paper/short/01_introduction.md` should be 950–1,250. If outside that band, tighten if over, accept if under.
- [ ] 3.4 Numerics audit: `grep -oE "Gemma 4 E2B-it|Arditi 2024|rank-1|over-refusal" paper/short/01_introduction.md | sort | uniq -c`; expected each term ≥ 1.
- [ ] 3.5 Commit: `git add paper/short/01_introduction.md && git commit -m "feat(short): 01_introduction — drop preview/math/roadmap, tighten remaining"`.

## 4. §2 Background and Related Work (~1,800w from 3,432)

- [ ] 4.1 Read both sources: `paper/sections/02_background.md` (35 lines, 1,281 words) and `paper/sections/03_related_work.md` (45 lines, 2,151 words).
- [ ] 4.2 Draft `paper/short/02_background_related.md` with six subsections:
  - `### 2.1 How alignment training installs refusal` (~250w) — RLHF → DPO → Constitutional AI in one paragraph; cite [Christiano 2017], [Ouyang 2022], [Rafailov 2023], [Bai 2022] inline. From old §2.1 (lines 5–17).
  - `### 2.2 Linear representations and the mean-difference estimator` (~250w) — linear-rep hypothesis as the justification for mean-difference; cite `[Park 2023]` and `[Zou 2023]` (canonical keys — no `[Mikolov 2013]` / `[Park 2024]`). From old §2.2 (lines 19–35).
  - `### 2.3 The abliteration lineage` (~400w) — anchor on Arditi 2024 + Mlabonne 2024; one sentence each on [p-e-w 2025], [elder-plinius 2025], [grimjim 2025], [TrevorS 2026]. From old §3.1 (lines 5–17); drop variant-by-variant pedagogical walkthrough.
  - `### 2.4 Over-refusal benchmarks` (~250w) — [Röttger 2024]/XSTest, [Cui 2024]/OR-Bench. From old §3.2 (lines 19–27); drop tutorial discussion of why over-refusal is a problem (intro carries that).
  - `### 2.5 Gemma 4 architectural quirks` (~300w) — local/global attention split, RMSNorm placement, shared K/V tensors in layers 24–41 (bears on §4.3, must stay). From old §3.3 (lines 29–45).
  - `### 2.6 OBLITERATUS and TrevorJS at a glance` (~200w) — high-level intro of the two §4.3 comparison points only.
- [ ] 4.3 Word-count check: `wc -w paper/short/02_background_related.md` should be 1,550–2,050.
- [ ] 4.4 Citation audit using literal-bracket extraction:
  ```bash
  grep -oE "\[[^]]+\]" paper/short/02_background_related.md \
    | grep -vE "^\[(http|//|#|results/|\.\./|\^|\\\\)" \
    | sort -u
  ```
  Expected: list contains at least `[Arditi 2024]`, `[Mlabonne 2024]`, `[Röttger 2024]`, `[Cui 2024]`, `[elder-plinius 2025]`, `[TrevorS 2026]`, `[Park 2023]`, `[Zou 2023]`, `[Gemma Team 2025]`.
- [ ] 4.5 Cross-check every cited key against `paper/references.md`:
  ```bash
  comm -23 \
    <(grep -oE "\[[^]]+\]" paper/short/02_background_related.md | grep -vE "^\[(http|//|#|results/|\.\./)" | sort -u) \
    <(grep -oE "\*\*\[[^]]+\]\*\*" paper/references.md | sed 's/\*\*//g' | sort -u)
  ```
  Expected: empty output. If non-empty, fix the citation in the draft (use canonical key), not `references.md`.
- [ ] 4.6 Commit: `git add paper/short/02_background_related.md && git commit -m "feat(short): 02_background_related — merge §2+§3, drop tutorial material"`.

## 5. §3 Mathematical Tools (~1,800w from ~5,800) — biggest cut

- [ ] 5.1 Read both sources: `paper/sections/04_math_framework.md` (117 lines, 3,534 words) and `paper/sections/09_discussion.md` lines 1–55 (covering §9.1, ~2,300 words).
- [ ] 5.2 Draft `paper/short/03_math_tools.md` with one short intro paragraph plus four subsections:
  - Intro paragraph (~50w): "We state each tool once and apply it once."
  - `### 3.1 Setup: the rank-1 weight edit in coordinates` (~250w) — define `W ← W − α·d·dᵀW`, scope (`o_proj` and `down_proj`), what α and d are; one line on `‖E‖_F = ‖E‖_2 = |α|·‖dᵀW‖_2` for rank-1 (Frobenius and spectral coincide). Replaces old §4.1 (lines 5–17).
  - `### 3.2 Mirsky's bound and the partial-response prediction` (~600w) — state Mirsky 1960 for rectangular matrices; one paragraph on why this beats Hoffman-Wielandt (rectangular vs square eigenvalue); apply: per-layer `‖E‖_F / ‖W‖_F` is in the 0.2–0.8% range, so first-order linearization should hold and a partial behavioral response is predicted. Embed `![Mirsky bound, per-layer relative perturbation](../../results/figures/mirsky_bound_heatmap_d3.png)`. Fuses old §4.2 (lines 19–35) with §9.1 Connection 2 (lines 15–29).
  - `### 3.3 Near-orthogonality of learned directions, with the anisotropy caveat` (~500w) — high-dim concentration fact (random unit vectors near-orthogonal in 2560-d); caveat in one sentence (applies to learned directions, NOT raw activations whose mean pairwise cosine at L15 is ≈ 0.96); apply: over-refuse-category directions cluster at +0.93 mutual cosine, −0.015 vs should-refuse. Fuses old §4.4 (lines 43–61) with §9.1 Connection 1 (lines 9–13).
  - `### 3.4 Lazy training / NTK as plausibility, not prediction` (~300w) — under small ‖E‖, behavior is approximately linear in the perturbation; soft motivation for "expecting a partial response", not quantitative theory; cite [Jacot 2018] once. Fuses old §4.6 (lines 82–90) with §9.1 Connection 4 (lines 35–43).
  - **Drop entirely:** old §4.3 (Lec 7 SVD machinery, lines 37–41), §4.5 (per-layer Mirsky-bound heatmap subsection, lines 63–80; the heatmap figure stays as a reference inside §3.2), §4.7 (closing forward-pointer, lines 92–106); §9.1 Connection 3 (lines 31–33; duplicates §4.3) and Connection 5 (lines 45–51; duplicates Connection 1).
- [ ] 5.3 Verify the figure path resolves: `(cd paper/short && [ -f ../../results/figures/mirsky_bound_heatmap_d3.png ] && echo "OK" || echo "MISSING")`. Expected `OK`.
- [ ] 5.4 Word-count check: `wc -w paper/short/03_math_tools.md` should be 1,500–2,100.
- [ ] 5.5 Numerics audit: `grep -oE "Mirsky|2560|0\\.96|\\+0\\.93|−0\\.015|0\\.2|0\\.8|Jacot" paper/short/03_math_tools.md | sort | uniq -c`; expected each present.
- [ ] 5.6 Commit: `git add paper/short/03_math_tools.md && git commit -m "feat(short): 03_math_tools — fold §4 + §9.1 into tool+application once"`.

## 6. §4.1 F1 — Behavioral baselines and the over-refusal pivot (~900w from 2,649, −66%)

- [ ] 6.1 Read `paper/sections/05_over_refusal.md` (86 lines, 2,649 words).
- [ ] 6.2 Write `paper/short/04_findings.md` with the section header and §4.1:
  - `## 4. Empirical Findings` + one short intro paragraph stating the F1→F2→F4→F3 order.
  - `### 4.1 F1 — Behavioral baselines and the over-refusal pivot` (~900w prose) — keep the headline refusal-rate numbers (E4B refuses 42/42 should_refuse, 1/50 emergency_medical; E2B refuses 6/50 emergency_medical, 18/42 gray_zone). One compact summary table (rows: should_refuse, emergency_medical, gray_zone, mental_health, chemistry_safety, wilderness_survival, home_safety, safe_control, TOTAL; columns: E2B base, E4B base — see table in old §5.2 lines 26–31, verify each cell). One sentence on the emergency-responder-framing probe: *"An emergency-responder-framing probe (prepending 'I am an emergency first responder' to the 50 emergency_medical prompts on base E4B) moved that single category from 1 / 50 to 2 / 50; the probe was not run on other categories and we make no claim about them."* Do NOT add a responder column to the table. Close with the framing-pivot paragraph (project rerouted from over-refusal mitigation to geometry).
  - **Drop:** old §5.1 benchmark structure (methodology, appendix material), §5.4 over-refusal axis (fold the load-bearing number into the table), §5.5 phrasing-sensitivity probe (not load-bearing for the four findings).
  - **Move to §4.2:** old §5.3 (headline negative finding) — that becomes part of F2.
- [ ] 6.3 Word-count check on §4.1: `wc -w paper/short/04_findings.md` should be 800–1,000 at this point (only §4.1 + header).
- [ ] 6.4 Numerics audit: `grep -oE "42 / 42|1 / 50|6 / 50|18 / 42|2 / 50" paper/short/04_findings.md | sort | uniq -c`; expected each ≥ 1.
- [ ] 6.5 Commit: `git add paper/short/04_findings.md && git commit -m "feat(short): 04_findings §4.1 — F1 over-refusal pivot, summary-table format"`.

## 7. §4.2 F2 — Activation-space rank-1 + standard-recipe null (~1,200w)

- [ ] 7.1 Read `paper/sections/06_mechanistic.md` (80 lines, 2,101 words), `paper/sections/07_abliteration_weight_diff.md` lines 1–40 (§7.1–7.3), and re-read `paper/sections/05_over_refusal.md` §5.3 (lines 44–52).
- [ ] 7.2 Append §4.2 to `paper/short/04_findings.md`:
  - `### 4.2 F2 — Refusal is rank-1 in activations but doesn't budge under the standard recipe` (~1,200w) — one-paragraph gloss on activation extraction; per-layer signal-strength scan with L15 peak and Cohen's *d* = 2.87; rank-1 PCA with 86.6% top-PC variance; then the negative behavioral result. State the sweep and the head-to-head as **separate evidence**:
    - α-sweep across α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0}: refusal rate flat at 30–35% on a stratified n=20 subsample.
    - Layer-subset sweep across **nine** partitions (all_42, global_only, sliding_only, first_half, second_half, last_10, middle_14, peak_band_4_17, peak_layer_15_only) at α=1.0 on the same n=20: flat at 25–35%.
    - Random-direction control: 30%, indistinguishable from α=0.
    - Direct n=48 head-to-head on should_refuse: self-abliterated 6/6 = 100%, base 42/42 = 100%, delta = 0 percentage points.
  - Embed figures: `![Per-layer signal strength, refuse vs comply](../../results/figures/signal_vs_layer.png)` and `![PCA variance per layer](../../results/figures/pca_variance_per_layer.png)`.
  - **Drop:** old §6.4 category geometry (its numbers +0.93, −0.015 live in §3.3), §6.5 anisotropy (its ≈ 0.96 lives in §3.3), §6.6 summary, §7.1 intervention code-level walk (formula is in §3.1).
- [ ] 7.3 Word-count delta check: `wc -w paper/short/04_findings.md` should be ~2,000–2,200.
- [ ] 7.4 Numerics audit: `grep -oE "L15|Cohen|2\\.87|86\\.6|α-sweep|9.partition|nine.partition|random-direction|6 / 6|42 / 42|n.=.48|n.=.20" paper/short/04_findings.md | sort | uniq -c`; expected each term ≥ 1. Both `30–35%` and `25–35%` must appear; partition count is **nine** not seven.
- [ ] 7.5 Verify figures resolve: `(cd paper/short && [ -f ../../results/figures/signal_vs_layer.png ] && [ -f ../../results/figures/pca_variance_per_layer.png ] && echo OK)`. Expected `OK`.
- [ ] 7.6 Commit: `git add paper/short/04_findings.md && git commit -m "feat(short): 04_findings §4.2 — F2 activation-space rank-1 + standard-recipe null"`.

## 8. §4.3 F4 — Two abliterations land in orthogonal subspaces (~1,200w)

- [ ] 8.1 Read `paper/sections/07_abliteration_weight_diff.md` lines 35–89 (§7.4–7.6, 2,465 words).
- [ ] 8.2 Append §4.3 to `paper/short/04_findings.md`:
  - `### 4.3 F4 — Two published abliterations land in orthogonal subspaces` (~1,200w) — OBLITERATUS rank profile (median `rank_95 = 6` across `~201` modified tensors); TrevorJS rank profile (pure rank-1 across 84 matrices, norm-preserving biprojection); cross-method top-1 left-singular-vector cosine = −0.08, near-orthogonal in the §3.3 sense; cross-reference to §4.2 refusal direction: |cosine| ≈ 0.04 for both methods (neither weight-edit aligns with activation-space refusal); one paragraph on Gemma 4 architectural quirks that bear on the diff (local/global attention, shared K/V layers 24–41).
  - Embed figures: `![Cross-method singular vectors](../../results/figures/cross_method_singular_vectors.png)`, `![Refusal direction vs singular vector](../../results/figures/refusal_direction_vs_singular_vector.png)`, `![OBLITERATUS modification ranks](../../results/figures/obliteratus_modification_ranks.png)`.
  - Close with three bullets: (a) the two methods modify weights in nearly orthogonal directions; (b) neither aligns with the activation-space refusal direction of §4.2; (c) one uses multi-rank descent (rank_95 = 6), the other pure rank-1; both are effective in the wild.
  - **Drop:** per-tensor SVD walkthrough, long history of each method's development (§2.3 carries the intro), §7.7 summary (close inline).
- [ ] 8.3 Word-count delta check: `wc -w paper/short/04_findings.md` should be ~3,200–3,400.
- [ ] 8.4 Numerics audit: `grep -oE "rank_95 = 6|~201|84|−0\\.08|0\\.04|biprojection" paper/short/04_findings.md | sort | uniq -c`; expected each present.
- [ ] 8.5 Verify figures resolve from `paper/short/`: `(cd paper/short && for f in ../../results/figures/cross_method_singular_vectors.png ../../results/figures/refusal_direction_vs_singular_vector.png ../../results/figures/obliteratus_modification_ranks.png; do [ -f "$f" ] || echo "MISSING: $f"; done)`.
- [ ] 8.6 Commit: `git add paper/short/04_findings.md && git commit -m "feat(short): 04_findings §4.3 — F4 cross-method weight-diff orthogonality"`.

## 9. §4.4 F3 — Causal-isolation cascade with D3 stacked partial (~1,500w from 2,334)

This is the experimental contribution and gets the smallest relative cut. D3 stacked variant (H2 + H3 + H4) keeps full treatment; H1, H2, H3, H5 collapse to one combined paragraph.

**Critical phrasing constraint** (from `paper/sections/07_abliteration_weight_diff.md:35`): the prose must say D3 is *"necessary in combination with prior ingredients to produce the partial 40.5% result; we do not claim sufficient — Stage 2.5 unstacked isolation was not run, so we cannot rule out the possibility that D1 or D2 is doing the real work and the Gram-Schmidt layer is decorative."* Do NOT write "isolates Gram-Schmidt as load-bearing" or "the load-bearing ingredient is H4".

- [ ] 9.1 Read `paper/sections/08_rank1_cascade.md` (54 lines, 2,334 words); confirm phrasing against `paper/sections/07_abliteration_weight_diff.md:35`.
- [ ] 9.2 Append §4.4 to `paper/short/04_findings.md`:
  - `### 4.4 F3 — A causal-isolation cascade identifies D3 as the smallest stacked variant that moves the gate`
  - Setup (~150w): recap §4.2's null and the question it raised.
  - `#### Hypothesis taxonomy` (~200w) — compact table or bullets listing H1 (bnb int8), H2 (chat-template direction), H3 (winsorization), H4 (Gram-Schmidt against harmless mean), H5 (norm-preserving biprojection), one-clause descriptions.
  - `#### What was ruled out` (~150w combined) — one sentence each:
    - H1 (bnb int8 in-place edit): bf16 edit path reproduces the int8 failure character-for-character, ruling out quantization.
    - H2 (chat-template direction alone): not tested unstacked.
    - H3 (winsorization alone): not tested unstacked.
    - H5 (norm-preserving biprojection): vanilla projection changes per-row L2 norms by mean 0.038%, well below the RMSNorm sensitivity threshold, so H5 is not needed.
  - `#### D3 — the smallest stacked variant that moves the gate` (~600w) — D3 = chat-template direction (D1) + per-layer winsorization at 99.5% (D2) + two-pass Gram-Schmidt against harmless mean (D3), applied via the same vanilla rank-1 projection at α=1.0 across all 42 layers as §4.2. Result: 17/42 = 40.5% should_refuse refusal at n=42, down from base 42/42 = 100%. Residual concentrates on extreme topics (CSAM, ICS/hospital malware, weapons). State the necessary-not-sufficient phrasing verbatim per the critical phrasing constraint above. Per-prompt breakdown if word budget permits.
  - Embed figures: `![Cascade gate](../../results/figures/m6_cascade_gate.png)` and `![Per-prompt n=42](../../results/figures/m6_perprompt_n42.png)`.
  - `#### Implications` (~250w) — "partial first-order behavioral response under small rank-1 perturbation, consistent with §3.2's Mirsky-bound prediction. Residual ≈ 60% indicates the refusal manifold is not fully one-dimensional in weight space on Gemma 4, even when it is in activation space (§4.2). We characterize D3 as the smallest stacked direction-build recipe that produces the partial effect; we make no claim about which individual ingredient is load-bearing."
- [ ] 9.3 Word-count delta check: `wc -w paper/short/04_findings.md` should be ~4,700–4,900.
- [ ] 9.4 Numerics audit on §4.4: `grep -oE "17 / 42|40\\.5|Gram-Schmidt|D3|CSAM|H1|H2|H3|H4|H5|biprojection|0\\.038" paper/short/04_findings.md | sort | uniq -c`; expected each present.
- [ ] 9.5 Sufficiency-disclaimer audit: `grep -F "we do not claim sufficient" paper/short/04_findings.md`; expected the phrase to appear (or equivalent prose containing both "necessary" and an explicit "not sufficient" / "not claim sufficient" about D3). Run `grep -E "isolates Gram-Schmidt as load-bearing|load-bearing ingredient is H4" paper/short/04_findings.md`; expected empty.
- [ ] 9.6 Verify figures resolve: `(cd paper/short && [ -f ../../results/figures/m6_cascade_gate.png ] && [ -f ../../results/figures/m6_perprompt_n42.png ] && echo OK)`. Expected `OK`.
- [ ] 9.7 Commit: `git add paper/short/04_findings.md && git commit -m "feat(short): 04_findings §4.4 — F3 cascade, compress rejected H1/H2/H3/H5"`.

## 10. §4.5 — Connecting paragraph (~200w)

- [ ] 10.1 Append §4.5 to `paper/short/04_findings.md`:
  - `### 4.5 What the four findings say together` (~200w synthesis) — refusal is rank-1 in activations (F2) but multi-rank in weights (F3 cascade + F4 cross-method); the over-refusal motivation that opened the project (F1) turned out smaller than the smaller-model baseline suggested; the question the paper now answers is "what does and does not transfer from the activation-space rank-1 picture to weight-space removal".
- [ ] 10.2 Final §4 word-count check: `wc -w paper/short/04_findings.md` should be 4,800–5,200. If over 5,500, surface to operator and tighten §4.1 or §4.3.
- [ ] 10.3 Commit: `git add paper/short/04_findings.md && git commit -m "feat(short): 04_findings §4.5 — F1-F4 connecting synthesis"`.

## 11. §5 Discussion and Conclusion (~1,800w from ~3,700)

- [ ] 11.1 Read both sources: `paper/sections/09_discussion.md` lines 53–96 (§9.2–9.5) and `paper/sections/10_conclusion.md` (35 lines, 1,873 words).
- [ ] 11.2 Draft `paper/short/05_discussion_conclusion.md` with four subsections:
  - `### 5.1 Limitations` (~550w including the over-parametrization paragraph at the close) — residual 60% concentrating on extreme topics; n = 42 small-sample caveat; Gemma 4 specifics; one paragraph on the over-parametrization framing (folded from old §9.2 lines 53–59).
  - `### 5.2 Future work` (~150w) — pointers for follow-up: rank-2/rank-3 weight edits, multi-direction selective safety, larger n. From old §9.4 (lines 75–77).
  - `### 5.3 Ethics` (~500w) — dual-use surface of abliteration research; constructive selective-safety angle; what the residual 60% means for misuse risk (it does NOT make weights safe-against-abliteration; the methods that work in the wild still work). From old §10.3 (lines 23–31).
  - `### 5.4 Conclusion` (~600w) — recap the four findings; the central empirical claim (rank-1 in activations, multi-rank in weights on Gemma 4); the methodological contribution (cascade structure); the framing pivot (over-refusal smaller than expected on E4B). From old §10.1 (lines 5–17).
  - **Drop:** old §9.1 (already moved to §3), §9.5 closing (duplicates conclusion intro), §10.2 "What we did not expect" (narrative duplicate of §4.5), §10.4 closing.
- [ ] 11.3 Word-count check: `wc -w paper/short/05_discussion_conclusion.md` should be 1,600–2,000.
- [ ] 11.4 Lecture-connection leakage check: `grep -E "Connection [1-5]|Lec [5-9]|lecture connection" paper/short/05_discussion_conclusion.md`; expected empty.
- [ ] 11.5 Commit: `git add paper/short/05_discussion_conclusion.md && git commit -m "feat(short): 05_discussion_conclusion — merge §9.2-9.5 + §10, drop §9.1 leakage"`.

## 12. References — copy and prune orphans

- [ ] 12.1 Copy verbatim: `cp paper/references.md paper/short/references.md`.
- [ ] 12.2 Build the list of citation keys actually used in the new paper (literal-bracket extractor handles multi-word keys, accented chars, non-year keys; filter out non-citation brackets):
  ```bash
  grep -hoE "\[[^]]+\]" paper/short/0[0-5]*.md \
    | grep -vE "^\[(http|//|#|\.\./|\^|\\\\|results/)" \
    | sort -u > /tmp/cited_keys.txt
  ```
- [ ] 12.3 Build the list of citation keys defined in references.md:
  ```bash
  grep -oE "\*\*\[[^]]+\]\*\*" paper/short/references.md \
    | sed 's/\*\*//g' \
    | sort -u > /tmp/defined_keys.txt
  ```
- [ ] 12.4 Identify orphans: `comm -23 /tmp/defined_keys.txt /tmp/cited_keys.txt > /tmp/orphan_keys.txt && cat /tmp/orphan_keys.txt`. Expected 0–10 keys.
- [ ] 12.5 For each orphan key, find the `- **[Key]** …` bullet in `paper/short/references.md` and delete the whole bullet (plus any continuation lines). Preserve alphabetical order. Skip this step if `/tmp/orphan_keys.txt` is empty. **No in-line rewriting of retained bullets.**
- [ ] 12.6 Verify no cited key is missing: `comm -23 /tmp/cited_keys.txt /tmp/defined_keys.txt > /tmp/missing_keys.txt && cat /tmp/missing_keys.txt`; expected empty. If non-empty, the citation is wrong or over-pruning happened — fix it before committing.
- [ ] 12.7 Commit: `git add paper/short/references.md && git commit -m "feat(short): references — copy from canonical and prune orphans"`.

## 13. Assemble `paper_short.md`

- [ ] 13.1 Build `paper/short/paper_short.md` by concatenation:
  ```bash
  {
    echo "# Geometry of Alignment"
    echo ""
    echo "**Where Safety Behavior Lives in Gemma 4 E4B-it, and What a Rank-1 Weight Edit Can and Cannot Remove**"
    echo ""
    echo "*EECS 6699 (Mathematics of Deep Learning), Columbia University, Spring 2026*"
    echo ""
    echo "*Date: 2026-05-12*"
    echo ""
    cat paper/short/00_abstract.md
    echo ""
    echo "## Table of Contents"
    echo ""
    echo "- [1. Introduction](#1-introduction)"
    echo "- [2. Background and Related Work](#2-background-and-related-work)"
    echo "- [3. Mathematical Tools](#3-mathematical-tools)"
    echo "- [4. Empirical Findings](#4-empirical-findings)"
    echo "- [5. Discussion and Conclusion](#5-discussion-and-conclusion)"
    echo "- [References](#references)"
    echo ""
    cat paper/short/01_introduction.md
    echo ""
    cat paper/short/02_background_related.md
    echo ""
    cat paper/short/03_math_tools.md
    echo ""
    cat paper/short/04_findings.md
    echo ""
    cat paper/short/05_discussion_conclusion.md
    echo ""
    sed 's/^# References$/## References/' paper/short/references.md
  } > paper/short/paper_short.md
  ```
- [ ] 13.2 Sanity-check structure: `head -20 paper/short/paper_short.md && echo "---" && grep -n "^## " paper/short/paper_short.md`. Expected title + abstract + TOC at top; `## 1. Introduction`, `## 2. Background…`, …, `## References` in order.
- [ ] 13.3 Full-assembly word count: `wc -w paper/short/paper_short.md`. Expected 10,500–13,500. If outside the band, surface to operator before continuing to Task 14.
- [ ] 13.4 Commit: `git add paper/short/paper_short.md && git commit -m "feat(short): paper_short.md — assemble final cut version"`.

## 14. Acceptance audit

- [ ] 14.1 Confirm originals untouched: `git diff agent/writeup..feat/paper-short -- paper/sections/ paper/paper.md paper/references.md | head -5`. Expected empty output. If non-empty, revert with `git checkout agent/writeup -- paper/sections/ paper/paper.md paper/references.md` and re-commit.
- [ ] 14.2 Confirm all expected files exist: `ls paper/short/`. Expected: `00_abstract.md`, `01_introduction.md`, `02_background_related.md`, `03_math_tools.md`, `04_findings.md`, `05_discussion_conclusion.md`, `references.md`, `paper_short.md`, `.gitkeep`.
- [ ] 14.3 Headline numerics audit on the assembled file:
  ```bash
  for pattern in "42 / 42" "1 / 50" "Cohen's \\*d\\* = 2\\.87" "86\\.6" "17 / 42" "40\\.5" "rank_95 = 6" "~201" "−0\\.08" "0\\.04" "6 / 6" "30–35%" "25–35%"; do
    count=$(grep -cE "$pattern" paper/short/paper_short.md)
    echo "$pattern : $count"
  done
  ```
  Expected: every count ≥ 1. If any is 0, restore from the original section file.
- [ ] 14.4 Figure-path audit (markdown-relative resolution from `paper/short/`):
  ```bash
  (
    cd paper/short
    grep -oE "\!\[[^]]*\]\([^)]+\)" paper_short.md \
      | sed -E 's/.*\(([^)]+)\).*/\1/' \
      | while read -r rel; do
          [ -f "$rel" ] || echo "MISSING (resolved from paper/short/): $rel"
        done
  )
  ```
  Expected: no `MISSING` lines. Then check link-form conformance:
  ```bash
  grep -oE "\!\[[^]]*\]\([^)]+\)" paper/short/paper_short.md \
    | sed -E 's/.*\(([^)]+)\).*/\1/' \
    | grep -vE "^\.\./\.\./results/figures/" \
    | grep -vE "^https?://" \
    || echo "(no non-conforming links)"
  ```
  Expected: `(no non-conforming links)`. Rewrite any printed link to the `../../results/figures/…` form.
- [ ] 14.5 Section-level word counts: `wc -w paper/short/0[0-5]*.md`. Expected (soft): `00_abstract.md` ≈ 390, `01_introduction.md` ≈ 1,100, `02_background_related.md` ≈ 1,800, `03_math_tools.md` ≈ 1,800, `04_findings.md` ≈ 5,000, `05_discussion_conclusion.md` ≈ 1,800. If any single section is more than 30% off target and reads bloated, do one tightening pass; otherwise accept.
- [ ] 14.6 Orphan / missing reference audit:
  ```bash
  grep -hoE "\[[^]]+\]" paper/short/0[1-5]*.md \
    | grep -vE "^\[(http|//|#|\.\./|\^|\\\\|results/)" \
    | sort -u > /tmp/cited.txt
  grep -oE "\*\*\[[^]]+\]\*\*" paper/short/references.md \
    | sed 's/\*\*//g' \
    | sort -u > /tmp/defined.txt
  comm -23 /tmp/defined.txt /tmp/cited.txt > /tmp/orphans.txt
  comm -23 /tmp/cited.txt /tmp/defined.txt > /tmp/missing.txt
  echo "Orphans (defined but not cited):"; cat /tmp/orphans.txt
  echo "Missing (cited but not defined):"; cat /tmp/missing.txt
  ```
  Expected: both `/tmp/orphans.txt` and `/tmp/missing.txt` empty. If `missing.txt` non-empty, add the entry by copying from `paper/references.md`, or fix the citation to use the canonical key. If `orphans.txt` non-empty, re-run the Task 12 prune step.
- [ ] 14.7 Optional summary commit if clean-up was needed: `git add paper/short/ && git commit -m "fix(short): acceptance-audit clean-up (figure paths / orphans / numerics)"`. Skip if everything was clean on first pass.
- [ ] 14.8 Push the branch and surface to the operator:
  ```bash
  git push -u origin feat/paper-short
  ```
  Report: total assembled word count, per-section word counts, branch name and push status, any judgment calls made during cuts that warrant operator review.

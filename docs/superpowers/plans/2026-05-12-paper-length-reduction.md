# Paper Length Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a roughly half-length version of `paper/paper.md` in a new `paper/short/` folder, preserving the original draft untouched.

**Architecture:** Pure prose / structural rewrite of an existing markdown paper. Six section files plus an assembled `paper_short.md` go into a new sibling folder. No source code changes, no experiments rerun, no figures regenerated. The cut is driven by (a) merging the §4↔§9.1 duplicate math walkthrough, (b) compressing four rejected cascade hypotheses to one paragraph, (c) trimming tutorial background. Work happens on a feature branch off `agent/writeup` in the `gb-paper` worktree.

**Tech Stack:** Plain markdown. `wc -w` for word counts. `grep` for numerics audit. `git` for commits. No build system.

**Spec:** `docs/superpowers/specs/2026-05-12-paper-length-reduction-design.md`. Read it before starting — every "what to keep / drop" decision and its reason lives there. This plan operationalizes it but does not restate the rationale.

**Operating notes for the implementing agent:**

- All paths in this plan are relative to the `gb-paper` worktree root: `/home/nyavana/columbia/6699/gb-paper/`. Run commands from that directory.
- Word budgets in this plan are **soft targets**, matching the spec. If a section reads better at ±15% of target, that wins over hitting the number.
- For prose cuts, work directly from the original section files via `Read` and `Edit` against the new copy in `paper/short/`. Do not modify `paper/sections/*.md` or `paper/paper.md`.
- Each task ends with a commit. Use `feat(short):`, `refactor(short):`, etc. prefixes; conventional-commits format matches the repo.
- "Headline numerics" list, preserve verbatim across all cuts: `42 / 42`, `1 / 50`, `6 / 50`, `18 / 42`, `Cohen's *d* = 2.87`, `86.6 %`, `17 / 42`, `40.5 %`, `median rank_95 = 6`, `~201`, `84`, `−0.08`, `0.04`, `≈ 0.96`, `+0.93`, `−0.015`, `2560`.

---

## Task 0: Branch + folder scaffold

**Files:**
- Create: `paper/short/` (folder)
- Modify (git only): branch `feat/paper-short` off `agent/writeup`

- [ ] **Step 1: Verify clean working tree on agent/writeup**

Run: `git status && git branch --show-current`
Expected: `clean — nothing to commit` and `agent/writeup`.
If dirty, stop and surface to the operator.

- [ ] **Step 2: Cut a feature branch**

Run: `git checkout -b feat/paper-short`
Expected: `Switched to a new branch 'feat/paper-short'`

- [ ] **Step 3: Create the empty target folder**

Run: `mkdir -p paper/short`
Expected: silent success. `ls paper/short/` should be empty.

- [ ] **Step 4: Commit the scaffold**

Add a `.gitkeep` so the empty folder is tracked, then commit.
```bash
touch paper/short/.gitkeep
git add paper/short/.gitkeep
git commit -m "feat(short): scaffold paper/short/ folder for length-cut version"
```

---

## Task 1: Abstract — copy unchanged

**Files:**
- Create: `paper/short/00_abstract.md`
- Source: `paper/paper.md` lines 9–11

The existing abstract is 393 words and lives at the top of `paper/paper.md`. The spec says keep as-is, light touch-up only if cuts orphan a number (they should not).

- [ ] **Step 1: Read the abstract from `paper/paper.md`**

Use `Read` on `paper/paper.md` lines 9–12. The abstract is the prose between `## Abstract` and the next `##` heading.

- [ ] **Step 2: Write `paper/short/00_abstract.md`**

Create the file with the heading `## Abstract` plus the exact abstract prose copied verbatim. Do not edit a word.

- [ ] **Step 3: Verify word count is unchanged**

Run: `wc -w paper/short/00_abstract.md`
Expected: 395 ± 5 (header words add a tiny delta over the spec's 393-word body count).

- [ ] **Step 4: Verify all headline numerics survived the copy**

Run: `grep -oE "42 / 42|1 / 50|2\\.87|86\\.6|rank_95 = 6|≈ 0\\.04|17 / 42|40\\.5" paper/short/00_abstract.md | sort | uniq -c`
Expected: at least the numbers that appear in the original abstract (counts may vary; what matters is non-zero matches for the numbers present in the source). If a number that appeared in the source is missing from your copy, you mistyped — recopy.

- [ ] **Step 5: Commit**

```bash
git add paper/short/00_abstract.md
git commit -m "feat(short): 00_abstract — verbatim copy of original abstract"
```

---

## Task 2: §1 Introduction

**Files:**
- Create: `paper/short/01_introduction.md`
- Source: `paper/sections/01_introduction.md` (2,107 words, six subsections)
- Target: ~1,100 words

**What to keep (from the spec):**
- The hiking-emergency motivation paragraph (currently `01_introduction.md` §1 "A hiking-emergency scenario", lines 3–9). Compress from ~360w to ~250w by removing the second-paragraph elaboration and tightening the [Röttger 2024] / [Cui 2024] over-refusal-benchmark mention to one sentence.
- The research-question framing with the Arditi 2024 anchor (currently §2 "Research question and approach", lines 11–17). Compress from ~380w to ~300w; drop the "selective-safety reduces to angle" sentence — §3 carries that.
- The condensed contributions list (currently §5 "Contributions", lines 35–43). Keep all five contributions; tighten each bullet to ~70w from ~120w. Total ~400w.
- One short bridge paragraph (~150w) summarizing how the paper's structure follows from the contributions. This replaces the current §3/§4/§6 transitional subsections.

**What to drop (from the spec):**
- §3 "Preview of empirical findings" (lines 19–29). Reason: abstract already names F1–F4.
- §4 "Mathematical framing" (lines 31–33). Reason: new §3 introduces math directly.
- §6 "Roadmap" (lines 45–47). Reason: TOC handles it.

- [ ] **Step 1: Read the original**

Use `Read` on `paper/sections/01_introduction.md` (entire file, 47 lines).

- [ ] **Step 2: Draft the compressed introduction in `paper/short/01_introduction.md`**

Structure:
```markdown
## 1. Introduction

### 1.1 The hiking-emergency scenario
<~250w — keep the Gemma 4 E2B refusal anecdote and the wilderness-medicine framing; cut the second elaboration paragraph; one sentence on over-refusal benchmarks>

### 1.2 Research question
<~300w — two questions (where does refusal live; can we remove it selectively); anchor on Arditi 2024 rank-1 finding; one sentence that on Gemma 4 the picture turns out partially true>

### 1.3 Contributions
<~400w — five bullets, each ~70w, matching the spec's contributions list>

### 1.4 What follows
<~150w — short bridge paragraph: §2 background, §3 math tools, §4 four findings F1→F2→F4→F3, §5 discussion+conclusion; do NOT enumerate each section, just gesture>
```

Write the actual prose. No `<placeholder>` text in the committed file.

- [ ] **Step 3: Word-count check**

Run: `wc -w paper/short/01_introduction.md`
Expected: 950–1,250.
If outside that band, decide: tighten further if over; let it be if under (under-budget is fine per spec).

- [ ] **Step 4: Numerics audit**

Run: `grep -oE "Gemma 4 E2B-it|Arditi 2024|rank-1|over-refusal" paper/short/01_introduction.md | sort | uniq -c`
Expected: at least one occurrence of each. If any is zero, the cut went too deep — restore.

- [ ] **Step 5: Commit**

```bash
git add paper/short/01_introduction.md
git commit -m "feat(short): 01_introduction — drop preview/math/roadmap, tighten remaining"
```

---

## Task 3: §2 Background & Related Work — merge

**Files:**
- Create: `paper/short/02_background_related.md`
- Sources:
  - `paper/sections/02_background.md` (1,281 words, two subsections)
  - `paper/sections/03_related_work.md` (2,151 words, three subsections)
- Target: ~1,800 words

**What to keep:**
- From `02_background.md` §1 "Foundational Alignment" (lines 5–17): compress to ~250w. One paragraph covering RLHF → DPO → Constitutional AI as the standard pipeline that installs refusal behavior. Cite [Christiano 2017], [Ouyang 2022], [Rafailov 2023], [Bai 2022] inline.
- From `02_background.md` §2 "Linear Representations" (lines 19–35): compress to ~250w. State the linear-representation hypothesis and how it justifies the mean-difference direction estimator. Cite [Mikolov 2013], [Park 2024].
- From `03_related_work.md` §1 "The Abliteration Lineage" (lines 5–17): compress to ~400w. Anchor on Arditi 2024 and Mlabonne 2024; one sentence each on [p-e-w 2025], [elder-plinius 2025], [grimjim 2025], [TrevorS 2026] as the published variants. Drop the variant-by-variant pedagogical walkthrough.
- From `03_related_work.md` §2 "Over-Refusal: Why the Question Matters" (lines 19–27): compress to ~250w. [Röttger 2024]/XSTest and [Cui 2024]/OR-Bench. Drop tutorial discussion of why over-refusal is a problem (intro carries that).
- From `03_related_work.md` §3 "Gemma 4 Architectural Quirks" (lines 29–45): compress to ~300w. Local/global attention split, RMSNorm placement, shared K/V tensors in layers 24–41. These bear on §4.3 so they must stay.
- A brief intro of OBLITERATUS and TrevorJS as the two comparison points used in §4 — fold into the abliteration-lineage subsection (~200w).

**What to drop:**
- Deep RLHF tutorial (loss functions, PPO mechanics) — keep one sentence.
- Representation-engineering line beyond linear-rep hypothesis.
- History of abliteration variants the paper does not compare against in §4 (e.g., extended p-e-w discussion).
- Per-section References blocks at the bottom of the source files — the canonical `references.md` carries them.

- [ ] **Step 1: Read both source files**

Use `Read` on `paper/sections/02_background.md` (35 lines) and `paper/sections/03_related_work.md` (45 lines).

- [ ] **Step 2: Draft `paper/short/02_background_related.md`**

Structure:
```markdown
## 2. Background and Related Work

### 2.1 How alignment training installs refusal
<~250w — RLHF/DPO/Constitutional AI sweep>

### 2.2 Linear representations and the mean-difference estimator
<~250w — linear-rep hypothesis as justification>

### 2.3 The abliteration lineage
<~400w — Arditi 2024, Mlabonne 2024 core recipe; one sentence on each of [p-e-w 2025], [elder-plinius 2025], [grimjim 2025] / [TrevorS 2026]>

### 2.4 Over-refusal benchmarks
<~250w — [Röttger 2024] XSTest, [Cui 2024] OR-Bench>

### 2.5 Gemma 4 architectural quirks
<~300w — local/global attention, RMSNorm, shared K/V layers 24-41>

### 2.6 OBLITERATUS and TrevorJS at a glance
<~200w — set up the two §4.3 comparison points; high-level method description only>
```

Write the actual prose. Cite each `[Author Year]` inline using the format established in `paper/references.md` (alphabetical-key, no parens).

- [ ] **Step 3: Word-count check**

Run: `wc -w paper/short/02_background_related.md`
Expected: 1,550–2,050.

- [ ] **Step 4: Citation audit**

Run: `grep -oE "\\[(Arditi|Mlabonne|Röttger|Cui|p-e-w|elder-plinius|grimjim|TrevorS|Christiano|Ouyang|Rafailov|Bai|Mikolov|Park) [0-9]{4}" paper/short/02_background_related.md | sort | uniq`
Expected: at least Arditi 2024, Mlabonne 2024, Röttger 2024, Cui 2024, elder-plinius 2025, TrevorS 2026 (these are load-bearing for later sections). Missing any → restore.

- [ ] **Step 5: Commit**

```bash
git add paper/short/02_background_related.md
git commit -m "feat(short): 02_background_related — merge §2+§3, drop tutorial material"
```

---

## Task 4: §3 Mathematical Tools — biggest cut

**Files:**
- Create: `paper/short/03_math_tools.md`
- Sources:
  - `paper/sections/04_math_framework.md` (3,534 words, seven subsections)
  - `paper/sections/09_discussion.md` §9.1 "Five lecture connections" (lines 5–51, ~2,300 words)
- Target: ~1,800 words

The structure is "tool, then immediate application". Three tools, each stated once and applied to a specific finding. See spec §6 (§3 Mathematical Tools) for the exact subsection layout.

**Subsection map and word budgets:**

| New subsection | Replaces | Target |
|---|---|---:|
| 3.1 Setup | old §4.1 (lines 5–17) | ~250w |
| 3.2 Mirsky's bound and partial-response prediction | old §4.2 (lines 19–35) + §9.1 Connection 2 (lines 15–29) | ~600w |
| 3.3 Near-orthogonality with anisotropy caveat | old §4.4 (lines 43–61) + §9.1 Connection 1 (lines 9–13) | ~500w |
| 3.4 Lazy training / NTK as plausibility | old §4.6 (lines 82–90) + §9.1 Connection 4 (lines 35–43) | ~300w |

**What to drop entirely:**
- `04_math_framework.md` §4.3 "Lec 7 SVD machinery" (lines 37–41). The only load-bearing fact (`‖E‖_F = ‖E‖_2` for rank-1) goes into §3.2 in one line.
- `04_math_framework.md` §4.5 "Per-layer Mirsky-bound heatmaps" (lines 63–80). The 0.2–0.8% number lives in §3.2 as one sentence; the heatmap figure stays as a `![](results/figures/mirsky_bound_heatmap_*.png)` reference inside §3.2.
- `04_math_framework.md` §4.7 "Closing" forward-pointer (lines 92–106). TOC handles it.
- `09_discussion.md` §9.1 Connection 3 (lines 31–33). Duplicates the dropped §4.3.
- `09_discussion.md` §9.1 Connection 5 (lines 45–51). Duplicates Connection 1.
- Any `## References` block at the bottom of source files.

- [ ] **Step 1: Read both sources**

Use `Read` on `paper/sections/04_math_framework.md` (117 lines) and `paper/sections/09_discussion.md` lines 1–55 (covering §9.1).

- [ ] **Step 2: Draft `paper/short/03_math_tools.md`**

Structure:
```markdown
## 3. Mathematical Tools

The empirical sections cite three results from matrix-perturbation theory and high-dim geometry. We state each tool once and apply it once.

### 3.1 Setup: the rank-1 weight edit in coordinates
<~250w — define W ← W − α·d·dᵀW, scope (o_proj and down_proj), what α and d are; one line: ‖E‖_F = ‖E‖_2 = |α|·‖dᵀW‖_2 for rank-1 because rank-1 makes Frobenius and spectral coincide>

### 3.2 Mirsky's bound and the partial-response prediction
<~600w — state Mirsky 1960 for rectangular matrices; one paragraph on why this beats Hoffman-Wielandt (rectangular vs square eigenvalue); apply: per-layer ‖E‖_F / ‖W‖_F is in the 0.2–0.8% range, so first-order linearization should hold and a partial behavioral response is predicted; embed Mirsky-bound heatmap figure>

### 3.3 Near-orthogonality of learned directions, with the anisotropy caveat
<~500w — high-dim concentration fact (random unit vectors near-orthogonal in 2560-d); the caveat in one sentence (applies to learned directions, NOT raw activations whose mean pairwise cosine at L15 is ≈ 0.96); apply: over-refuse-category directions cluster at +0.93 mutual cosine, −0.015 vs should-refuse>

### 3.4 Lazy training / NTK as plausibility, not prediction
<~300w — one paragraph: under small ‖E‖, behavior is approximately linear in the perturbation; frame as soft motivation for "expecting a partial response", not quantitative theory; cite [Jacot 2018] once>
```

Embed exactly one figure reference: `![Mirsky bound, per-layer relative perturbation](../../results/figures/mirsky_bound_heatmap_d3.png)` inside §3.2. (Path is relative to `paper/short/`. Double-check the path resolves before committing.)

- [ ] **Step 3: Verify the figure path resolves**

Run: `ls paper/../results/figures/mirsky_bound_heatmap_d3.png`
Expected: file listed.
If the path doesn't resolve from where the markdown will be read, adjust to `../../results/figures/mirsky_bound_heatmap_d3.png` or whichever form works for the eventual renderer. The existing `paper/paper.md` uses `results/figures/...` from the repo root — match that convention.

- [ ] **Step 4: Word-count check**

Run: `wc -w paper/short/03_math_tools.md`
Expected: 1,500–2,100.

- [ ] **Step 5: Numerics audit**

Run: `grep -oE "Mirsky|2560|0\\.96|\\+0\\.93|−0\\.015|0\\.2|0\\.8|Jacot" paper/short/03_math_tools.md | sort | uniq -c`
Expected: every term present at least once.

- [ ] **Step 6: Commit**

```bash
git add paper/short/03_math_tools.md
git commit -m "feat(short): 03_math_tools — fold §4 + §9.1 into tool+application once"
```

---

## Task 5a: §4 Findings — F1 (Behavioral baselines)

**Files:**
- Create: `paper/short/04_findings.md` (this task writes the §4 header + §4.1 only; later tasks 5b/5c/5d/5e append)
- Source: `paper/sections/05_over_refusal.md` (2,649 words, six subsections)
- Target for §4.1: ~900w

**What to keep:**
- Headline refusal-rate numbers (E4B refuses 42/42 should_refuse, 1/50 emergency_medical; +emergency-responder-framing nudges to 2/50; E2B refuses 6/50 emergency_medical and 18/42 gray_zone). Currently in §5.2 (lines 22–42).
- The framing-pivot narrative (paper rerouted from over-refusal mitigation to geometry). Currently in §5.6 (lines 68–76).
- One compact summary table replacing per-category walkthrough.

**What to drop:**
- §5.1 "Benchmark structure" (lines 5–20). The benchmark file `data/benchmark_prompts.json` is cited; methodology detail is appendix material.
- §5.3 "Headline negative finding: self-abliteration leaves should_refuse" (lines 44–52) — moves into §4.2 (Task 5b), not §4.1.
- §5.4 "Over-refusal axis" (lines 54–62) — fold the one load-bearing number into the summary table.
- §5.5 "Phrasing-sensitivity probe" (lines 64–66) — drop entirely; not load-bearing for the four findings.

- [ ] **Step 1: Read the source**

Use `Read` on `paper/sections/05_over_refusal.md` (86 lines).

- [ ] **Step 2: Draft the section header and §4.1**

Write `paper/short/04_findings.md`:
```markdown
## 4. Empirical Findings

This section reports the four findings F1–F4 in the same order as the abstract: behavioral framing → activation-space mechanism → weight-space comparison → causal isolation. Each subsection cites the tools introduced in §3.

### 4.1 F1 — Behavioral baselines and the over-refusal pivot
<~900w prose>

<one summary table — rows: should_refuse, emergency_medical, gray_zone; columns: E2B, E4B, E4B+emergency-responder-framing>

<close with one paragraph on the framing pivot>
```

The summary table should look like (approximate numbers, fill from source):

```markdown
| Category (n) | E2B | E4B | E4B + responder framing |
|---|---:|---:|---:|
| should_refuse (42) | 42 / 42 (100%) | 42 / 42 (100%) | 42 / 42 (100%) |
| emergency_medical (50) | 6 / 50 (12%) | 1 / 50 (2%) | 2 / 50 (4%) |
| gray_zone (42) | 18 / 42 (43%) | (fill from source) | — |
```

Verify each cell against `paper/sections/05_over_refusal.md` before committing.

- [ ] **Step 3: Word-count check on §4.1**

Run: `wc -w paper/short/04_findings.md`
Expected: 800–1,000 (only §4.1 is written so far, plus the section header).

- [ ] **Step 4: Numerics audit**

Run: `grep -oE "42 / 42|1 / 50|6 / 50|18 / 42|2 / 50" paper/short/04_findings.md | sort | uniq -c`
Expected: each number appears at least once.

- [ ] **Step 5: Commit**

```bash
git add paper/short/04_findings.md
git commit -m "feat(short): 04_findings §4.1 — F1 over-refusal pivot, summary-table format"
```

---

## Task 5b: §4.2 F2 — Activation-space rank-1 + standard-recipe negative

**Files:**
- Modify: `paper/short/04_findings.md` (append §4.2)
- Sources:
  - `paper/sections/06_mechanistic.md` (2,101 words, six subsections)
  - `paper/sections/07_abliteration_weight_diff.md` §7.1–7.3 (lines 5–35) for the standard-recipe negative
  - `paper/sections/05_over_refusal.md` §5.3 (lines 44–52) for the headline-negative restatement
- Target for §4.2: ~1,200w

**What to keep:**
- Activation-hooking method gloss (one paragraph, not the methodology walkthrough). From `06_mechanistic.md` §1 (lines 7–22).
- Per-layer signal-strength scan, the L15 peak, Cohen's *d* = 2.87. From `06_mechanistic.md` §2 (lines 24–36).
- The rank-1 PCA result: top PC captures 86.6% of squared-norm difference. From `06_mechanistic.md` §3 (lines 38–46).
- The standard-recipe self-abliteration negative result: still 100% should_refuse refusal across α-sweep [0, 2.0] and 7-subset layer partition, statistically indistinguishable from random-direction control. From `07_abliteration_weight_diff.md` §7.2 (lines 15–29) and `05_over_refusal.md` §5.3.
- Embed two figures: `signal_vs_layer.png` and `pca_variance_per_layer.png`.

**What to drop:**
- `06_mechanistic.md` §4 "Category geometry" (lines 48–59) — its load-bearing numbers (+0.93, −0.015) already live in §3.3.
- `06_mechanistic.md` §5 "Anisotropy of L15 activation cloud" (lines 61–69) — its load-bearing fact (mean pairwise cosine ≈ 0.96) already lives in §3.3.
- `06_mechanistic.md` §6 "Summary" (lines 71–80) — section ends with the negative finding, no separate summary needed.
- `07_abliteration_weight_diff.md` §7.1 "The intervention" detailed code-level walk (lines 5–13) — §3.1 has the formula.

- [ ] **Step 1: Read the sources**

Use `Read` on `paper/sections/06_mechanistic.md` (80 lines) and `paper/sections/07_abliteration_weight_diff.md` lines 1–40.

- [ ] **Step 2: Append §4.2 to `paper/short/04_findings.md`**

```markdown
### 4.2 F2 — Refusal is rank-1 in activations but doesn't budge under the standard recipe
<~1,200w covering: activation extraction (one paragraph), per-layer signal scan with L15 peak and Cohen's d = 2.87, rank-1 PCA with 86.6% top-PC variance, then the negative behavioral result — standard rank-1 mean-difference abliteration leaves should_refuse at 100% across the α-sweep and the 7-subset layer partition, indistinguishable from random-direction control>

![Per-layer signal strength, refuse vs comply](../../results/figures/signal_vs_layer.png)

![PCA variance per layer](../../results/figures/pca_variance_per_layer.png)
```

- [ ] **Step 3: Word-count delta check**

Run: `wc -w paper/short/04_findings.md`
Expected: ~2,000–2,200 (previous ~900 + new ~1,200).

- [ ] **Step 4: Numerics audit**

Run: `grep -oE "L15|Cohen|2\\.87|86\\.6|α-sweep|7-subset|random-direction" paper/short/04_findings.md | sort | uniq -c`
Expected: each term present.

- [ ] **Step 5: Commit**

```bash
git add paper/short/04_findings.md
git commit -m "feat(short): 04_findings §4.2 — F2 activation-space rank-1 + standard-recipe null"
```

---

## Task 5c: §4.3 F4 — Two abliterations land in orthogonal subspaces

**Files:**
- Modify: `paper/short/04_findings.md` (append §4.3)
- Source: `paper/sections/07_abliteration_weight_diff.md` §7.4–7.6 (lines 37–80)
- Target for §4.3: ~1,200w

**What to keep:**
- OBLITERATUS median `rank_95` = 6 across ~201 modified tensors. From §7.4 (lines 37–59).
- TrevorJS pure rank-1 across 84 matrices. From §7.4.
- Cross-method top-1 left-singular-vector cosine = −0.08. From §7.4.
- Cross-reference to activation-space refusal direction: |cosine| ≈ 0.04 for both methods. From §7.5 (lines 61–69).
- One paragraph on Gemma 4 architectural quirks that bear on the diff (already gestured at in §2.5). From §7.6 (lines 71–79), compressed to ~150w.
- The three-bullet finding statement at the close.
- Embed figures: `cross_method_singular_vectors.png`, `refusal_direction_vs_singular_vector.png`, `obliteratus_modification_ranks.png`.

**What to drop:**
- Per-tensor SVD walkthrough.
- Long discussion of how OBLITERATUS and TrevorJS each developed their methods (§2.3 in the new background carries the high-level intro).
- §7.7 "Summary" (lines 81–89) — close inline.

- [ ] **Step 1: Read the source**

Use `Read` on `paper/sections/07_abliteration_weight_diff.md` lines 35–89.

- [ ] **Step 2: Append §4.3 to `paper/short/04_findings.md`**

```markdown
### 4.3 F4 — Two published abliterations land in orthogonal subspaces
<~1,200w covering: OBLITERATUS rank profile (median rank_95 = 6 across ~201 modified tensors); TrevorJS rank profile (pure rank-1 across 84 matrices, norm-preserving biprojection); cross-method top-1 singular-vector cosine = −0.08, near-orthogonal in the §3.3 sense; cross-reference to §4.2 refusal direction: |cosine| ≈ 0.04 for both methods, i.e., neither weight-edit direction aligns with activation-space refusal; one paragraph on Gemma 4 architectural quirks (local/global attention, shared K/V layers 24–41) that constrain the diff>

![Cross-method singular vectors](../../results/figures/cross_method_singular_vectors.png)
![Refusal direction vs singular vector](../../results/figures/refusal_direction_vs_singular_vector.png)
![OBLITERATUS modification ranks](../../results/figures/obliteratus_modification_ranks.png)

Three bullets:
- The two methods modify weights in nearly orthogonal directions.
- Neither aligns with the activation-space refusal direction of §4.2.
- One uses multi-rank descent (rank_95 = 6), the other pure rank-1; both are effective in the wild.
```

- [ ] **Step 3: Word-count delta check**

Run: `wc -w paper/short/04_findings.md`
Expected: ~3,200–3,400.

- [ ] **Step 4: Numerics audit**

Run: `grep -oE "rank_95 = 6|~201|84|−0\\.08|0\\.04|biprojection" paper/short/04_findings.md | sort | uniq -c`
Expected: each present.

- [ ] **Step 5: Commit**

```bash
git add paper/short/04_findings.md
git commit -m "feat(short): 04_findings §4.3 — F4 cross-method weight-diff orthogonality"
```

---

## Task 5d: §4.4 F3 — Causal-isolation cascade with H4 partial

**Files:**
- Modify: `paper/short/04_findings.md` (append §4.4)
- Source: `paper/sections/08_rank1_cascade.md` (2,334 words, four subsections)
- Target for §4.4: ~1,500w

This is the experimental contribution and gets the smallest relative cut. H4 keeps full treatment; H1, H2, H3, H5 collapse to one combined paragraph.

**What to keep, full treatment:**
- §1 "Setup and the negative-finding starting point" (lines 5–9). Compress to ~150w.
- §2 "Hypothesis taxonomy" (lines 11–23). Keep the H1–H5 taxonomy as a table or compact bullet list (~200w).
- §3 "Cascade results" H4 portion (find the H4 paragraph inside lines 25–44). Full ~600w treatment with all numerics: D3 variant (chat-template direction + winsorize + Gram-Schmidt against harmless mean) → 17/42 = 40.5% should_refuse refusal at α = 1.0; residual concentrates on extreme topics (CSAM, ICS/hospital malware, weapons).
- §4 "Implications" (lines 46–54). Compress to ~250w. Frame as "partial first-order behavioral response under small rank-1 perturbation; residual ≈ 60% indicates refusal is not fully one-dimensional on Gemma 4."

**What to compress, hard:**
- §3 H1/H2/H3/H5 paragraphs (everything in lines 25–44 except H4) → one combined "What was ruled out" paragraph at ~150w total. One sentence each:
  - H1 (bnb int8 in-place edit): bf16 edit path reproduces the int8 failure character-for-character, ruling out quantization.
  - H2 (chat-template direction alone): insufficient on its own to move refusal off 100%.
  - H3 (winsorization alone): insufficient on its own.
  - H5 (norm-preserving biprojection): refuted — vanilla projection changes per-row L2 norms by mean 0.038%, below the threshold at which RMSNorm sensitivity would plausibly drift.

- [ ] **Step 1: Read the source**

Use `Read` on `paper/sections/08_rank1_cascade.md` (54 lines).

- [ ] **Step 2: Append §4.4 to `paper/short/04_findings.md`**

```markdown
### 4.4 F3 — A causal-isolation cascade isolates Gram-Schmidt as the load-bearing ingredient
<~150w setup recapping §4.2's null and the question it raised>

#### Hypothesis taxonomy
<~200w compact table or bullet list listing H1 (bnb int8), H2 (chat-template direction), H3 (winsorization), H4 (Gram-Schmidt against harmless mean), H5 (norm-preserving biprojection), with one-clause descriptions>

#### What was ruled out
<~150w combined paragraph — one sentence each for H1, H2, H3, H5 with what was tested and what it showed>

#### H4 — Gram-Schmidt against the harmless mean
<~600w full treatment: the D3 variant; 17/42 = 40.5% should_refuse refusal at α = 1.0 across all 42 layers; residual concentrates on extreme topics (CSAM, ICS/hospital malware, weapons); per-prompt breakdown if it fits in word budget; embed m6_cascade_gate.png and m6_perprompt_n42.png>

![Cascade gate](../../results/figures/m6_cascade_gate.png)
![Per-prompt n=42](../../results/figures/m6_perprompt_n42.png)

#### Implications
<~250w — "partial first-order behavioral response under small rank-1 perturbation, consistent with §3.2's Mirsky-bound prediction. Residual ≈ 60% indicates the refusal manifold is not fully one-dimensional in weight space on Gemma 4, even when it is in activation space (§4.2)">
```

- [ ] **Step 3: Word-count delta check**

Run: `wc -w paper/short/04_findings.md`
Expected: ~4,700–4,900.

- [ ] **Step 4: Numerics audit on §4.4 specifically**

Run: `grep -oE "17 / 42|40\\.5|Gram-Schmidt|D3|CSAM|H1|H2|H3|H4|H5|biprojection|0\\.038" paper/short/04_findings.md | sort | uniq -c`
Expected: each present.

- [ ] **Step 5: Commit**

```bash
git add paper/short/04_findings.md
git commit -m "feat(short): 04_findings §4.4 — F3 cascade, compress rejected H1/H2/H3/H5"
```

---

## Task 5e: §4.5 — Connecting paragraph

**Files:**
- Modify: `paper/short/04_findings.md` (append §4.5)
- Source: condensed synthesis of §4.1–§4.4 (no new source material)
- Target: ~200w

A short paragraph at the close of §4 that says what F1–F4 collectively show: refusal in Gemma 4 E4B-it is geometrically clean in activations (F2) but multi-rank in weights (F3, F4); the over-refusal motivation that started the project (F1) turned out smaller than expected. This replaces the four separate section-intro paragraphs that glued §5/§6/§7/§8 together in the original draft.

- [ ] **Step 1: Append §4.5 to `paper/short/04_findings.md`**

```markdown
### 4.5 What the four findings say together
<~200w synthesis — refusal is rank-1 in activations (F2) but multi-rank in weights (F3 cascade + F4 cross-method); the over-refusal motivation that opened the project (F1) turned out smaller than the smaller-model baseline suggested; the question the paper now answers is "what does and does not transfer from the activation-space rank-1 picture to weight-space removal">
```

- [ ] **Step 2: Word-count final check**

Run: `wc -w paper/short/04_findings.md`
Expected: 4,800–5,200.
If over 5,500: surface to operator; consider another pass on §4.1 or §4.3.

- [ ] **Step 3: Commit**

```bash
git add paper/short/04_findings.md
git commit -m "feat(short): 04_findings §4.5 — F1-F4 connecting synthesis"
```

---

## Task 6: §5 Discussion & Conclusion — merge

**Files:**
- Create: `paper/short/05_discussion_conclusion.md`
- Sources:
  - `paper/sections/09_discussion.md` §9.2–9.5 (lines 53–82, ~1,400 words)
  - `paper/sections/10_conclusion.md` (1,873 words, four subsections)
- Target: ~1,800 words

**What to keep:**
- Limitations from §9.3 (lines 61–73) — compress to ~450w. Cover the residual 60%, n = 42 sample size, Gemma-4-specific caveats.
- Ethics from §10.3 (lines 23–31) — compress to ~500w. Dual-use surface, constructive selective-safety, what the residual 60% means for misuse risk.
- Conclusion summary from §10.1 (lines 5–17) — compress to ~500w.
- Future-work pointer from §9.4 (lines 75–77) — compress to ~150w.
- Over-parametrization framing from §9.2 (lines 53–59) — compress to ~100w, fold into Limitations.

**What to drop:**
- §9.1 already moved to §3 — must not reappear here.
- §9.5 "Closing" (lines 79–82) — duplicates conclusion intro.
- §10.2 "What we did not expect" (lines 19–21) — narrative duplicate of §4.5's synthesis paragraph; drop.
- §10.4 "Closing" (lines 33–37) — drop.

- [ ] **Step 1: Read both sources**

Use `Read` on `paper/sections/09_discussion.md` lines 53–96 and `paper/sections/10_conclusion.md` (35 lines).

- [ ] **Step 2: Draft `paper/short/05_discussion_conclusion.md`**

```markdown
## 5. Discussion and Conclusion

### 5.1 Limitations
<~550w including the over-parametrization paragraph at the close — residual 60% concentrating on extreme topics; n = 42 small-sample caveat; Gemma 4 specifics; one paragraph on the over-parametrization framing>

### 5.2 Future work
<~150w — pointers for follow-up: rank-2/rank-3 weight edits, multi-direction selective safety, larger n>

### 5.3 Ethics
<~500w — dual-use surface of abliteration research; constructive selective-safety angle; what the residual 60% means for misuse risk (it does NOT make weights safe-against-abliteration; the methods that work in the wild still work)>

### 5.4 Conclusion
<~600w — recap the four findings; the central empirical claim (rank-1 in activations, multi-rank in weights on Gemma 4); the methodological contribution (cascade structure); the framing pivot (over-refusal smaller than expected on E4B)>
```

- [ ] **Step 3: Word-count check**

Run: `wc -w paper/short/05_discussion_conclusion.md`
Expected: 1,600–2,000.

- [ ] **Step 4: Verify no §9.1 / lecture-connection content leaked back in**

Run: `grep -E "Connection [1-5]|Lec [5-9]|lecture connection" paper/short/05_discussion_conclusion.md`
Expected: NO matches. If any, those moved to §3 — remove from §5.

- [ ] **Step 5: Commit**

```bash
git add paper/short/05_discussion_conclusion.md
git commit -m "feat(short): 05_discussion_conclusion — merge §9.2-9.5 + §10, drop §9.1 leakage"
```

---

## Task 7: References — copy and prune orphans

**Files:**
- Create: `paper/short/references.md` (copied from `paper/references.md`)
- Source: `paper/references.md` (~60 lines)
- Target: same as source minus any orphans

- [ ] **Step 1: Copy verbatim**

Run: `cp paper/references.md paper/short/references.md`
Expected: silent success.

- [ ] **Step 2: Build the list of citation keys actually used in the new paper**

Run:
```bash
grep -hoE "\\[[A-Za-z][A-Za-z-]+ [0-9]{4}[a-z]?\\]" paper/short/0[0-5]*.md | sort -u > /tmp/cited_keys.txt
wc -l /tmp/cited_keys.txt
cat /tmp/cited_keys.txt
```
Expected: a sorted list of `[Author Year]` keys, ~15–25 entries.

- [ ] **Step 3: Build the list of citation keys defined in references.md**

Run:
```bash
grep -oE "\\*\\*\\[[A-Za-z][A-Za-z-]+ [0-9]{4}[a-z]?\\]\\*\\*" paper/short/references.md | sed 's/\\*\\*//g' | sort -u > /tmp/defined_keys.txt
wc -l /tmp/defined_keys.txt
```
Expected: a longer list than cited (orphans exist).

- [ ] **Step 4: Identify orphans**

Run: `comm -23 /tmp/defined_keys.txt /tmp/cited_keys.txt > /tmp/orphan_keys.txt && cat /tmp/orphan_keys.txt`
Expected: 0–10 keys that are defined but not cited.

- [ ] **Step 5: Remove orphan entries from `paper/short/references.md`**

For each key in `/tmp/orphan_keys.txt`, find the matching `- **[Key]** ...` line in `paper/short/references.md` and delete the bullet (and any continuation lines). Use `Edit` with the full bullet as `old_string`. Keep the alphabetical order.

If `/tmp/orphan_keys.txt` is empty, skip this step.

- [ ] **Step 6: Verify no cited key is missing from references.md**

Run: `comm -23 /tmp/cited_keys.txt /tmp/defined_keys.txt > /tmp/missing_keys.txt && cat /tmp/missing_keys.txt`
Expected: empty file. If non-empty, you cited something not in references — either add the entry by copying from the original `paper/references.md` (which you may have over-pruned), or remove the citation from the paper.

- [ ] **Step 7: Commit**

```bash
git add paper/short/references.md
git commit -m "feat(short): references — copy from canonical and prune orphans"
```

---

## Task 8: Assemble `paper_short.md`

**Files:**
- Create: `paper/short/paper_short.md`
- Sources: all five section files + `00_abstract.md` + `references.md` in `paper/short/`

The assembled file mirrors the existing `paper/paper.md` layout: title, abstract, TOC, sections, references.

- [ ] **Step 1: Write `paper/short/paper_short.md` by concatenation**

Build the file with this structure:

```markdown
# Geometry of Alignment

**Where Safety Behavior Lives in Gemma 4 E4B-it, and What a Rank-1 Weight Edit Can and Cannot Remove**

*EECS 6699 (Mathematics of Deep Learning), Columbia University, Spring 2026*

*Date: 2026-05-12*

<paste contents of paper/short/00_abstract.md, starting from the `## Abstract` heading>

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Background and Related Work](#2-background-and-related-work)
- [3. Mathematical Tools](#3-mathematical-tools)
- [4. Empirical Findings](#4-empirical-findings)
- [5. Discussion and Conclusion](#5-discussion-and-conclusion)
- [References](#references)

<paste contents of paper/short/01_introduction.md>

<paste contents of paper/short/02_background_related.md>

<paste contents of paper/short/03_math_tools.md>

<paste contents of paper/short/04_findings.md>

<paste contents of paper/short/05_discussion_conclusion.md>

<paste contents of paper/short/references.md, starting from the `# References` heading replaced with `## References`>
```

Practical approach: use `cat` to concatenate, then edit the title/TOC block by hand.

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
  # references.md uses "# References" as top-level; demote to "## References"
  sed 's/^# References$/## References/' paper/short/references.md
} > paper/short/paper_short.md
```

- [ ] **Step 2: Sanity-check the assembled file**

Run: `head -20 paper/short/paper_short.md && echo "---" && grep -n "^## " paper/short/paper_short.md`
Expected: title + abstract + TOC at the top; `## 1. Introduction`, `## 2. Background…`, … `## References` appearing in order.

- [ ] **Step 3: Word-count check on the full assembly**

Run: `wc -w paper/short/paper_short.md`
Expected: 10,500–13,500 (acceptance band per spec §8).
If outside the band: surface to operator before continuing to Task 9.

- [ ] **Step 4: Commit**

```bash
git add paper/short/paper_short.md
git commit -m "feat(short): paper_short.md — assemble final cut version"
```

---

## Task 9: Acceptance audit

**Files:** read-only audit across `paper/short/` plus a comparison against `paper/sections/` to confirm originals untouched.

- [ ] **Step 1: Confirm originals untouched**

Run: `git diff agent/writeup..feat/paper-short -- paper/sections/ paper/paper.md paper/references.md | head -5`
Expected: empty output. If non-empty, you accidentally modified an original file — revert with `git checkout agent/writeup -- paper/sections/ paper/paper.md paper/references.md` and re-commit.

- [ ] **Step 2: Confirm all expected files exist**

Run: `ls paper/short/`
Expected: `00_abstract.md  01_introduction.md  02_background_related.md  03_math_tools.md  04_findings.md  05_discussion_conclusion.md  references.md  paper_short.md  .gitkeep`

- [ ] **Step 3: Headline numerics audit on the assembled file**

Run:
```bash
for pattern in "42 / 42" "1 / 50" "Cohen's \\*d\\* = 2\\.87" "86\\.6" "17 / 42" "40\\.5" "rank_95 = 6" "~201" "−0\\.08" "0\\.04"; do
  count=$(grep -cE "$pattern" paper/short/paper_short.md)
  echo "$pattern : $count"
done
```
Expected: every count ≥ 1. If any is 0, the cut lost a load-bearing number — restore from the original section file.

- [ ] **Step 4: Figure-path audit**

Run:
```bash
grep -oE "results/figures/[a-z0-9_.-]+\\.(png|md)" paper/short/paper_short.md | sort -u | while read f; do
  if [ ! -f "$f" ]; then echo "MISSING: $f"; fi
done
```
Expected: no `MISSING:` lines. Each figure path must resolve to an existing file in `results/figures/`.

- [ ] **Step 5: Section-level word counts**

Run: `wc -w paper/short/0[0-5]*.md`
Expected, roughly:
- `00_abstract.md`: ~390
- `01_introduction.md`: ~1,100
- `02_background_related.md`: ~1,800
- `03_math_tools.md`: ~1,800
- `04_findings.md`: ~5,000
- `05_discussion_conclusion.md`: ~1,800
- `paper_short.md`: ~11,500–13,500 (assembled total; abstract + sections + references prose contribute)

If any single section is more than 30% off its target and reads bloated, do one tightening pass. If it reads cleanly, accept.

- [ ] **Step 6: Orphan-reference audit**

Run:
```bash
grep -hoE "\\[[A-Za-z][A-Za-z-]+ [0-9]{4}[a-z]?\\]" paper/short/0[1-5]*.md | sort -u > /tmp/cited.txt
grep -oE "\\*\\*\\[[A-Za-z][A-Za-z-]+ [0-9]{4}[a-z]?\\]\\*\\*" paper/short/references.md | sed 's/\\*\\*//g' | sort -u > /tmp/defined.txt
comm -23 /tmp/defined.txt /tmp/cited.txt > /tmp/orphans.txt
comm -23 /tmp/cited.txt /tmp/defined.txt > /tmp/missing.txt
echo "Orphans: $(wc -l < /tmp/orphans.txt)"
echo "Missing: $(wc -l < /tmp/missing.txt)"
cat /tmp/missing.txt
```
Expected: Orphans = 0, Missing = 0. If Missing > 0, those keys must be added to `references.md` (copy from the original `paper/references.md`).

- [ ] **Step 7: Final summary commit (if any clean-up was needed)**

If audit found and fixed issues:
```bash
git add paper/short/
git commit -m "fix(short): acceptance-audit clean-up (figure paths / orphans / numerics)"
```
If everything was clean on first pass, skip this commit.

- [ ] **Step 8: Push the branch and surface to operator**

```bash
git push -u origin feat/paper-short
```
Then report to the operator:
- Total assembled word count
- Per-section word counts
- Branch name and push status
- Any judgment calls made during cuts that warrant operator review

---

## Self-Review of this plan against the spec

Spec coverage check (each spec requirement → plan task):

- **Spec §4 output layout** (new `paper/short/` folder, 6 section files, `references.md`, `paper_short.md`) → Tasks 0, 1–6, 7, 8.
- **Spec §5 new section structure table** → Tasks 1 through 6 each target one row.
- **Spec §6.§1 Introduction cuts** (drop preview/math-framing/roadmap, tighten remaining) → Task 2.
- **Spec §6.§2 Background + Related merge** (keep alignment overview, linear-rep, abliteration lineage, over-refusal benchmarks, Gemma 4 quirks, OBLITERATUS+TrevorJS intro; drop RLHF tutorial detail) → Task 3.
- **Spec §6.§3 Math Tools four subsections + tool-then-application + drop §4.3/§4.5/§4.7/Connection-3/Connection-5** → Task 4 step 2 explicitly enumerates each.
- **Spec §6.§4 Findings four-finding structure + per-finding cuts** (F1 −66%, F2 −43%, F4 −51%, F3 −36%; H1/H2/H3/H5 compressed to one paragraph; H4 full) → Tasks 5a, 5b, 5c, 5d, 5e.
- **Spec §6.§5 Discussion + Conclusion merge** (keep limitations + ethics + conclusion + future-work; fold §9.2 into limitations; drop §9.5 + §10.2 + §10.4) → Task 6.
- **Spec §6 Abstract unchanged** → Task 1.
- **Spec §6 References copy + prune orphans** → Tasks 7 and 9 Step 6.
- **Spec §7 build step** (assemble paper_short.md) → Task 8.
- **Spec §8 acceptance criteria** (folder exists, word count 10.5K–13.5K, headline numerics intact, originals unmodified, figures resolve, no orphan references) → Task 9 covers each item explicitly.

Placeholder scan: no `TBD`, `TODO`, "fill in later", or "similar to Task N" in the plan. Code blocks contain executable commands; prose drafts in section tasks contain concrete `<~Nw — what to write>` directives rather than empty placeholders, and the implementer writes the actual prose under those directives based on the precisely-cited source line ranges.

Type consistency: `paper/short/` path is consistent across every task. Branch `feat/paper-short` referenced consistently. Section file naming `0N_<topic>.md` consistent across Tasks 1–6 and Task 9.

Plan is ready for execution.

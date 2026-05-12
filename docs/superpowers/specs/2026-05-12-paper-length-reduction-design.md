# Paper Length Reduction — Design Spec

**Date:** 2026-05-12
**Target paper:** `paper/sections/01_introduction.md` … `paper/sections/10_conclusion.md` (assembled as `paper/paper.md`) on branch `agent/writeup` in worktree `gb-paper/`.
**Driver:** professor feedback that the current ~25.5K-word paper is too long; aim for roughly half.

## 1. Problem

The current paper is 25,524 words across ten section files. The professor's feedback is that it should be roughly halved. Two specific hints from the operator:

- The math section can be reduced; the paper does not need a comprehensive guide on how the math works.
- Failed experimental attempts can be compressed.

A scan of the existing draft surfaces two structural redundancies on top of those hints:

- **§4 (Mathematical Framework) and §9.1 (Five Lecture Connections) walk through the same Lec 5/6/7/8/9 material twice** — once as setup, once as application to findings. Roughly 5,800 combined words on tools that could be introduced and applied once.
- **§8's cascade walks five hypotheses H1–H5 at near-equal depth, but only H4 turned out load-bearing.** H1, H2, H3, H5 occupy roughly half the section while contributing rejected-hypothesis content the operator has flagged as compressible.

## 2. Goal

Produce a roughly half-length version of the paper in a new folder, preserving the original as a side-by-side reference. Cuts are driven by (a) removing the §4↔§9.1 redundancy, (b) compressing rejected cascade hypotheses, (c) collapsing pedagogical math walkthroughs into "tool + application" presentations, and (d) trimming tutorial background that the professor (the primary reader) does not need.

The word budgets below are *guidance, not hard limits*. The implementing agent should use judgment — if a section reads cleanly at a slightly larger or smaller size, that wins over hitting an exact number. The global "roughly half" target is the actual constraint.

## 3. Non-goals

- **Not** changing any empirical claim, number, or finding. All quantitative results (Cohen's *d* = 2.87, 86.6% top-PC variance, 17/42 = 40.5%, median `rank_95` = 6, cross-method cosine −0.08, etc.) are preserved verbatim.
- **Not** regenerating figures. All figure references continue to point at `results/figures/` by the same relative paths.
- **Not** editing the existing `paper/paper.md` or `paper/sections/*.md` files. The original draft stays intact as a reference.
- **Not** changing the abstract beyond mechanical fixes if cuts orphan a citation or number.
- **Not** re-running any experiments. This is a pure prose / structural pass over an existing draft.

## 4. Output layout

A new folder `paper/short/` sibling to `paper/sections/`:

```
paper/short/
  00_abstract.md            # copied from paper/sections (currently lives at top of paper.md), light touch-up only
  01_introduction.md
  02_background_related.md
  03_math_tools.md
  04_findings.md
  05_discussion_conclusion.md
  references.md             # copied from paper/references.md, orphans pruned after text settles
  paper_short.md            # assembled: title + abstract + TOC + 5 sections + references
```

The current `paper/paper.md` carries the abstract inline at the top rather than in a separate section file. The new folder splits it into `00_abstract.md` so the assembled `paper_short.md` build is uniform across pieces.

Reason for "new folder" rather than overwriting in place: operator explicitly asked to keep the long version as a reference. A new folder gives a clean side-by-side diff and zero risk of losing the original draft if the cut version overshoots.

Work happens on a feature branch off `agent/writeup` so the cut can be reviewed via `git diff` before any merge.

## 5. New section structure

Six top-level sections (down from ten), in F1→F2→F4→F3 narrative order matching the abstract:

| New § | Merges from | Target | Old | Cut |
|---|---|---:|---:|---:|
| 1. Introduction | old §1 | ~1,100 | 2,107 | −48% |
| 2. Background & Related Work | old §2 + §3 | ~1,800 | 3,432 | −48% |
| 3. Mathematical Tools | old §4 + §9.1 | ~1,800 | ~5,800 | −69% |
| 4. Empirical Findings | old §5 + §6 + §7 + §8 | ~5,000 | 9,549 | −48% |
| 5. Discussion & Conclusion | old §9.2–9.5 + §10 | ~1,800 | ~3,700 | −51% |
| References | unchanged (orphans pruned) | — | — | — |
| **Sections subtotal** | | **~11,500** | **24,548** | **−53%** |
| Abstract (unchanged) | abstract | ~390 | ~390 | 0 |
| **Assembled paper_short.md** | | **~11,900** | **24,938** | **−52%** |

**All word counts are budgets, not hard ceilings.** If a section needs 200 more words to read cleanly, that is the right call. If a section comes in 400 words under budget without losing content, that is also fine. The agent picks the cut depth per paragraph based on what reads best. Per-section subtotals in §6 may not sum exactly to the §6 budget — the difference is slack for connective sentences and subsection transitions inside that section.

## 6. Per-section design

### §1 Introduction (~1,100w from 2,107)

**Keep:** the hiking-emergency motivation (one tightened paragraph), the research question framing with the Arditi 2024 anchor, the five-item contributions list condensed.

**Drop:** the "Preview of empirical findings" subsection (abstract already lists F1–F4), the "Roadmap" subsection (TOC handles it), the standalone "Mathematical framing" subsection (new §3 introduces math directly).

**Reason for drops:** abstract + TOC structurally duplicate two of the six current intro subsections; removing them is a clean cut with zero information loss. The math-framing paragraph was a forward pointer to old §4 — with the new §3 living right after §2, the forward pointer is no longer earning its words.

### §2 Background & Related Work (~1,800w from 3,432)

**Keep:** alignment-training overview compressed to one paragraph (RLHF → DPO → Constitutional AI in a tight sweep), linear-representation hypothesis as the justification for mean-difference direction estimation, abliteration lineage centered on Arditi 2024 and Mlabonne 2024, the over-refusal benchmarks Röttger/XSTest and Cui/OR-Bench, Gemma 4 architectural quirks that bear on §4.3 (local/global attention split, RMSNorm placement, shared K/V tensors in layers 24–41), brief intro of OBLITERATUS and TrevorJS as the two comparison points used in §4.

**Drop:** detailed RLHF tutorial, representation-engineering line beyond the linear-rep hypothesis, history of abliteration variants not directly cited in §4 (p-e-w, grimjim background where not load-bearing), DPO/Constitutional AI implementation detail.

**Reason:** keep only citations and constructs the empirical sections actively reference. The current §2 + §3 read as a literature survey; the professor does not need that — they need the minimum context to follow §3 and §4.

### §3 Mathematical Tools (~1,800w from ~5,800) — biggest single cut

The cleanest "tool, then immediate application" presentation. Three tools, each stated once and applied to a specific finding.

**§3.1 Setup (~250w).** The rank-1 weight edit `W ← W − α·d·dᵀW` in coordinates, applied to `o_proj` and `down_proj`. Define α, d, the Frobenius norm of the update.

**§3.2 Mirsky's bound and the partial-response prediction (~600w).** State Mirsky 1960 for rectangular matrices, specialize to rank-1: `‖E‖_F = |α|·‖dᵀW‖_2`. One short paragraph on why this is the right tool instead of Hoffman-Wielandt (rectangular vs. square eigenvalue). Apply immediately: per-layer `‖E‖_F / ‖W‖_F` is ~0.2–0.8% range; small relative perturbation predicts a first-order linearization regime and a partial behavioral response.
*Fuses old §4.2 with old §9.1 Connection 2.*

**§3.3 Near-orthogonality with the anisotropy caveat (~500w).** State the high-dimensional concentration fact (random unit vectors near-orthogonal in 2560-d). State the caveat in one sentence: this applies to *learned conceptual directions*, not raw residual activations (whose mean pairwise cosine at L15 is ≈0.96, far from the Gaussian null). Apply immediately: over-refuse-category directions cluster at +0.93 mutual cosine and −0.015 cosine against the should-refuse direction — orthogonal in the sense the theorem predicts.
*Fuses old §4.4 with old §9.1 Connection 1. The caveat is the load-bearing distinction the paper relies on; it stays.*

**§3.4 Lazy training / NTK as plausibility, not prediction (~300w).** One paragraph: under small ‖E‖, behavior is approximately linear in the perturbation. Soft motivation for "we expected a partial response", not quantitative theory.
*Fuses old §4.6 with old §9.1 Connection 4, compressed hard.*

**Dropped from old §4:**
- §4.3 (Lec 7 SVD machinery + spectral-Frobenius coincidence) — *Reason: pure lecture-walkthrough; the only load-bearing fact (`‖E‖_F = ‖E‖_2` for rank-1) fits in one line of §3.2.*
- §4.5 (per-layer Mirsky-bound heatmap subsection) — *Reason: the actual number (0.2–0.8%) is what matters and now lives in §3.2 in one sentence; the heatmap stays as a figure reference without surrounding prose.*
- §4.7 (forward pointer "how §5–§9 cite back to §4") — *Reason: TOC + abstract handle navigation.*

**Dropped from old §9.1:**
- Connection 3 (Lec 7 SVD/Frobenius/spectral reuse) — *Reason: same material as the dropped §4.3.*
- Connection 5 (Lec 5 follow-up: thin-shell + cosine) — *Reason: duplicates Connection 1's near-orthogonality argument.*
- The "lecture connection" framing itself — *Reason: presenting tool + application together once reads cleaner than presenting them twice.*

**Overall reason for §3 design:** the operator explicitly asked for a non-tutorial math section. "Tool + application" once, instead of "tool" then "application of tool", removes the largest single block of redundancy in the paper.

### §4 Empirical Findings (~5,000w from 9,549)

Four sub-findings in the abstract's F1→F2→F4→F3 order: behavioral framing → activation-space mechanism → weight-space comparison → causal isolation.

**§4.1 F1 — Behavioral baselines and the over-refusal pivot (~900w from 2,649, −66%).**
- *Keep:* the headline numbers (E4B refuses 42/42 should_refuse, 1/50 emergency_medical; E2B refuses 6/50 emergency_medical and 18/42 gray_zone), the framing-pivot narrative (project rerouted from over-refusal mitigation to geometry), one compact summary table.
- *Drop:* prompt-construction methodology, classifier validation prose, per-category walkthrough.
- *Reason:* this section is framing, not contribution. The numbers carry it; methodological detail belongs in an appendix reference if anywhere.

**§4.2 F2 — Refusal is rank-1 in activations, doesn't budge under the standard recipe (~1,200w from 2,101, −43%).**
- *Keep:* per-layer signal scan and the peak at L15, Cohen's *d* = 2.87, top-PC captures 86.6% of the squared-norm difference, the standard rank-1 mean-difference abliteration result (still 100% refusal on should_refuse, indistinguishable from random-direction control across α-sweep [0, 2.0] and 7-subset layer partition).
- *Drop:* hook-implementation details, alternative direction estimators tried before settling on mean-difference.
- *Reason:* the activation-space anchor and earns full numeric treatment; implementation detail goes.

**§4.3 F4 — Two published abliterations land in orthogonal subspaces (~1,200w from 2,465, −51%).**
- *Keep:* OBLITERATUS median `rank_95` = 6 across ~201 modified tensors vs. TrevorJS pure rank-1 across 84; cross-method top-1 left-singular-vector cosine = −0.08; cross-reference to the activation-space refusal direction |cosine| ≈ 0.04 for both; the three-bullet finding statement.
- *Drop:* per-tensor SVD walkthrough, weight-diff methodology tutorial, long history of each method's development.
- *Reason:* the contribution is the three numbers; everything else is scaffolding.

**§4.4 F3 — Causal-isolation cascade with partial H4 result (~1,500w from 2,334, −36%).** *Smallest relative cut because this is the paper's experimental contribution.*
- *Keep, full treatment:* setup + negative-finding starting point, hypothesis taxonomy table, the H4 result with all numerics (D3 variant — chat-template direction + winsorize + Gram-Schmidt against harmless mean — yielding 17/42 = 40.5% should_refuse refusal at α = 1.0; residual concentrates on extreme topics: CSAM, ICS/hospital malware, weapons), the "partial first-order behavioral response under small rank-1 perturbation" framing, the implications subsection.
- *Compress hard:* H1 (bnb int8 in-place edit), H2 (chat-template direction alone), H3 (winsorization alone), H5 (norm-preserving biprojection) → one combined "What was ruled out" paragraph (~150w total), one sentence each for what was tested and what it showed.
- *Reason:* operator explicitly flagged failed attempts as compressible. The value of the failures is that they sharpen the H4 attribution; one sentence each carries that. H4 keeps full treatment because it is the headline contribution.

**§4.5 (~200w) — short connecting paragraph.** Replaces the four separate section-intro paragraphs that currently glue §5/§6/§7/§8 together.

### §5 Discussion & Conclusion (~1,800w from ~3,700)

**Keep:** limitations subsection covering the residual 60%, n = 42 sample size, Gemma-4-specific caveats (~450w); ethics discussion of the dual-use surface of abliteration research, the constructive selective-safety angle, and what the residual 60% means for misuse risk (~500w); conclusion summary (~500w); brief future-work pointer (~250w).

**Compress:** over-parametrization framing (§9.2) folded into limitations as one paragraph (~100w).

**Drop:** §9.1 already moved into §3; the §9.5 standalone "Closing" subsection that duplicates the conclusion intro.

**Reason:** limitations + ethics are non-negotiable for an alignment paper at this venue. The over-parametrization paragraph is course-flavored framing that compresses cleanly. §9.5 was a pure transition device whose words can go.

### Abstract (~393w, keep as-is)

Proportional to a 12K-word paper. Light touch-up only if cuts change any of the numbers it cites (they should not — see §3 non-goals).

### References

Copy `paper/references.md` verbatim into `paper/short/references.md`. After the prose cuts settle, do one pass to remove any citations that no longer appear in the new text (likely candidates: tutorial-only citations dropped from §2, math-source citations dropped from §3.3/§3.4).

## 7. Build step

`paper_short.md` is the assembled artifact: title + abstract + TOC + the five section files + references. Mirrors the existing `paper/paper.md` build pattern. The assembly can be a simple `cat`-and-glue script or a manual concatenation; either works.

## 8. Acceptance criteria

- New folder `paper/short/` exists on a feature branch off `agent/writeup`.
- All five new section files plus `paper_short.md` and `references.md` are present.
- Assembled `paper_short.md` total word count (sections + abstract, excluding references) lands in the **10,500–13,500** range (soft target around 11,900; the band reflects "roughly half" being a judgment call rather than a precise number).
- All four findings F1–F4 are present with their headline numerics intact (Cohen's *d* = 2.87, 86.6% top-PC, 17/42 = 40.5%, median `rank_95` = 6, cross-method cosine −0.08, |cosine| ≈ 0.04 vs. activation-space refusal direction).
- No empirical claim or number from the original draft is changed.
- All figure references point to existing files in `results/figures/`.
- The original `paper/paper.md` and `paper/sections/*.md` are unmodified.
- References file contains no orphan entries (entries not cited in the new text).

## 9. Open questions for implementation

None at design time. The writing-plans skill will translate this spec into per-section editing tasks; per-section judgment calls (e.g., which exact sentences to cut in a given paragraph) are left to the implementing agent rather than pre-specified here.

## 10. Reasoning summary — why these particular design choices

1. **Six sections, not ten** — the operator gave explicit permission to merge, and the §4↔§9.1 redundancy was the largest single source of compressible words. Merging §5/§6/§7/§8 into one "Findings" chapter is a smaller but real saving from cross-section connective tissue.
2. **Soft budgets, not hard caps** — operator explicitly requested this. Prose quality wins over hitting a number.
3. **New folder, original untouched** — operator explicitly requested this. Gives a clean reference for diff review.
4. **"Tool + application" math presentation** — direct response to the operator's "don't need a comprehensive guide on how the math works"; presenting Mirsky-then-NTK once with the empirical application attached is shorter and more useful to a reader than presenting the lectures twice.
5. **H1/H2/H3/H5 to one paragraph** — direct response to the operator's "failed attempts can be compressed". The methodological contribution (the cascade structure) survives in one sentence each; only the prose around it is cut.
6. **F1 cut hardest (−66%)** — F1 is framing, not contribution. The numbers are what matter, and they fit in a tight paragraph plus table.
7. **F3 cut least (−36%)** — F3 is the paper's experimental contribution. Cutting it harder would damage the load-bearing result.
8. **Abstract and figures unchanged** — both already proportional to a much shorter paper; abstract is 393 words for a 25K paper which would still be fine for a 12K paper.
9. **References pruned, not rewritten** — cheaper to do one orphan-removal pass at the end than to maintain in parallel.

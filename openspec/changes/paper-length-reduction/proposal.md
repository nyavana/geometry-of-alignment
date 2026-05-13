## Why

The current paper at `paper/paper.md` runs ~25.5K words across ten section files. The EECS 6699 professor's feedback is that this should be roughly halved. Two specific hints: (1) the math section does not need to be a comprehensive tutorial, and (2) failed experimental attempts can be compressed. On top of those, a scan of the draft surfaces two structural redundancies — §4 (Mathematical Framework) and §9.1 (Five Lecture Connections) walk through the same Lec 5/6/7/8/9 material twice (~5.8K combined words), and §8's cascade walks H1–H5 at near-equal depth when only H4 turned out load-bearing.

## What Changes

- Produce a roughly half-length version of `paper/paper.md` in a new sibling folder `paper/short/`. The original draft stays untouched as a side-by-side reference.
- Re-organize ten sections into six top-level sections, in F1→F2→F4→F3 narrative order matching the abstract: Abstract (verbatim), §1 Introduction, §2 Background & Related Work, §3 Mathematical Tools, §4 Empirical Findings, §5 Discussion & Conclusion, References.
- Drop the §4↔§9.1 duplicate math walkthrough by adopting a "tool then immediate application" structure: state each tool once and apply it once to a finding.
- Compress rejected cascade hypotheses (H1, H2, H3, H5) to a single "What was ruled out" paragraph; H4 keeps full treatment.
- Trim tutorial background in §2 (RLHF mechanics, abliteration-variant history) to the minimum that supports §3 and §4.
- Preserve every empirical claim and headline number verbatim (Cohen's *d* = 2.87, 86.6% top-PC, 17/42 = 40.5%, median `rank_95` = 6, cross-method cosine −0.08, |cosine| ≈ 0.04 vs. activation-space refusal direction, etc.).
- Preserve all figure references; figure links resolve as `../../results/figures/<name>.png` from the new `paper/short/` location.
- Copy `paper/references.md` to `paper/short/references.md` and prune orphan entries after prose settles.
- Assemble a final `paper/short/paper_short.md` (title + abstract + TOC + five sections + references) mirroring the existing `paper/paper.md` build pattern.
- Work happens on a feature branch `feat/paper-short` off `agent/writeup` so the cut can be reviewed via `git diff` before any merge.

## Capabilities

### New Capabilities
- `paper-short`: A halved companion version of the paper at `paper/short/`, structured as six top-level sections with a "tool + application" math chapter, compressed rejected hypotheses, and verbatim-preserved numerics. Includes per-section word budgets, a global word-count band, headline-numeric preservation rules, figure-link conventions, and citation-orphan rules.

### Modified Capabilities
<!-- None — this introduces a new artifact alongside the existing paper rather than changing requirements of any existing capability. -->

## Impact

- **New files** under `paper/short/`: `00_abstract.md`, `01_introduction.md`, `02_background_related.md`, `03_math_tools.md`, `04_findings.md`, `05_discussion_conclusion.md`, `references.md`, `paper_short.md`.
- **Untouched**: `paper/paper.md`, `paper/sections/*.md`, `paper/references.md`, `results/figures/`, all source code, all data files. No experiments rerun, no figures regenerated.
- **Git**: a new feature branch `feat/paper-short` off `agent/writeup`. No merge to `agent/writeup` or `main` as part of this change — operator reviews via `git diff` and decides.
- **No build-system or dependency changes.** Plain markdown editing; `wc -w` and `grep` for verification.
- **Reader audience**: the professor is the primary reader. Cuts prioritize what they need (math tools, findings, ethics, limitations) over what is tutorial.

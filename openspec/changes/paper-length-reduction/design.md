## Context

The current paper at `paper/paper.md` is assembled from ten section files under `paper/sections/` plus an abstract held inline at the top of `paper.md` and a bibliography at `paper/references.md`. Total word count is 25,524. The driving feedback is the professor's "this is too long — roughly halve it"; two specific hints (the math does not need to be a tutorial; failed attempts can be compressed) and two structural redundancies surfaced by a scan (§4↔§9.1 duplicate Lec walkthroughs; §8 walks five hypotheses at near-equal depth when only one is load-bearing).

Stakeholders:
- Operator (paper author) — wants the original draft preserved unchanged and a side-by-side cut version for review before any merge.
- Implementing agent — needs a deterministic enough plan to drive prose cuts without inventing claims; every "what to keep / drop" decision needs an explicit rationale.
- Reader (professor) — has already read the long draft; values terseness and judgment over completeness.

Constraints:
- No empirical claim, number, or finding may change. All figures stay as-is and continue to live at `results/figures/`.
- The original `paper/paper.md` and `paper/sections/*.md` must stay byte-identical on this branch.
- This is a pure prose / structural pass — no experiments rerun, no code touched, no build-system changes.
- Work happens in the `gb-paper` worktree on a feature branch off `agent/writeup`.

## Goals / Non-Goals

**Goals:**
- Land an assembled `paper/short/paper_short.md` whose total word count falls in **10,500–13,500** (soft target ≈ 11,900; "roughly half").
- Preserve every headline numeric from the long draft verbatim: `42 / 42`, `1 / 50`, `6 / 50`, `18 / 42`, `Cohen's *d* = 2.87`, `86.6 %`, `17 / 42`, `40.5 %`, `median rank_95 = 6`, `~201`, `84`, `−0.08`, `0.04`, `≈ 0.96`, `+0.93`, `−0.015`, `2560`.
- Six top-level sections (down from ten) in F1→F2→F4→F3 abstract-matching order, with the math chapter reorganized as "tool, then immediate application" once per tool.
- Compress rejected cascade hypotheses (H1, H2, H3, H5) to a single "What was ruled out" paragraph (~150w combined); H4 keeps full ~600w treatment because it is the experimental contribution.
- Every figure link in the new paper resolves correctly from `paper/short/` (form: `../../results/figures/<name>.png`).
- `paper/short/references.md` contains no orphan entries (entries defined but not cited) and no missing entries (citations without definitions).

**Non-Goals:**
- Not changing any empirical claim, finding, or quantitative result.
- Not regenerating figures.
- Not editing `paper/paper.md` or `paper/sections/*.md` — the long draft stays intact.
- Not rewriting the abstract beyond mechanical fixes if cuts orphan a number (none expected).
- Not rerunning any experiments.
- Not merging the cut version onto `agent/writeup` or `main` — that decision is the operator's after diff review.
- Not introducing new citations; only pruning orphans from the existing bibliography.

## Decisions

**Decision 1 — Output to a new folder `paper/short/`, not in-place rewrite.**
- *Choice:* Create `paper/short/` with `00_abstract.md` … `05_discussion_conclusion.md`, plus `references.md` and the assembled `paper_short.md`. Leave `paper/paper.md` and `paper/sections/*.md` untouched.
- *Alternatives considered:* (a) Rewrite `paper/paper.md` in place. (b) Branch the long draft and rewrite on the branch. (c) New folder.
- *Why this:* Operator explicitly asked to keep the long version as reference. A new folder gives a clean side-by-side diff (`paper/sections/` vs. `paper/short/`) and zero risk of losing the original if the cut overshoots and needs to be redone.

**Decision 2 — Six sections in F1→F2→F4→F3 order, not ten in original order.**
- *Choice:* §1 Introduction, §2 Background & Related Work (merging old §2+§3), §3 Mathematical Tools (merging old §4 + §9.1), §4 Empirical Findings (merging old §5+§6+§7+§8 with internal F1→F2→F4→F3 ordering), §5 Discussion & Conclusion (merging old §9.2–9.5 + §10).
- *Alternatives considered:* (a) Keep ten-section layout and cut within each. (b) Six sections in original §5→§6→§7→§8 chronological order. (c) Six sections in abstract order (F1→F2→F4→F3).
- *Why this:* The largest single block of compressible text is the cross-section connective tissue that glued §5/§6/§7/§8 together; merging into one Findings chapter recovers that and removes four "what's next" intro paragraphs. Matching the abstract's narrative order makes the chapter read as one story rather than four mini-papers.

**Decision 3 — "Tool, then immediate application" math chapter.**
- *Choice:* §3 has three subsections (3.1 Setup, 3.2 Mirsky + partial-response prediction, 3.3 Near-orthogonality + anisotropy caveat, 3.4 Lazy training / NTK), each stating a tool and applying it once. §9.1's "Five Lecture Connections" framing is dissolved.
- *Alternatives considered:* (a) Keep old §4 (Math Framework) and old §9.1 (Lecture Connections) as separate chapters cut by ~30% each. (b) Drop §9.1 entirely and keep §4. (c) Drop §4 entirely and keep §9.1 reformatted as the math chapter. (d) Fuse the two as proposed.
- *Why this:* Direct response to the operator's "doesn't need a comprehensive guide on how the math works". Presenting Mirsky / near-orthogonality / NTK once with the empirical application attached reads cleaner for a reader who already knows the lectures than presenting them twice. The §4.3 / Connection 3 SVD-machinery duplicate collapses to one line; the Connection 5 thin-shell duplicate collapses entirely.

**Decision 4 — F1 cut hardest, F3 cut least.**
- *Choice:* F1 −66% (2,649 → ~900w), F2 −43% (2,101 → ~1,200w), F4 −51% (2,465 → ~1,200w), F3 −36% (2,334 → ~1,500w).
- *Alternatives considered:* Uniform −50% cut across all four findings.
- *Why this:* F1 is framing, not contribution — the numbers carry it and fit in a summary table plus one prose paragraph. F3 (the cascade) is the paper's experimental contribution and needs room for the careful "necessary but not isolated as sufficient" phrasing on D3. Uneven cuts honor the contribution structure rather than appearance of fairness.

**Decision 5 — H1/H2/H3/H5 compress to one combined paragraph; H4 keeps full treatment.**
- *Choice:* In §4.4 (the cascade), H1, H2, H3, H5 each get one sentence inside a single "What was ruled out" paragraph (~150w total). H4 (D3 stacked variant) gets ~600w with all numerics.
- *Alternatives considered:* (a) One paragraph per rejected hypothesis (~200w each, 800w total). (b) Drop H1/H2/H3/H5 entirely. (c) One combined paragraph (~150w).
- *Why this:* Direct response to the operator's "failed attempts can be compressed". The methodological contribution of the cascade is that each rejection sharpens the H4 attribution; one sentence each preserves that contribution. Dropping them entirely loses the cascade structure.

**Decision 6 — Preserve the source's careful "necessary, not sufficient" phrasing on D3.**
- *Choice:* §4.4 prose must read: *"D3 is necessary in combination with prior ingredients to produce the partial 40.5% result; we do not claim sufficient — Stage 2.5 unstacked isolation was not run, so we cannot rule out the possibility that D1 or D2 is doing the real work and the Gram-Schmidt layer is decorative."* The prose must NOT say "isolates Gram-Schmidt as load-bearing" or "the load-bearing ingredient is H4".
- *Alternatives considered:* (a) Lighter phrasing that's easier to compress. (b) Adopt the source's full phrasing verbatim.
- *Why this:* Source `paper/sections/07_abliteration_weight_diff.md:35` explicitly disclaims sufficiency. Adopting lighter phrasing would silently change a paper claim, which the non-goals forbid.

**Decision 7 — Figure links use markdown-relative `../../results/figures/<name>.png`.**
- *Choice:* Every `![…](…)` link in `paper/short/0N_*.md` and the assembled `paper_short.md` uses `../../results/figures/<name>.png`. Never repo-root-relative `results/figures/<name>.png`.
- *Alternatives considered:* (a) Repo-root-relative paths. (b) Absolute paths. (c) Markdown-relative paths.
- *Why this:* Both files live two levels below the repo root, so `../../results/figures/` resolves the same way from either. Repo-root-relative renders correctly in some tools (GitHub) but breaks in others (local pandoc, VSCode preview). Markdown-relative is the universal form.

**Decision 8 — Citation keys use the canonical `paper/references.md` form.**
- *Choice:* `[Park 2023]` (not `[Mikolov 2013]`, not `[Park 2024]`). The standard reference-engineering citation chain is `[Park 2023]` → `[Zou 2023]` → `[Arditi 2024]`.
- *Alternatives considered:* Cite from memory or from external knowledge of the linear-representation literature.
- *Why this:* `paper/references.md` is the canonical bibliography; any new citation must use the exact key defined there. Acceptance criterion §8 requires zero "missing" citations (cited but not defined). Pre-confirming canonical keys avoids audit-pass churn.

**Decision 9 — Per-section commits, prefix `feat(short):` / `refactor(short):` / `fix(short):`.**
- *Choice:* Each task ends with a commit. One file per commit where practical; the §4 findings file accumulates across §4.1–§4.5 with one commit per subsection.
- *Alternatives considered:* (a) Single end-of-work commit. (b) Per-section commits.
- *Why this:* Allows the operator to diff-review section by section, and gives clean rollback points if a particular cut overshoots.

**Decision 10 — `wc -w` and `grep` as the only verification tools.**
- *Choice:* Word counts via `wc -w paper/short/<file>.md`. Numerics audit via `grep -oE "<pattern>"`. Citation audit via literal-bracket extraction (`grep -oE "\[[^]]+\]"`) plus `comm` against the bibliography. Figure-path audit via `[ -f "$rel" ]` from the markdown file's directory.
- *Alternatives considered:* Pandoc render + manual review. A custom Python verifier.
- *Why this:* No build system exists in the repo; the verification needs to run from a plain shell. `grep` + `wc` + `comm` cover all acceptance criteria without new tooling.

## Risks / Trade-offs

- **[Risk]** Cuts go too deep and lose a load-bearing numeric → **Mitigation:** every section task ends with a `grep` numerics audit against the headline-numerics list; if any expected number is absent, the task fails and the cut is reverted.
- **[Risk]** Prose paraphrase changes a claim subtly (e.g., "sufficient" vs. "necessary") → **Mitigation:** Decision 6 mandates verbatim phrasing on the D3 nuance; spec acceptance requires the exact disclaimer string appears in §4.4.
- **[Risk]** New citation introduced with a non-canonical key → **Mitigation:** Decision 8 plus the orphan-/missing-audit step (`comm -23 cited.txt defined.txt`) before commit. If a citation is missing from the bibliography, the audit fails.
- **[Risk]** Figure link uses repo-root-relative path and renders in author's local tool but breaks for the operator → **Mitigation:** Decision 7 plus a path-form audit that rejects any link not starting with `../../results/figures/` (excluding `http(s)://`).
- **[Risk]** Word count comes in over the 10.5K–13.5K band → **Mitigation:** per-section word-count check after each task; if a section is more than 30% over its budget and reads bloated, do one tightening pass before moving on. Final-assembly word count gates Task 9 (acceptance audit).
- **[Risk]** Implementing agent edits a source file (`paper/sections/*.md` or `paper/paper.md`) by mistake → **Mitigation:** Task 9 Step 1 runs `git diff agent/writeup..feat/paper-short -- paper/sections/ paper/paper.md paper/references.md` and requires empty output before declaring done.
- **[Trade-off]** Soft word budgets (±15% per spec) mean the cut depth is judgment-call territory and not perfectly reproducible. Accepted because operator explicitly requested judgment over hitting a number, and the global 10.5K–13.5K band is the actual constraint.
- **[Trade-off]** Merging §5+§6+§7+§8 into one Findings chapter loses the per-section section-intro reframing that the long draft uses to remind the reader where they are. Accepted because the abstract + a new §4.5 connecting paragraph reconstruct that orientation more cheaply.

## Migration Plan

This change does not migrate anything — it adds a new folder alongside the existing paper. Steps:

1. Verify clean working tree on `agent/writeup` (`git status` clean, branch is `agent/writeup`).
2. Cut feature branch: `git checkout -b feat/paper-short`.
3. Create `paper/short/` with a `.gitkeep` to track the empty folder.
4. Write the six section files plus `references.md` plus `paper_short.md` (one task per file/section per the tasks artifact).
5. Run the acceptance audit (Task 9): originals untouched, all files exist, numerics intact, figures resolve, no orphan references, word counts in budget.
6. Push the branch (`git push -u origin feat/paper-short`) and surface to operator with total word count, per-section word counts, and any judgment calls flagged.
7. Operator reviews via `git diff agent/writeup..feat/paper-short -- paper/short/` and decides whether to merge.

**Rollback:** If the cut version is unsatisfactory, delete the branch (`git branch -D feat/paper-short`) and the long draft is unaffected. No state outside the new branch was touched.

## Open Questions

None at design time. Per-section judgment calls (which exact sentences to drop from a given paragraph) are intentionally left to the implementing agent rather than pre-specified. If a particular section reads bloated 30%+ over budget, the agent surfaces it for operator review before doing a deeper cut.

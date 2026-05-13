## ADDED Requirements

### Requirement: Output folder layout
The system SHALL produce a new folder `paper/short/` sibling to `paper/sections/`, containing exactly: `00_abstract.md`, `01_introduction.md`, `02_background_related.md`, `03_math_tools.md`, `04_findings.md`, `05_discussion_conclusion.md`, `references.md`, `paper_short.md`, and a `.gitkeep` to track the initially-empty folder. The original `paper/paper.md`, `paper/sections/*.md`, and `paper/references.md` SHALL remain byte-identical to their state on `agent/writeup`.

#### Scenario: All expected files present
- **WHEN** the change is complete and `ls paper/short/` is run
- **THEN** the listing contains `00_abstract.md`, `01_introduction.md`, `02_background_related.md`, `03_math_tools.md`, `04_findings.md`, `05_discussion_conclusion.md`, `references.md`, `paper_short.md`, and `.gitkeep`

#### Scenario: Original draft untouched
- **WHEN** `git diff agent/writeup..feat/paper-short -- paper/sections/ paper/paper.md paper/references.md` is run
- **THEN** the output is empty

### Requirement: Branch and commit hygiene
The system SHALL perform all work on a feature branch named `feat/paper-short` cut from `agent/writeup`. Each task SHALL end with a commit using a conventional-commits prefix matching the repo style (`feat(short):`, `refactor(short):`, `fix(short):`).

#### Scenario: Branch exists on remote
- **WHEN** `git branch -r | grep feat/paper-short` is run after Task 9 pushes
- **THEN** the output contains `origin/feat/paper-short`

#### Scenario: Commit messages are conventional
- **WHEN** `git log agent/writeup..feat/paper-short --pretty=format:%s` is run
- **THEN** every line starts with one of `feat(short):`, `refactor(short):`, `fix(short):`

### Requirement: Six-section narrative structure in F1→F2→F4→F3 order
The assembled `paper_short.md` SHALL contain exactly six top-level sections plus an abstract and a references block, in this order: Abstract, §1 Introduction, §2 Background and Related Work, §3 Mathematical Tools, §4 Empirical Findings, §5 Discussion and Conclusion, References. §4 SHALL present its findings in F1→F2→F4→F3 order matching the abstract.

#### Scenario: Section headers in assembled file
- **WHEN** `grep -n "^## " paper/short/paper_short.md` is run
- **THEN** the output lists, in order: `## Abstract`, `## Table of Contents`, `## 1. Introduction`, `## 2. Background and Related Work`, `## 3. Mathematical Tools`, `## 4. Empirical Findings`, `## 5. Discussion and Conclusion`, `## References`

#### Scenario: Findings ordered F1→F2→F4→F3
- **WHEN** `grep -n "^### 4\." paper/short/04_findings.md` is run
- **THEN** §4.1 is F1 (behavioral baselines), §4.2 is F2 (activation-space rank-1 + standard-recipe null), §4.3 is F4 (cross-method weight-diff orthogonality), §4.4 is F3 (causal-isolation cascade)

### Requirement: Word-count band for the assembled paper
The assembled `paper/short/paper_short.md` SHALL have a total word count in the band **10,500–13,500** (soft target around 11,900; "roughly half" of the 25,524-word original). Per-section word budgets are guidance, not hard limits — ±15% of the per-section target is acceptable; ±30% requires a tightening pass and operator surfacing.

#### Scenario: Assembled total in budget
- **WHEN** `wc -w paper/short/paper_short.md` is run
- **THEN** the count is in the closed interval [10500, 13500]

#### Scenario: Per-section targets (soft)
- **WHEN** `wc -w paper/short/0[0-5]*.md` is run
- **THEN** approximate per-file counts are: `00_abstract.md` ≈ 390 ± 15, `01_introduction.md` ≈ 1,100 ± 165, `02_background_related.md` ≈ 1,800 ± 270, `03_math_tools.md` ≈ 1,800 ± 270, `04_findings.md` ≈ 5,000 ± 750, `05_discussion_conclusion.md` ≈ 1,800 ± 270

### Requirement: Headline numerics preserved verbatim
The assembled `paper_short.md` SHALL contain every member of the headline-numerics list at least once, with the exact textual form below: `42 / 42`, `1 / 50`, `6 / 50`, `18 / 42`, `Cohen's *d* = 2.87`, `86.6 %` (or `86.6%`), `17 / 42`, `40.5 %` (or `40.5%`), `median rank_95 = 6` (or `rank_95 = 6`), `~201`, `84`, `−0.08`, `0.04`, `≈ 0.96`, `+0.93`, `−0.015`, `2560`.

#### Scenario: Numerics audit passes
- **WHEN** `grep -cE "<pattern>" paper/short/paper_short.md` is run for each pattern in the headline-numerics list
- **THEN** every count is ≥ 1

### Requirement: Empirical claims unchanged
No empirical claim, finding, or quantitative result from the original draft SHALL be changed, paraphrased into a different meaning, or invented. Where the original draft makes a careful "necessary, not sufficient" disclaimer (e.g., on D3 in the cascade), the cut SHALL preserve that disclaimer.

#### Scenario: D3 sufficiency disclaimer preserved
- **WHEN** `grep -F "we do not claim sufficient" paper/short/04_findings.md` is run
- **THEN** the output contains that exact phrase (or a paraphrase that includes both "necessary" and an explicit "not sufficient" or "not claim sufficient" statement about D3)

#### Scenario: D3 not overclaimed
- **WHEN** `grep -E "isolates Gram-Schmidt as load-bearing|load-bearing ingredient is H4" paper/short/04_findings.md` is run
- **THEN** the output is empty

### Requirement: Tool-then-application math chapter
§3 Mathematical Tools SHALL have four subsections — 3.1 Setup, 3.2 Mirsky's bound and partial-response prediction, 3.3 Near-orthogonality with anisotropy caveat, 3.4 Lazy training / NTK as plausibility — each stating a tool once and applying it once. Content duplicated between old §4 and old §9.1 (Lec 7 SVD machinery / Connection 3, thin-shell / Connection 5) SHALL NOT appear in any other section.

#### Scenario: Four subsections present
- **WHEN** `grep -n "^### 3\." paper/short/03_math_tools.md` is run
- **THEN** the output lists §3.1, §3.2, §3.3, §3.4 in order

#### Scenario: No "lecture connection" framing leaks into §5
- **WHEN** `grep -E "Connection [1-5]|Lec [5-9]|lecture connection" paper/short/05_discussion_conclusion.md` is run
- **THEN** the output is empty

### Requirement: Rejected cascade hypotheses compressed; H4 keeps full treatment
§4.4 SHALL collapse H1, H2, H3, H5 into a single "What was ruled out" paragraph of ≤ 200 words combined (one sentence each). H4 (D3 stacked variant: chat-template direction + winsorization + Gram-Schmidt against harmless mean) SHALL keep a full treatment of approximately 600 words with all source numerics intact.

#### Scenario: Combined "ruled out" paragraph contains all four
- **WHEN** the cut version is read
- **THEN** the "What was ruled out" paragraph in §4.4 mentions H1, H2, H3, H5 by their abbreviation or by their concrete intervention (bnb int8, chat-template direction alone, winsorization alone, norm-preserving biprojection)

#### Scenario: H4 numerics intact
- **WHEN** `grep -oE "17 / 42|40\\.5|Gram-Schmidt|D3" paper/short/04_findings.md` is run
- **THEN** each pattern appears at least once

### Requirement: Figure links resolve from `paper/short/`
Every markdown image link `![…](…)` in `paper/short/0N_*.md` and `paper/short/paper_short.md` SHALL be of the form `../../results/figures/<name>.png` (markdown-relative from `paper/short/`). Repo-root-relative `results/figures/<name>.png` links and absolute paths SHALL NOT appear. Every link SHALL resolve to an existing file.

#### Scenario: All links resolve
- **WHEN** every `![…](…)` link target in `paper/short/paper_short.md` is extracted and tested with `[ -f "$rel" ]` from inside `paper/short/`
- **THEN** every link target exists

#### Scenario: Link form is markdown-relative
- **WHEN** image-link targets are extracted from `paper/short/paper_short.md` and filtered to exclude `http(s)://`
- **THEN** every remaining target starts with `../../results/figures/`

### Requirement: Citation keys canonical; no orphans, no missing
Every citation in `paper/short/0[1-5]*.md` SHALL be a `[Key]` literal-bracket form whose key matches an entry defined in `paper/short/references.md` as `- **[Key]** …`. Canonical keys SHALL be used (e.g., `[Park 2023]`, `[Zou 2023]`, `[Arditi 2024]`; not `[Mikolov 2013]` or `[Park 2024]`). After Task 7's prune pass, `paper/short/references.md` SHALL contain no orphan entries (defined but not cited).

#### Scenario: No missing citations
- **WHEN** the set of cited keys from `paper/short/0[1-5]*.md` is compared (`comm -23`) against the set of defined keys in `paper/short/references.md`
- **THEN** the difference is empty

#### Scenario: No orphan citations
- **WHEN** the set of defined keys in `paper/short/references.md` is compared (`comm -23`) against the set of cited keys from `paper/short/0[1-5]*.md`
- **THEN** the difference is empty

### Requirement: Abstract verbatim
`paper/short/00_abstract.md` SHALL contain the original abstract prose copied verbatim from `paper/paper.md`, with no edits except adding the leading `## Abstract` heading if the source uses a different heading level. Light touch-up is permitted only if cuts elsewhere orphan a number cited in the abstract; no such touch-up is expected.

#### Scenario: Abstract word count matches source
- **WHEN** `wc -w paper/short/00_abstract.md` is run
- **THEN** the count is within ±10 words of the source abstract's word count (~393)

### Requirement: References pruned after prose settles
The system SHALL copy `paper/references.md` verbatim to `paper/short/references.md`, then remove only entries whose `[Key]` is not cited in any of `paper/short/0[1-5]*.md`. No reference entry SHALL be rewritten; only whole bullets are removed.

#### Scenario: Orphan prune does not rewrite entries
- **WHEN** `git diff paper/references.md paper/short/references.md` is run after Task 7
- **THEN** the diff shows only whole-bullet deletions (no in-line edits to retained bullets)

### Requirement: Assembled file structure
`paper/short/paper_short.md` SHALL be the concatenation of: a title block (`# Geometry of Alignment` + subtitle + course + date), the abstract section, a Table of Contents linking to §1–§5 and References, the five section files in order, and the references block with the heading demoted from `# References` to `## References`.

#### Scenario: Title and TOC present at the top
- **WHEN** `head -20 paper/short/paper_short.md` is read
- **THEN** the output contains `# Geometry of Alignment`, the subtitle, the course identifier, the date, and the `## Abstract` heading

#### Scenario: References at level `##`
- **WHEN** `grep -n "^## References$\|^# References$" paper/short/paper_short.md` is run
- **THEN** at least one `## References` line is present and no `# References` line is present

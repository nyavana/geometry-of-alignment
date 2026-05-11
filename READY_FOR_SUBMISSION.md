# READY_FOR_SUBMISSION.md

Single canonical artifact the operator reads after M5 (paper writeup) completes,
before the final merge to `main`. Mirrors the role of `STATUS_FOR_HUMAN.md` at
the M4 gate, but scoped to the M5 paper deliverable.

Branch: `agent/writeup` (worktree `../gb-paper`)
Compiled paper: `paper/paper.md` (~25,500 words, 10 sections, integrated draft)
Per-section sources of truth: `paper/sections/01_introduction.md` … `10_conclusion.md`
Bibliography: `paper/references.md`
Numeric-claim ledger: `paper/sources.md`

---

## Spec coverage (`openspec/changes/gemma-only-execution-plan/specs/research-paper/spec.md`)

| Requirement / Scenario | Status | Where |
|---|---|---|
| Paper structure with survey and experiments — all 10 sections present | ✓ | `paper/paper.md` §1–§10 |
| Section 7 narrative shape — M2c + OBLITERATUS + TrevorJS, no MoE/router | ✓ | `07_abliteration_weight_diff.md` §7.1–§7.6 |
| M6 outcomes integrated — partial 17/42 = 40.5% result, not "clean win" | ✓ | §1 Finding 3 · §8 throughout · §9 Connection 4 |
| M6 ablation appendix — five hypotheses + Stage 0/1/1.5/3a results | ✓ | §8 §§ 1–4 + Table |
| M6 claims phrased as "necessary in combination", never "sufficient" | ✓ | §7.3 · §8.4 · §9.3 |
| Survey covers RLHF, DPO, CAI, abliteration, repr. engineering, ft attacks, over-refusal | ✓ | §2 · §3 |
| ≥15 citations | ✓ | `references.md` (32 entries) |
| ≥3 published Gemma 4-era abliteration variants discussed | ✓ | §3 — OBLITERATUS, TrevorJS, Heretic; §7 — OBLITERATUS + TrevorJS weight-diff geometry |
| §4 invokes Mirsky (singular values), explicitly notes Hoffman-Wielandt obstruction on rectangular `W` | ✓ | §4.2 |
| §4 embeds Mirsky heatmaps + 3 anisotropy figures | ✓ | 5 figures in §4 (`mirsky_bound_heatmap_{d3,m2b}.png` + `projection_energy_L15.png` + `learned_directions_cosine_L15.png` + `activation_anisotropy_L15.png`) |
| Anisotropy caveat for Lec 5 framing | ✓ | §4.4 — mean pairwise cos 0.958, ratio 0.71 |
| §9 ≥5 explicit lecture connections, ≥1 quantitative, ≥1 from each bucket | ✓ | §9 Connections 1–5; Connection 2 carries the quantitative tie (median `‖E‖_F / ‖W‖_F` = 0.022 ↔ §8's 40.5%) |
| §9 lazy/NTK framing phrased as "plausible", not "predicts" | ✓ | §9 Connection 4 verbatim |
| Figures — min set (a)–(l) | ✓ | 13 figures embedded across `paper/paper.md` |
| §10 ethics — misuse acknowledged, fragility→improvement, selective safety as constructive flip | ✓ | §10.3 |
| Slides — hiking opener + ≥3 figures + Math Framework slide + course-connections | ✓ | `paper/presentation-slides/` (delivered 2026-05-07 + 2026-05-08; M5 10.14) |
| Writeup gated by `STATUS_FOR_HUMAN.md` operator green-light | ✓ | `STATUS_FOR_HUMAN.md` line 211 + 340 |
| All numeric claims traceable to `results/...` + commit hash | ✓ | `paper/sources.md` (~75 distinct claims tabulated; 9 marked Unverified, none of them headline) |

---

## Sub-task completion ledger (M5 10.1–10.18)

| Sub-task | Status | Commit(s) |
|---|---|---|
| 10.1 green-light verified | ✓ | `b93c67c` (operator) |
| 10.2 §1 Introduction | ✓ | `47d2787` + `c70a507` (humanizer) |
| 10.3 Mirsky bound numerics | ✓ | `c8891c3` |
| 10.4 anisotropy figures | ✓ | `eabb5ce` |
| 10.5 §4 Math Framework | ✓ | `e428501` + `ad74345` |
| 10.6 §5 Over-Refusal | ✓ | `809f6be` + `8a3ced1` |
| 10.7 §6 Mechanistic | ✓ | `636c5fb` + `e1cd637` |
| 10.8 §7 Abliteration + Weight Diff | ✓ | `47a4f8c` + `9eb8439` |
| 10.9 §8 M6 Cascade integration | ✓ | `461eb95` + `e1c4a6f` |
| 10.10 §9 Discussion | ✓ | `062d7b6` + `907e122` |
| 10.11 §10 Conclusion + Ethics | ✓ | `d21e5a0` + `81b2eaf` |
| 10.11a §2 + §3 humanizer backfill | ✓ | `4be76a9` + `84d953d` |
| 10.12 integrate + compile `paper.md` | ✓ | `afcfdc4` + `dc91759` + `db1aa45` |
| 10.13 section-reference audit | ✓ | `9d40f2d` (7 fixes in §2/§3) |
| 10.14 slide deck | ✓ | delivered earlier (2026-05-07 / 2026-05-08) |
| 10.15 `sources.md` | ✓ | `9c918b8` (~75 claims, 9 Unverified non-headline) |
| 10.16 consistency + paper-wide humanizer | ✓ | `1645b94` + `5f9fbfd` |
| 10.17 this file | ✓ | (this commit) |
| 10.18 final push | pending | operator confirms after review |

---

## Known caveats / honest disclosures

1. **OBLITERATUS weight count rounding.** §1 / §7 / §9 / §10 quote "~201 (200 with `rank_95` computed)". On-disk dedup'd JSON has exactly 200 entries with `rank_95` populated and 367 entries with non-zero Frobenius before the LM-only filter. The "~201" hedge survives because the dedup figure was originally reported as 201 in `STATUS_FOR_HUMAN.md` M3 section before the borrower-alias removal landed cleanly; the on-disk number is canonical.

2. **§9 length.** §9 Discussion is 3,765 words — longer than the 1–2 page target. The spec required ≥5 lecture connections + a quantitative tie + over-parametrization framing + limitations + future-work in one section; trimming further would have dropped spec-mandated content.

3. **`phrasing_sensitivity.png` not embedded.** §5.5 explains: the M2a 3.10 variant-run was killed at 16% completion, so per-prompt consistency saturates at 1.00 across all rows and the figure would over-sell a non-finding. Path referenced, figure not embedded.

4. **9 Unverified entries in `sources.md`.** None are headline numbers; all are either derivable from committed artifacts but not separately tabulated (per-layer Cohen's d curve, PCA per-layer table, etc.) or recorded only in `STATUS_FOR_HUMAN.md` (emergency-responder 4% probe). Operator may choose to leave as-is or generate tabulations on request.

5. **Stage 2.5 unstacked isolation never run.** §8 + §9 are careful to phrase D3 (Gram-Schmidt) as "necessary in combination" — Stage 2.5 was deferred (cost not justified given M6's partial-result terminus). The dispatch-prompt scenario "M6 paper claims phrased as necessary, not sufficient" governs the prose.

6. **`paper.md` is the submission artifact.** Per-section files in `paper/sections/` remain canonical authoring sources; `paper.md` is the integrated compile. If post-review edits are needed, prefer editing per-section files and re-running 10.12 integration, rather than editing `paper.md` directly — otherwise the per-section and integrated views will drift.

---

## Operator action

1. **Review** `paper/paper.md` end-to-end (or section-by-section via `paper/sections/`).
2. **Verify** the four headline findings against the per-section CSV/JSON paths in `paper/sources.md`.
3. **Decide** on the OBLITERATUS "~201 (200)" hedge — keep as-is or rewrite to a clean "200".
4. **Merge** `agent/writeup` → `main` (per `openspec/changes/gemma-only-execution-plan/tasks.md` task 11.2).
5. **Archive** the openspec change once everything is on `main` (`openspec archive gemma-only-execution-plan --skip-specs --yes`).

The writeup branch is fully pushed to `origin/agent/writeup`; no local-only commits remain.

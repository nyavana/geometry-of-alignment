# M3 Summary — Weight-Diff Analysis (OBLITERATUS vs TrevorJS)

This document captures the M3 phase findings for the gemma-only execution
plan. The block below is parseable for M4 9.x consumption.

## Parseable summary

```
M3-summary:
  OBLITERATUS:
    low-rank-verdict: rank-low (multi-rank, NOT rank-1) — median rank_95=6
    fraction-changed: 201/2094 after dedup
    top-1-cosine-vs-refusal-direction: 0.000–0.158 (median 0.043), headline L40 down_proj=0.158
    shared-tensor-dedup-count: 36
  TrevorJS:
    low-rank-verdict: rank-1 (median rank_95=1)
    fraction-changed: 84/2094 after dedup
    top-1-cosine-vs-refusal-direction: 0.000–0.209 (median 0.042), headline L2 o_proj=0.209
    shared-tensor-dedup-count: 36
  cross-method:
    median-cosine: -0.08 (from 7.9)
    interpretation: Refusal in Gemma 4 has a low-rank but multi-modal weight
      footprint — TrevorJS removes it via a single rank-1 stroke per layer
      while OBLITERATUS spreads its edits across many directions and many
      component types, and the two methods' top-1 modification directions
      live in nearly orthogonal subspaces (median |cos|=0.08), implying
      that "the geometry of refusal" admits a continuum of equally-effective
      low-rank fixes rather than one canonical safety basis.
```

## Per-task pointers

- **7.10** Cross-reference of M2b refusal directions vs top-1 left singular
  vectors of weight-diffs:
  `$RESULTS_DIR/weight_diffs/refusal_direction_vs_singular_vector.csv`,
  `$RESULTS_DIR/figures/refusal_direction_vs_singular_vector.png`.
  Key finding: median |cos| ≈ 0.04 for both variants — the
  activation-derived refusal direction is largely orthogonal to the top-1
  singular direction of either method's weight edits.
- **7.11** Shared-tensor de-dup policy (Gemma 4 K/V borrowers at layers
  24–41): `$RESULTS_DIR/weight_diffs/.shared_tensor_handling.md`,
  `*_dedup` JSONs/CSVs/PNGs alongside the originals. 36 alias slots removed
  per variant; numerator unchanged (no aliased tensor had a non-zero diff
  in either variant).
- **7.12** Full SV spectra of top-3 most-modified weights per variant:
  `$RESULTS_DIR/figures/singular_value_spectra_per_method.png`. TrevorJS
  σ_1/σ_2 ≈ 110×, OBLITERATUS σ_1/σ_2 ≈ 1.05–1.33×.
- **7.13** Component-type breakdown:
  `$RESULTS_DIR/weight_diffs/component_type_breakdown.csv`. TrevorJS
  exclusively edits `o_proj` and `down_proj` (rank-1 each); OBLITERATUS
  edits q/k/v/o, gate/up/down MLP, and embed_tokens with mean rank_95
  ranging from 2 (embed_tokens) to 720 (down_proj).

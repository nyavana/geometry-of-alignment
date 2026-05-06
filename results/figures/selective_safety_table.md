# Selective safety: category-direction cosine similarity table

Source: `$RESULTS_DIR/activations/category_cosine_summary.json` produced by `scripts/m2c_category_directions.py` (commit `f2e2485`). Per-layer values computed by slicing `refuse_activations.pt` from M2b 4.3 by `category` and computing `mean(refuse_<cat>) - mean(safe_control)` per layer, then normalizing — no GPU re-extraction. Entries below are means across all 42 Gemma 4 E4B-it layers (`mean`) and across the M2b peak band layers 4-17 (`peak`). Diagonal is 1.0 by construction.

## Cosine similarity (mean across all 42 layers)

| | emergency_medical | wilderness_survival | home_safety | chemistry_safety | mental_health | should_refuse |
|---|---:|---:|---:|---:|---:|---:|
| emergency_medical | +1.000 | +0.905 | +0.946 | +0.933 | +0.931 | +0.001 |
| wilderness_survival | +0.905 | +1.000 | +0.950 | +0.926 | +0.935 | -0.008 |
| home_safety | +0.946 | +0.950 | +1.000 | +0.958 | +0.935 | -0.024 |
| chemistry_safety | +0.933 | +0.926 | +0.958 | +1.000 | +0.900 | -0.024 |
| mental_health | +0.931 | +0.935 | +0.935 | +0.900 | +1.000 | -0.020 |
| should_refuse | +0.001 | -0.008 | -0.024 | -0.024 | -0.020 | +1.000 |

## Cosine similarity (mean across peak band, layers 4-17)

| | emergency_medical | wilderness_survival | home_safety | chemistry_safety | mental_health | should_refuse |
|---|---:|---:|---:|---:|---:|---:|
| emergency_medical | +1.000 | +0.932 | +0.969 | +0.946 | +0.950 | +0.026 |
| wilderness_survival | +0.932 | +1.000 | +0.953 | +0.932 | +0.932 | +0.036 |
| home_safety | +0.969 | +0.953 | +1.000 | +0.961 | +0.939 | +0.014 |
| chemistry_safety | +0.946 | +0.932 | +0.961 | +1.000 | +0.909 | +0.009 |
| mental_health | +0.950 | +0.932 | +0.939 | +0.909 | +1.000 | +0.012 |
| should_refuse | +0.026 | +0.036 | +0.014 | +0.009 | +0.012 | +1.000 |

## Headline numbers

- Over-refuse categories form a tight cluster: pairwise mean cosine across the 10 over-refuse pairs = **+0.932** (range +0.900 to +0.958).
- Each over-refuse category vs `should_refuse` is essentially orthogonal: mean cosine across the 5 over_refuse-vs-should_refuse pairs = **-0.015** (range -0.024 to +0.001).

## Interpretation

The five over-refuse categories share a common direction in residual-stream space (cosine ~+0.9, peak band tighter). The `should_refuse` direction is geometrically distinct from all five (cosine ~0). This is the geometric precondition for selective abliteration: a projection along the over-refuse cluster mean would, in principle, leave the `should_refuse` axis untouched. The behavioural cost — that standard rank-1 mean-diff abliteration does not remove refusal even when the direction is known (see `alpha_sweep.png`, `layer_subset_comparison.png` and the M2c 5.7 commit body) — caps how much practical headroom this geometric separation buys on Gemma 4 E4B-it under 8-bit quantization. The accompanying selective-abliteration evaluation that would close the loop is M2c 5.10, deferred to a separate GPU dispatch.

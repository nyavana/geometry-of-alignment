## ADDED Requirements

### Requirement: Weight diff computation
The system SHALL load original (`google/gemma-4-E4B-it`) and published-variant model weights from safetensors files on CPU, compute the element-wise difference for each parameter tensor, and report Frobenius norm, relative change, and max absolute change.

#### Scenario: Layer-by-layer diff computed
- **WHEN** original and a published-variant model directory are provided
- **THEN** the system SHALL iterate over all common parameter keys, compute `W_diff = W_variant - W_original`, and save a JSON results file (one per variant) under `results/weight_diffs/<variant_slug>/weight_diff_results.json` with per-parameter statistics

#### Scenario: Pre-flight key/shape compatibility
- **WHEN** a variant's state-dict is opened
- **THEN** the system SHALL assert that its parameter keys are a subset of the base model's keys and that all common keys have matching shapes; if either assertion fails, the system SHALL log to `results/weight_diffs/.compat_log.md` and SHALL skip that variant rather than producing a corrupt diff

#### Scenario: Only changed parameters flagged
- **WHEN** the diff is computed
- **THEN** the system SHALL flag parameters with Frobenius norm > 1e-6 as changed and report the count of changed vs unchanged parameters

### Requirement: SVD rank analysis of weight modifications
The system SHALL compute SVD of the weight diff for all 2D parameter tensors that were significantly modified (relative change > 0.001).

#### Scenario: Effective rank computed
- **WHEN** SVD is computed on a weight diff matrix
- **THEN** the system SHALL report the number of singular values needed to explain 95% and 99% of the diff's energy (Frobenius norm squared), and the fraction of energy in the top-1 singular value

#### Scenario: Low rank per variant
- **WHEN** a variant's effective rank at 95% is 1-3 across most modified layers
- **THEN** this SHALL be interpreted (in the analysis output and in the paper) as evidence consistent with a rank-1-style abliteration; when effective rank is higher, this SHALL be interpreted as evidence the variant used a different technique (e.g., LoRA fine-tuning, multi-direction projection, or norm-preserving biprojection)

#### Scenario: Top singular vectors saved
- **WHEN** a weight diff has relative change > 0.001
- **THEN** the system SHALL save the top-5 left singular vectors, singular values, and right singular vectors per variant for downstream cross-method comparison and refusal-direction cross-reference

### Requirement: Per-variant per-layer modification profile
The system SHALL produce a bar chart showing total Frobenius norm of weight changes per layer, one chart per variant, indicating which layers were most heavily modified.

#### Scenario: Per-layer change plotted per variant
- **WHEN** all weight diffs for a variant are computed
- **THEN** the system SHALL aggregate Frobenius norms by layer index and produce a bar chart with layer index on the x-axis, saved as `results/figures/weight_diff_per_layer_<variant_slug>.png`

### Requirement: Cross-method comparison of published Gemma 4 E4B uncensoring
The system SHALL compare the weight diffs of two or more published variants by overlaying per-layer Frobenius profiles and by computing cosine similarity between corresponding top singular vectors.

#### Scenario: Per-layer Frobenius overlay
- **WHEN** weight diffs from two or more variants exist
- **THEN** the system SHALL produce a single bar chart with grouped bars per layer index showing each variant's Frobenius norm, saved as `results/figures/weight_diff_per_layer_overlay.png`

#### Scenario: Cross-method singular vector cosine
- **WHEN** two variants have both modified the same parameter (per-tensor rel-change > 0.001 in both)
- **THEN** the system SHALL compute cosine similarity between their top-1 left singular vectors and write a row to `results/weight_diffs/cross_method_cosine_table.csv` with columns `parameter_name, layer_idx, variant_a, variant_b, cosine, rank_a_95, rank_b_95`

#### Scenario: Single-variant fallback
- **WHEN** only one variant's weight diff is available (e.g., TrevorJS pre-flight failed)
- **THEN** the system SHALL skip cross-method outputs without erroring and SHALL note the fallback in `results/weight_diffs/.compat_log.md`

### Requirement: Cross-reference weight-diff singular vectors with M2b refusal directions (quantitative)
The system SHALL quantitatively compare the top-1 left singular vector from the weight diff against the per-layer refusal direction computed by the activation-analysis pipeline.

#### Scenario: Cosine similarity per layer
- **WHEN** `results/activations/refusal_directions.pt` (from M2b) and `results/weight_diffs/<variant_slug>/svd_results.pt` (from M3) both exist for a given layer where the variant modified `self_attn.o_proj.weight` or `mlp.down_proj.weight`
- **THEN** the system SHALL compute the cosine similarity between the refusal direction and the top-1 left singular vector of the modified weight's diff, and write a row to `results/weight_diffs/refusal_direction_vs_singular_vector.csv` with columns `variant_slug, layer_idx, parameter_name, cosine`

#### Scenario: Cross-reference figure produced
- **WHEN** the cosine table is complete
- **THEN** the system SHALL produce `results/figures/refusal_direction_vs_singular_vector.png` showing cosine similarity vs layer index, with one line per variant

### Requirement: Architectural quirk handling
The Gemma 4 architecture has shared K/V tensors (per OBLITERATUS model card: layers 24–41 reference layer 24's `k_proj`/`v_proj`). The system SHALL identify shared tensors and report each unique tensor's diff exactly once, to avoid double-counting in per-layer plots.

#### Scenario: Shared tensors de-duplicated
- **WHEN** computing per-layer Frobenius aggregates
- **THEN** the system SHALL detect tensors whose `data_ptr` matches a previously-seen tensor (or whose state-dict key indicates sharing per the model config) and SHALL count each unique tensor in only one layer bucket (the owning layer); the de-duplication count SHALL be logged to `results/weight_diffs/.shared_tensor_handling.md`

#### Scenario: Documented in analysis output
- **WHEN** producing the per-layer overlay chart
- **THEN** layers whose K/V tensors are shared (i.e., owned by an earlier layer) SHALL be visually distinguished or annotated, so a reader does not misread "no diff" as "no modification"

### Requirement: Singular value spectrum visualization
The system SHALL produce singular value bar charts for the most significantly modified weight matrices in each variant.

#### Scenario: Spectrum plotted for significant diffs
- **WHEN** a weight diff has relative change > 0.01
- **THEN** the system SHALL produce a bar chart of the top-10 singular values, saved as `results/figures/singular_value_spectrum_<variant_slug>_<param_name>.png`

### Requirement: Component-type summary (dense model only)
The system SHALL report total modification magnitude broken down by component type. Component categories: `attention.q_proj`, `attention.k_proj`, `attention.v_proj`, `attention.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `embedding`, `norm`, `other`.

#### Scenario: Component breakdown produced
- **WHEN** all weight diffs for a variant are computed
- **THEN** the system SHALL aggregate Frobenius norms by component category and write `results/weight_diffs/<variant_slug>/component_type_breakdown.csv`. **MoE-specific categories (expert, router, shared expert) MUST NOT appear** — Gemma 4 E4B is dense.

## ADDED Requirements

### Requirement: Core abliteration algorithm
The system SHALL modify model weights to project out the refusal direction from the residual stream. For each target layer, the modification SHALL apply `W_new = W - alpha * d * (d^T @ W)` to the specified weight matrices, where `d` is the refusal direction and `alpha` is the projection strength.

#### Scenario: Full abliteration with alpha=1.0
- **WHEN** refusal directions for all 42 layers and alpha=1.0 are provided
- **THEN** the system SHALL modify `self_attn.o_proj.weight` and `mlp.down_proj.weight` at each layer, and the resulting model SHALL refuse 0 or near-0 of the benchmark's should-comply prompts

#### Scenario: Partial abliteration with alpha < 1.0
- **WHEN** alpha=0.5 is specified
- **THEN** the system SHALL apply half-strength projection, resulting in reduced but not eliminated refusal behavior

#### Scenario: Random direction control
- **WHEN** a random unit vector is used instead of the computed refusal direction
- **THEN** the model's refusal behavior SHALL remain largely unchanged (validating that the real refusal direction is meaningful)

### Requirement: Ablation study framework
The system SHALL support systematic sweeps across multiple experimental axes and save results in a structured JSON format.

#### Scenario: Alpha sweep
- **WHEN** an alpha sweep is requested with values [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
- **THEN** the system SHALL abliterate and evaluate the model at each alpha value, recording refusal rate and capability scores

#### Scenario: Layer subset sweep
- **WHEN** a layer subset sweep is requested
- **THEN** the system SHALL test at minimum: all 42 layers, global-only (7 layers), sliding-only (35 layers), first half (0-20), second half (21-41), and last 10 layers

#### Scenario: Prompt count sweep
- **WHEN** a prompt count sweep is requested with values [10, 25, 50, 100, 200]
- **THEN** the system SHALL recompute refusal directions using only that many prompt pairs and report how refusal removal effectiveness changes

### Requirement: Capability preservation measurement
The system SHALL evaluate the abliterated model on general capability benchmarks to detect capability degradation.

#### Scenario: Capability benchmarks run
- **WHEN** an abliterated model is produced
- **THEN** the system SHALL evaluate it on at least 2 capability benchmarks (e.g., MMLU subset, GSM8K subset) and report scores alongside the original model's scores

### Requirement: Selective safety via category-specific directions
The system SHALL compute separate refusal directions for different prompt categories and test whether removing one category's direction preserves refusal in other categories.

#### Scenario: Category-specific directions computed
- **WHEN** activations are available for emergency_medical, wilderness_survival, and should_refuse prompts
- **THEN** the system SHALL compute a separate refusal direction for each category against the safe_control baseline

#### Scenario: Direction similarity analysis
- **WHEN** category-specific directions are computed
- **THEN** the system SHALL report pairwise cosine similarity between all category directions at each layer, determining whether selective abliteration is geometrically feasible

#### Scenario: Selective abliteration applied
- **WHEN** the medical refusal direction is removed while the weapons refusal direction is preserved
- **THEN** the system SHALL evaluate: (a) over-refusal rate on emergency_medical prompts (target: < 10%), (b) appropriate refusal rate on should_refuse prompts (target: > 80%)

### Requirement: Model save and export
The system SHALL save abliterated models in HuggingFace format for downstream evaluation.

#### Scenario: Abliterated model saved
- **WHEN** abliteration is complete
- **THEN** the system SHALL save the modified model weights and tokenizer to a specified directory using `model.save_pretrained()`

### Requirement: bf16 vs bnb int8 edit-path isolation (M6 H1)
The system SHALL support applying the rank-1 abliteration projection to either a bf16-loaded model or a bitsandbytes int8 in-place edit (controlled by the existing `--use-8bit` flag). The two paths SHALL be testable as independent variables so that any positive result from omitting `--use-8bit` is attributable specifically to the bnb int8 in-place edit wrapper, not to "int8 quantization at inference" generically.

#### Scenario: bf16 edit-path run
- **WHEN** `abliterate.py` is invoked without `--use-8bit` against `google/gemma-4-E4B-it`
- **THEN** the system SHALL load the base model in bf16, apply `W_new = W - alpha * d * (d^T @ W)` to `self_attn.o_proj.weight` and `mlp.down_proj.weight` at each target layer in bf16, and save a bf16 checkpoint suitable for downstream transformers-backend evaluation

#### Scenario: Framing assertion in run output
- **WHEN** the bf16 edit-path run produces a positive result (e.g., `should_refuse` ≤ 30% on the 48-prompt stratified subset)
- **THEN** the run summary SHALL describe the result as "bnb int8 in-place edit-path test," NOT as a generic "precision toggle" or "int8 quantization is the cause." The explicit counterexample SHALL be cited: HauhauCS's quantized-but-uncensored Q8 GGUF rules out "quantization at inference" as the mechanism

#### Scenario: VRAM fallback under 16 GB constraint
- **WHEN** the bf16 base model OOMs on a 16 GB GPU during the edit
- **THEN** the system SHALL fall back to `device_map="auto"` with CPU offload of late layers, and SHALL NOT silently fall back to a smaller model class (e.g., E2B), since changing the model class would confound any positive result

### Requirement: Norm-preserving biprojection variant (M6 H5)
The system SHALL support a `--norm-preserving` flag on `src/abliterate/abliterate.py` that implements a magnitude+direction biprojection preserving each row norm `‖W_i‖` of the modified weight, mirroring the algebra used by TrevorJS's published Gemma 4 abliteration. This is an optional algorithm path activated only when vanilla projection has been ruled out as sufficient (M6 Stage 3a).

#### Scenario: Norm-preserving projection applied
- **WHEN** `abliterate.py --norm-preserving` is invoked with a refusal direction `d` and target weight `W`
- **THEN** the system SHALL compute a modified weight `W'` such that for every row `i`: the component of `W_i` along `d` is removed (or attenuated by `alpha`), and `‖W'_i‖ ≈ ‖W_i‖` to float32 precision

#### Scenario: Post-impl self-check on random inputs
- **WHEN** the `--norm-preserving` implementation lands and is exercised before its first real run
- **THEN** an in-script self-check (matching the project's existing "smoke-test" pattern from M3 7.6 — the repo has no test framework configured) SHALL generate a random unit vector `d ∈ R^2560` and a random matrix `W ∈ R^(d_out × 2560)`, apply the norm-preserving projection, and assert `|‖W'_i‖ − ‖W_i‖| < 1e-5` for every row, exiting non-zero on failure

#### Scenario: Comparison against vanilla projection
- **WHEN** both vanilla and `--norm-preserving` variants are run with the same direction artifact and alpha
- **THEN** the system SHALL produce two distinct checkpoints whose downstream `should_refuse` rates are reported side-by-side, so the load-bearing nature of norm preservation can be isolated from direction-quality effects

## ADDED Requirements

### Requirement: Activation extraction with forward hooks
The system SHALL hook into each of the 42 transformer layers of Gemma 4 E4B-it via `register_forward_hook` and capture the residual stream output tensor (shape: batch x seq_len x 2560) for each prompt.

#### Scenario: Extract activations for a batch of prompts
- **WHEN** a list of prompts and a position strategy ("last" token or "mean" over positions) are provided
- **THEN** the system SHALL return a dictionary mapping layer index (0-41) to a tensor of shape (num_prompts, 2560)

#### Scenario: Activations stored on CPU to preserve VRAM
- **WHEN** a forward hook captures activations
- **THEN** the captured tensor SHALL be detached and moved to CPU before storage, and the hook SHALL clear previous activations before each new forward pass

#### Scenario: Per-prompt category metadata persisted
- **WHEN** activations are saved to disk
- **THEN** a sibling `prompt_metadata.json` SHALL record, for each prompt index in the activation tensor, the benchmark `prompt_id`, `category`, and `expected` fields, so downstream consumers (e.g., selective-safety category-specific direction computation) can slice the saved activations by category without re-running forward passes on the GPU

### Requirement: Refusal direction computation
The system SHALL compute the refusal direction at each layer as the normalized mean difference between refuse-class and comply-class activations.

#### Scenario: Mean-diff refusal direction
- **WHEN** refuse activations (N_r, 2560) and comply activations (N_c, 2560) are provided for a layer
- **THEN** the system SHALL compute `direction = normalize(mean(refuse) - mean(comply))` and return a unit vector of dimension 2560

#### Scenario: PCA-based refusal direction
- **WHEN** the PCA method is selected
- **THEN** the system SHALL compute PCA on the combined refuse+comply activations and return the top principal component, oriented so that it points from comply-mean to refuse-mean

### Requirement: Layer-wise signal strength analysis
The system SHALL compute a separation score at each layer measuring how distinguishable refuse and comply activations are along the refusal direction.

#### Scenario: Separation score computed
- **WHEN** refuse and comply activations and the refusal direction are available for a layer
- **THEN** the system SHALL project all activations onto the refusal direction, compute `(mean_refuse_proj - mean_comply_proj) / pooled_std`, and return this as the separation score

#### Scenario: Sliding vs global attention comparison
- **WHEN** separation scores are computed for all 42 layers
- **THEN** the system SHALL produce a bar chart colored by layer type (blue for sliding attention, red for global attention) and report mean separation for each type

### Requirement: Refusal rank analysis
The system SHALL determine the effective dimensionality of the refusal subspace via PCA at each layer.

#### Scenario: Rank analysis at a layer
- **WHEN** PCA with 20 components is fitted on the combined refuse+comply activations at a given layer
- **THEN** the system SHALL report the number of components needed to explain 95% and 99% of variance, and the fraction of variance explained by the top-1 component

### Requirement: Activation visualization
The system SHALL produce 2D projections (UMAP or t-SNE) of activations colored by refuse/comply class.

#### Scenario: Single-layer visualization
- **WHEN** a layer index is specified
- **THEN** the system SHALL produce a scatter plot with refuse points in red and comply points in blue, saved as PNG

#### Scenario: Multi-layer grid visualization
- **WHEN** a list of layer indices is specified
- **THEN** the system SHALL produce a grid of UMAP plots (4 columns) showing how refuse/comply separation evolves across layers

### Requirement: Chat-template-aware activation extraction
The system SHALL support extracting per-layer residual-stream activations through the same `tokenizer.apply_chat_template(...)` path used by the inference / evaluation pipeline (`src/benchmark/evaluate.py`), so that the activation space in which the refusal direction is computed matches the activation space in which it is ablated against. This is exposed as a `--use-chat-template` flag on `src/mechanistic/extract_activations.py`.

#### Scenario: Chat-template extraction enabled
- **WHEN** `extract_activations.py --use-chat-template` is run on a benchmark prompt set
- **THEN** the system SHALL tokenize each prompt via `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")` (matching `evaluate.py:143`), run the forward pass, and write the resulting per-layer activations to `refuse_activations_chat.pt` / `comply_activations_chat.pt`

#### Scenario: Chat-template artifacts SHALL NOT overwrite M2b raw-prompt artifacts
- **WHEN** `extract_activations.py --use-chat-template` writes its outputs
- **THEN** the existing M2b raw-prompt artifacts (`refuse_activations.pt`, `comply_activations.pt`, and `refusal_directions.pt`) SHALL remain untouched on disk. The whole point of M6 H2 is to compare the two input spaces; silently overwriting M2b would destroy the comparison baseline. The `_chat` suffix is mandatory

#### Scenario: Direction comparison against raw-prompt baseline
- **WHEN** both `refusal_directions.pt` (raw-prompt M2b artifact) and a chat-template-derived direction artifact exist for the same layer
- **THEN** the system SHALL report the per-layer cosine similarity between the two; the expected range is `0 < |cos| < 1` (different but related), and a value at either extreme (≈0 or ≈1) SHALL be surfaced as a diagnostic finding

### Requirement: Composable direction-build helper (M6 Stage 2)
The system SHALL provide `src/mechanistic/build_directions_v2.py` as a single helper that produces per-layer refusal-direction artifacts with three composable, independent flags: `--use-chat-template`, `--winsorize-pct <pct>`, and `--orthogonalize-against-harmless-mean`. The flags compose; each variant artifact stacks the previous flags plus one new ingredient, so the first variant to crack identifies the marginal load-bearing ingredient.

#### Scenario: Variant D1 — chat-template direction only
- **WHEN** `build_directions_v2.py --use-chat-template` is invoked
- **THEN** the system SHALL re-extract activations through the chat template, compute `normalize(mean(refuse_chat) − mean(comply_chat))` per layer, and save 42 unit-norm vectors of dim 2560 to the specified output path

#### Scenario: Variant D2 — D1 + winsorization at 99.5th percentile
- **WHEN** `build_directions_v2.py --use-chat-template --winsorize-pct 99.5` is invoked
- **THEN** the system SHALL clip per-layer activations element-wise at the 99.5th percentile *before* taking the class means, then compute `normalize(mean(refuse_clipped) − mean(comply_clipped))`, and SHALL log a per-layer pre-clip and post-clip max-norm to the artifact directory; at the peak refusal layer (L15 per M2b), post-clip max-norm SHALL be ≤ 80% of pre-clip max-norm — failing this threshold means winsorization had no measurable effect at the layer that matters most, and the variant SHALL be flagged as suspect rather than silently produced

#### Scenario: Variant D3 — D2 + Gram-Schmidt against harmless mean
- **WHEN** `build_directions_v2.py --use-chat-template --winsorize-pct 99.5 --orthogonalize-against-harmless-mean` is invoked
- **THEN** the system SHALL perform a double-pass Gram-Schmidt of the per-layer direction against `mean_comply_clipped` (the same clipped mean used by D2 — D3 stacks on D2, so the harmless-mean reference is the winsorized one, not the raw `mean_comply`) and SHALL produce vectors satisfying `|dot(direction, mean_comply_clipped)| < 1e-4` in float32

#### Scenario: Each variant artifact preserves output shape and unit-norm contract
- **WHEN** any combination of D1/D2/D3 flags is selected
- **THEN** the resulting artifact SHALL contain 42 unit-norm vectors of dim 2560 (one per Gemma 4 E4B layer), keyed by layer index, in the same `.pt` schema as M2b's `refusal_directions.pt`

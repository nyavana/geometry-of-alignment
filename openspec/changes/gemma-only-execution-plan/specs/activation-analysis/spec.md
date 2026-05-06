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

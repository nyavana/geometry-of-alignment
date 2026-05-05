## ADDED Requirements

### Requirement: Weight diff computation
The system SHALL load original and cracked model weights from safetensors files on CPU, compute the element-wise difference for each parameter tensor, and report Frobenius norm, relative change, and max absolute change.

#### Scenario: Layer-by-layer diff computed
- **WHEN** original and modified model directories are provided
- **THEN** the system SHALL iterate over all common parameter keys, compute `W_diff = W_modified - W_original`, and save a JSON results file with per-parameter statistics

#### Scenario: Only changed parameters flagged
- **WHEN** the diff is computed
- **THEN** the system SHALL flag parameters with Frobenius norm > 1e-6 as changed and report the count of changed vs unchanged parameters

### Requirement: SVD rank analysis of weight modifications
The system SHALL compute SVD of the weight diff for all 2D parameter tensors that were significantly modified (relative change > 0.001).

#### Scenario: Effective rank computed
- **WHEN** SVD is computed on a weight diff matrix
- **THEN** the system SHALL report the number of singular values needed to explain 95% and 99% of the diff's energy (Frobenius norm squared), and the fraction of energy in the top-1 singular value

#### Scenario: Low rank indicates abliteration
- **WHEN** the effective rank at 95% is 1-3 across most modified layers
- **THEN** this SHALL be interpreted as evidence of abliteration (rank-1 weight perturbation), consistent with the model card's claim of "abliterated weights"

#### Scenario: Top singular vectors saved for comparison
- **WHEN** a weight diff has relative change > 0.001
- **THEN** the system SHALL save the top-5 left singular vectors, singular values, and right singular vectors for cross-comparison with Person B's refusal directions

### Requirement: Per-layer modification profile
The system SHALL produce a bar chart showing total Frobenius norm of weight changes per layer, indicating which layers were most heavily modified by the cracking process.

#### Scenario: Per-layer change plotted
- **WHEN** all weight diffs are computed
- **THEN** the system SHALL aggregate Frobenius norms by layer index and produce a bar chart with layer index on the x-axis

### Requirement: MoE expert-level analysis
For MoE models (Qwen3.5-35B-A3B), the system SHALL separately analyze modifications to expert weights, router/gate weights, shared expert weights, and attention weights.

#### Scenario: Expert modification heatmap
- **WHEN** the model is MoE with identifiable expert parameters
- **THEN** the system SHALL produce a heatmap (layer x expert_index) showing the Frobenius norm of weight changes per expert, revealing whether specific experts encode safety behavior

#### Scenario: Router modification detection
- **WHEN** router/gate parameters are found in the weight diff
- **THEN** the system SHALL report whether the router was modified and interpret this as evidence that safety is (or is not) encoded in expert routing

#### Scenario: Component-type summary
- **WHEN** the analysis is complete
- **THEN** the system SHALL report total modification magnitude broken down by component type: attention, MLP, expert, router, shared expert, embedding, norm

### Requirement: Singular value spectrum visualization
The system SHALL produce singular value bar charts for the most significantly modified weight matrices.

#### Scenario: Spectrum plotted for significant diffs
- **WHEN** a weight diff has relative change > 0.01
- **THEN** the system SHALL produce a bar chart of the top-10 singular values, saved as PNG

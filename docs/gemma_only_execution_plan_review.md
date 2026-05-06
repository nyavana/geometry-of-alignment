# Gemma-Only Execution Plan Review

Date: 2026-05-05

Reviewed change: `openspec/changes/gemma-only-execution-plan`

## Verdict

The plan makes sense. The OpenSpec change validates, the Gemma-only pivot is coherent, and replacing the Qwen MoE workstream with same-model Gemma 4 E4B comparisons is a stronger paper strategy for the available hardware and timeline.

The main risk is not the high-level scientific direction. The risk is that agents can execute the tasks in subtly incompatible ways because several artifact paths, output formats, and analysis conventions are underspecified or do not match the current code.

## Local Checks Run

- `openspec validate gemma-only-execution-plan` passed.
- `openspec list` showed only `gemma-only-execution-plan` active, with `15/109` tasks complete.
- Benchmark JSON parses successfully.
- `scripts/audit_benchmark.py` passed with 344 base prompts, 646 variants, 990 total prompt texts, schema/id/label checks passing, and no near-duplicate warnings at the configured threshold.

One local issue surfaced: `python` is not on PATH unless the shared environment is sourced or `.venv/bin/python` is used. Dispatch prompts should either source `shared/env.sh` or call `.venv/bin/python` explicitly.

## Highest-Value Improvements

1. Clarify artifact handoff.

   `.gitignore` ignores `results/**/*.pt`, `results/**/*.json`, `results/**/*.png`, `*.log`, and `models/`. This means branches will not naturally carry `refusal_directions.pt`, figures, logs, or abliterated model files. Add an explicit rule: generated artifacts live under `$RESULTS_DIR`, and each branch commits a small manifest with exact artifact paths, hashes, commands, and commit ids.

2. Align benchmark output format with current code.

   The task list asks for CSVs such as `results/refusal_rates/gemma4_e4b_base.csv`, but `src/benchmark/evaluate.py` currently writes `evaluation_results.json` inside an output directory, and `src/benchmark/analyze_results.py` expects that layout. Either update the task/spec language to use directory-per-model JSON outputs, or add an explicit CSV summary generation task.

3. Tighten SVD output and cosine conventions.

   The spec references `svd_results.pt`, while current code saves `significant_diff_svd.pt`. Pick one name. Also specify that singular-vector cosine comparisons should use `abs(cosine)` because SVD vector signs are arbitrary. For refusal-direction comparison, explicitly compare `U[:, 0]` from `o_proj` and `down_proj` diffs because the left singular vector matches residual-stream output dimension.

4. Make weight key compatibility stricter.

   The spec currently allows variant keys to be a subset of base keys. For this project, that is too permissive because missing shared K/V tensors are exactly a known Gemma 4 failure mode. Require all expected core base tensors to exist and shape-match for OBLITERATUS and TrevorJS, with only explicitly allowlisted metadata/adapters allowed to differ.

5. Define refusal activation classes precisely.

   The tasks say M2b should use `should_refuse + over-refused`, but current extraction code uses `should_refuse` and `safe_control`; the over-refusal categories are defined but not used. Specify whether refusal activations come from expected labels or actual model refusals from M2a.

6. Position the project abliteration correctly.

   The repo implementation is a textbook vanilla rank-1 projection baseline. Published Gemma 4 variants use stronger methods, including winsorized activations, whitened SVD, attention-head surgery, norm preservation, and harmful/harmless orthogonalization. The plan should explicitly frame the local M2c result as a controlled baseline rather than the expected best method.

## Ambiguities Likely To Mislead Agents

- "Commit smoke log" conflicts with `*.log` being ignored.
- "Pull/copy refusal_directions.pt from another branch" will fail unless the artifact is stored outside git or an operator copies it.
- "Full activation extraction: ~200 should_refuse + over-refused" conflicts with the current benchmark having 42 base `should_refuse` prompts and extraction code not using M2a actual refusals.
- "Capability preservation: MMLU + GSM8K subsets" is underspecified; there is no current local code path for those benchmarks.
- "Cross-precision validation: E2B BF16 cosine similarity vs E4B 8-bit" is scientifically useful but dimension/model mismatch handling needs to be specified, since E2B and E4B may not have directly comparable hidden sizes or layer counts.
- "Shared K/V de-dup by `data_ptr`" may not work from separately loaded safetensors state dicts. If tensors are loaded as separate CPU tensors, data pointers may not reveal sharing. Prefer config/key-based ownership rules and document the mapping.

## External Work Comparison

The plan is aligned with Arditi et al.'s refusal-direction result: refusal can be mediated by a one-dimensional residual-stream direction and controlled by intervention. It also fits XSTest's over-refusal framing: safe prompts plus unsafe contrasts are the right benchmark shape.

The main difference from recent community Gemma 4 work is method sophistication:

- OBLITERATUS reports Gemma 4 NaN activation issues, shared K/V handling, winsorized activations, whitened SVD, and attention-head surgery.
- TrevorJS uses norm-preserving biprojected abliteration, per-layer directions, harmless-mean orthogonalization, and cross-dataset validation over 686 prompts.
- Heretic optimizes refusal removal jointly with KL divergence to the original model.
- Jim Lai's norm-preserving biprojection writeup argues that vanilla projection can disturb weight norms and degrade model geometry.
- Harmfulness/refusal separation work suggests selective safety should distinguish "harmfulness" from surface-level "refusal."

This means the plan is strong as a research comparison: local vanilla ablation as a baseline, published variants as stronger methods, and weight-diff analysis as the explanation layer.

## Recommendation

Proceed with the Gemma-only plan, but patch the OpenSpec before dispatching agents for M2/M3. The highest-priority patch should be an "artifact and output contract" section covering:

- `$RESULTS_DIR` as the canonical artifact location.
- Per-task manifest format.
- Exact benchmark result layout.
- Exact SVD artifact names.
- Cosine sign convention.
- Strict compatibility checks for safetensors.

After that, the plan should be clear enough for agents to execute without accidentally producing mutually incompatible artifacts.

## Sources Consulted

- Local OpenSpec files under `openspec/changes/gemma-only-execution-plan/`
- Local code under `src/benchmark/`, `src/mechanistic/`, `src/abliterate/`, and `src/weight_diff/`
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction": https://arxiv.org/abs/2406.11717
- Rottger et al., "XSTest": https://arxiv.org/abs/2308.01263
- Zhao et al., "LLMs Encode Harmfulness and Refusal Separately": https://arxiv.org/abs/2507.11878
- OBLITERATUS Gemma 4 E4B model card: https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED
- TrevorJS Gemma 4 E4B model card: https://huggingface.co/TrevorJS/gemma-4-E4B-it-uncensored
- TrevorS Gemma 4 abliteration repo: https://github.com/TrevorS/gemma-4-abliteration
- Heretic repo: https://github.com/p-e-w/heretic
- Jim Lai, "Norm-Preserving Biprojected Abliteration": https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

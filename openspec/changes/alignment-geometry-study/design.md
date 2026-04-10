## Context

This is a research project for EECS 6699 (Mathematics of Deep Learning) at Columbia University. The team has 4 members, ~1 month, a server with an Intel i7-14700K, 100GB DDR4 RAM, and an NVIDIA 4070 Ti Super 16GB GPU. The professor has approved the topic.

The core observation: safety-aligned LLMs over-refuse beneficial queries (e.g., emergency medical instructions), and published "cracked" models remove safety via abliteration -- a rank-1 weight perturbation that projects out a single "refusal direction" from the residual stream. Both the Qwen3.5-35B-A3B-Uncensored (HauhauCS) and Gemma-4-31B-JANG_4M-CRACK models confirm use of abliterated weights.

The primary mechanistic analysis model is Gemma 4 E4B-it: 42 layers (35 sliding attention + 7 global attention), hidden_size=2560, 8 attention heads, 2 KV heads, intermediate_size=10240, 262K vocab with Per-Layer Embeddings (PLE). At 8-bit quantization it uses ~7.5GB VRAM, leaving ~8.5GB for activations and overhead.

The weight diff analysis target is Qwen3.5-35B-A3B: a MoE model with 256 experts (8 routed + 1 shared per token), 40 layers, running on CPU with 100GB RAM.

## Goals / Non-Goals

**Goals:**
- Quantify over-refusal rates across state-of-the-art aligned models on safety-adjacent beneficial queries
- Locate the refusal mechanism in Gemma 4 E4B's residual stream: which layers, what dimensionality, does it differ between sliding and global attention layers
- Reproduce abliteration from scratch on Gemma 4 E4B and determine the minimum intervention needed (layers, alpha, prompt count)
- Demonstrate selective safety: remove over-refusal on medical/survival queries while preserving refusal on genuinely harmful queries
- Reverse-engineer a published cracked MoE model (Qwen3.5-A3B) to determine whether safety lives in experts, routers, or attention weights
- Connect findings to course concepts: over-parametrization (safety is low-dimensional in a massive parameter space), matrix perturbation (abliteration is a rank-1 update), NTK/linearization (safety is approximately linear in activation space)
- Produce a research paper (survey + experiments) and presentation slides

**Non-Goals:**
- Building a production-ready safety system or alignment technique
- Training or fine-tuning models from scratch (we study existing aligned and cracked models)
- Achieving state-of-the-art jailbreaking (the goal is understanding, not breaking)
- Studying closed-source models (only open-weight models where we can inspect internals)
- Mathematical proofs (the project is experimental with course-concept connections)

## Decisions

### Decision 1: Gemma 4 E4B at 8-bit as primary mechanistic analysis model
**Choice**: Use Gemma 4 E4B-it loaded via `transformers` + `bitsandbytes` at 8-bit quantization for all activation extraction and abliteration work.
**Rationale**: E4B has stronger safety training than E2B (more nuanced refusal behavior to study), and at 8-bit (7.5GB) leaves 8.5GB headroom on the 16GB GPU. The refusal direction is computed from mean activations over 200+ prompts, so quantization noise averages out. E2B at full BF16 (9.6GB) serves as a validation model to confirm findings hold across precisions and sizes.
**Alternative considered**: E4B at BF16 (15GB) -- too tight, only 1GB headroom after model weights, would OOM during activation extraction. E2B only -- weaker safety behavior, less interesting to analyze.

### Decision 2: llama.cpp for benchmark evaluation, transformers for mechanistic work
**Choice**: Person A uses llama-cpp-python for fast benchmark inference (can run on CPU with 100GB RAM or GPU). Persons B and C use transformers for activation hooks and weight modification.
**Rationale**: Separates GPU-dependent work (mechanistic analysis) from GPU-optional work (benchmark evaluation), enabling true parallel execution. llama.cpp handles GGUF quantized models efficiently for batch inference; transformers provides the `register_forward_hook` API needed for activation extraction.
**Alternative considered**: All-transformers approach -- would create GPU contention between team members.

### Decision 3: CPU-based weight diff analysis for Qwen MoE model
**Choice**: Person D loads both Qwen3.5-35B-A3B original and uncensored checkpoints on CPU using safetensors, computes weight diffs layer-by-layer.
**Rationale**: The 100GB RAM comfortably holds two copies of the model weights for direct comparison. No GPU needed -- SVD and Frobenius norm computations run on CPU. This keeps the GPU free for Persons B and C.
**Alternative considered**: Loading via transformers with device_map -- unnecessary overhead when we only need raw weight tensors, not forward passes.

### Decision 4: Two-stage refusal classifier (keyword + heuristic)
**Choice**: Regex-based pattern matching for refusal detection, with manual validation on 50-100 samples.
**Rationale**: An LLM-as-judge approach would be more accurate but introduces dependency on another model and is slow for 400+ prompts x multiple models. Keyword patterns catch 90%+ of refusals reliably. Manual validation catches edge cases and gives confidence intervals.
**Alternative considered**: LLM-as-judge (GPT-4 / Claude) -- expensive, slow, adds external dependency. Fine-tuned classifier -- overkill for this scale.

### Decision 5: Shared benchmark prompt JSON created collaboratively in week 1
**Choice**: All 4 team members contribute to a single `benchmark_prompts.json` in the first 3 days, then work independently.
**Rationale**: The benchmark is the only true shared dependency. Creating it together ensures buy-in and coverage, and once finalized, every person can work without waiting for anyone else.
**Alternative considered**: One person creates the full benchmark -- risks bias and gaps in category coverage.

### Decision 6: Project out of both attention and MLP output projections
**Choice**: Abliteration modifies `self_attn.o_proj.weight` and `mlp.down_proj.weight` at each target layer.
**Rationale**: The refusal direction appears in the residual stream, which is the sum of attention output and MLP output. Both pathways can inject the refusal signal. Ablation study will test attention-only vs MLP-only vs both to determine where the signal primarily originates.
**Alternative considered**: Modifying only MLP down projections -- may miss refusal signal injected via attention heads.

## Risks / Trade-offs

- **[8-bit quantization noise affects refusal direction quality]** -> Mitigation: Validate by comparing refusal directions computed on E2B at BF16 vs E4B at 8-bit. If cosine similarity > 0.9, quantization is acceptable. This also becomes a paper finding.
- **[Gemma 4 model structure differs from expected hook paths]** -> Mitigation: First action for Person B is `print(model)` to verify layer attribute names. Adjust hook registration paths before any extraction runs.
- **[Over-refusal benchmark may not trigger refusal on some models]** -> Mitigation: Include the exact prompt that triggered refusal on Gemma 4 E4B on the user's phone (Heimlich maneuver). Test prompts on the target model early and iterate on phrasing.
- **[Selective safety may not work (category refusal directions too similar)]** -> Mitigation: This is a valid negative result. If cosine similarity between medical-refusal and weapons-refusal directions is >0.9, the paper reports that refusal is a single monolithic direction rather than category-specific. Still a contribution.
- **[GPU contention between Person B and Person C in weeks 2-3]** -> Mitigation: Person B develops on E2B first, Person C implements abliteration algorithm on CPU. When GPU needed: Person B extracts activations (1-2 days), then Person C gets GPU for abliteration experiments. Can also split: one uses E4B, other uses E2B simultaneously.
- **[Weight diff analysis may show the cracked model used fine-tuning, not abliteration]** -> Mitigation: If the diff is high-rank (not rank-1), this is still interesting -- it means the cracked model used a different technique than claimed, and comparing approaches becomes the finding.
- **[Ethics of publishing over-refusal bypass techniques]** -> Mitigation: Frame all work as improving alignment. The selective safety experiment is the constructive contribution. Paper includes ethics section. Professor has approved the topic.

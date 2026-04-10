# The Geometry of Alignment: How Safety Lives and Dies in Modern LLMs

**Course**: EECS 6699 — Mathematics of Deep Learning, Columbia University  
**Instructor**: Prof. Predrag R. Jelenkovic  
**Team Size**: 4 members  
**Timeline**: ~4 weeks (April 2026)

---

## 1. Motivation

You're hiking in a remote area with no cell service. Your companion starts choking and can't breathe. You pull out your phone and ask the locally installed LLM how to perform the Heimlich maneuver. It responds:

> *"I'm sorry, but I can't provide medical advice. Please seek professional medical help."*

This actually happened. We tested Gemma 4 E4B running locally and it refused to answer emergency medical queries. The safety mechanism meant to prevent harm was causing it.

At the same time, "cracked" versions of these models exist (e.g., `Qwen3.5-35B-A3B-Uncensored`, `Gemma-4-31B-JANG_4M-CRACK`) that remove all safety guardrails through a technique called **abliteration**. The method projects out a single "refusal direction" from the model's residual stream. No fine-tuning, no training data. A rank-1 weight perturbation and the model stops refusing.

That raises some questions we think are worth investigating:

- Where does safety/refusal behavior actually live inside the network?
- Why is it so fragile? Why does removing one vector break it?
- Could you remove over-refusal on medical queries while keeping refusal on genuinely harmful ones?

## 2. Research questions

We want to answer five things:

First, **how bad is over-refusal in practice?** We plan to measure how frequently current aligned models refuse beneficial queries: emergency medical instructions, survival skills, safety procedures. Not cherry-picked examples, but systematic evaluation across hundreds of prompts.

Second, **where in the network is refusal encoded?** We want to know which layers carry the refusal signal, what dimensionality the refusal subspace has, and whether the 7 global attention layers in Gemma 4 E4B behave differently from the 35 sliding attention layers.

Third, **what makes abliteration work?** A rank-1 weight perturbation removes safety. We want to find the minimum intervention: how many layers need to change, how many prompt pairs you need to compute the direction, and what happens when you scale the projection strength.

Fourth, **can we be selective?** If we compute separate refusal directions for "medical queries" and "weapons queries," are those directions different enough that we can remove one and keep the other?

Fifth, **what did the cracked models actually change?** By comparing the weights of published uncensored models against their originals, we can check whether the modifications are consistent with abliteration and whether MoE architectures (with their expert routing) reveal anything about how alignment is structured.

## 3. Connection to course content

The most direct connection is to over-parametrization (Lectures 5-8). Safety alignment takes millions of RLHF training examples, but all of that collapses to roughly a rank-1 subspace in the residual stream. Billions of parameters, and the "safety behavior" uses almost none of that capacity.

Abliteration is a rank-1 perturbation of weight matrices: `W_new = W - alpha * d * d^T * W`. The fact that this changes model behavior so drastically is exactly the kind of thing the matrix perturbation theory from Lecture 7 (Hoffman-Wielandt inequality) can reason about. How do small changes in a matrix propagate through a deep network?

The success of abliteration also suggests that safety behavior is approximately linear in activation space. You find a direction, project it out, and the behavior disappears. This fits the NTK / lazy training view from Lectures 8-9, where the network's function is well-approximated by a linear model around its trained weights.

And over-refusal is just a generalization failure. The model learned "refuse when medical harm is mentioned" from safety training but overgeneralized to beneficial medical queries. That connects to the generalization bounds from Lectures 8-9, though here we're observing the failure case rather than the success.

## 4. Models and compute

### Hardware
- Server: Intel i7-14700K, 100GB DDR4 RAM, NVIDIA 4070 Ti Super 16GB

### Models

| Model | Type | Role | Runs on |
|-------|------|------|---------|
| Gemma 4 E4B-it | Dense, 42 layers, PLE | Primary mechanistic analysis (8-bit, 7.5GB VRAM) | GPU |
| Gemma 4 E2B-it | Dense, PLE | Validation model (BF16, 9.6GB) | GPU |
| Qwen3.5-35B-A3B | MoE, 256 experts | Weight diff analysis target | CPU (100GB RAM) |
| Qwen3.5-35B-A3B-Uncensored | MoE, cracked | Reverse engineering target | CPU |

### Gemma 4 E4B architecture (primary model)
- 42 transformer layers: 35 sliding attention + 7 global attention
- Hidden size: 2560, attention heads: 8 (2 KV heads, GQA)
- Intermediate (MLP): 10,240
- Per-Layer Embeddings (PLE): 262K vocab per layer
- At 8-bit: ~7.5GB VRAM, leaving 8.5GB for activations and overhead

## 5. Experimental design

### 5.1 Benchmark construction (all members, Week 1)
Build `benchmark_prompts.json` with 300-400 prompts across 8 categories: emergency_medical, wilderness_survival, home_safety, chemistry_safety, mental_health, gray_zone, should_refuse, safe_control. Each prompt includes rephrased variants for phrasing sensitivity testing.

### 5.2 Cross-model refusal evaluation (Person A)
Evaluate all models on the benchmark using llama.cpp. Classify responses as refuse/comply using regex pattern matching validated against 50+ manual labels. Produce refusal rate heatmaps and phrasing sensitivity analysis.

### 5.3 Activation extraction and mechanistic analysis (Person B)
Hook into each of E4B's 42 layers via `register_forward_hook`. Collect residual stream activations for ~200 refuse and ~200 comply prompts. Compute the refusal direction per layer (mean difference), analyze its dimensionality with PCA, and visualize with UMAP. Compare sliding vs global attention layers.

### 5.4 Abliteration and selective safety (Person C)
Reproduce abliteration from scratch: compute the refusal direction, project it out of `o_proj.weight` and `down_proj.weight`. Run ablation sweeps over alpha (0 to 2.0), layer subsets (global-only, sliding-only), and prompt count (10 to 200). Include a random-direction control. Then attempt selective safety: compute category-specific refusal directions and remove only the medical over-refusal.

### 5.5 Weight diff analysis (Person D)
Load Qwen3.5-A3B original and uncensored on CPU. Compute element-wise weight diffs, SVD rank analysis, per-layer Frobenius norms. For the MoE architecture: check which experts were modified, whether the router changed, and whether the shared expert was touched.

## 6. Work division and timeline

| Week | Person A (Benchmark) | Person B (Mechanistic) | Person C (Abliteration) | Person D (Weight diff + Paper) |
|------|---------------------|----------------------|------------------------|-------------------------------|
| 1 | Build benchmark, set up llama.cpp | Build activation hooks, test on E2B | Implement abliteration, test on E2B | Download models, start weight diffs, start survey |
| 2 | Run evals on all original models | Extract activations from E4B, compute directions | Apply abliteration to E4B, run sweeps | Complete weight diff and SVD analysis |
| 3 | Eval abliterated models, analyze | Layer analysis, UMAP, cross-model validation | Selective safety experiments, capability tests | Cross-reference with B's directions, draft paper |
| 4 | Write Section 4, slides | Write Section 5, slides | Write Section 6, slides | Integrate paper, write Sections 1-3 and 7-9 |

All work runs in parallel after the shared benchmark is created in the first few days.

## 7. Deliverables

1. A 9-section research paper with a survey (RLHF, DPO, representation engineering, abliteration) and experimental results
2. Presentation slides covering the emergency scenario, key results, and course connections
3. Python codebase in `src/` with benchmark, mechanistic, abliterate, and weight_diff modules
4. Benchmark prompt dataset and all experimental results in `data/` and `results/`

## 8. Ethics

This work is about understanding alignment to improve it. The constructive contribution is selective safety: showing that over-refusal can be fixed without gutting genuine safety. We acknowledge that abliteration techniques have dual-use potential and discuss this in the paper. The motivating scenario, someone dying because the model refused to explain the Heimlich maneuver, is why understanding alignment fragility matters. The goal is better safety mechanisms, not weaker ones.

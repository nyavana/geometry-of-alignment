# Topic Selection Process — How We Arrived at "The Geometry of Alignment"

This document records the exploration and decision-making process that led to the final project topic.

---

## Course Context

**Course**: EECS 6699 — Mathematics of Deep Learning, Columbia University  
**Instructor**: Prof. Predrag R. Jelenkovic  

The course covers:
- Expressiveness / approximation power of neural networks (Lectures 1-4)
- Depth separation, Kolmogorov Superposition Theorem (Lectures 4-5)
- Over-parametrization, interpolation, high-dimensional geometry (Lectures 5-6)
- Convergence of GD to global minima, Neural Tangent Kernel (Lectures 6-7)
- Lazy training, PAC learning, generalization bounds, Rademacher complexity (Lectures 8-9)
- Concentration inequalities, matrix perturbation (Hoffman-Wielandt) (Lectures 6-7)

## Project Requirements

From the course syllabus:
- Groups of 4, can be mathematical, experimental, or a mix
- Deliverables: presentation/slides, paper (survey + research), code
- **Key constraint**: "Experiments need to study properties of deep learning networks rather than focus on solving a particular problem by training on a particular dataset"
- The topic should be related to the course but does not need to be directly covered

## Team Preferences

- Prefer **experimental** over theoretical/mathematical
- Prefer **state-of-the-art, trending** topics for portfolio value
- Prefer **Python** over R
- No math experts on the team — need something accessible
- Prefer **application-oriented** rather than unsolved math problems
- Want something connected to **real-world problem solving**

## Hardware Available

- Intel i7-14700K
- 100GB DDR4 RAM
- NVIDIA 4070 Ti Super 16GB VRAM

---

## Round 1: Initial Brainstorming (Theory-Oriented)

Seven topics were initially explored, all studying properties of deep learning:

| # | Topic | Trending | Portfolio | Course Fit | Difficulty |
|---|-------|----------|-----------|------------|------------|
| 1 | Scaling Laws | ★★★★★ | ★★★★★ | ★★★ | Medium |
| 2 | Grokking | ★★★★★ | ★★★★ | ★★★★★ | Low-Med |
| 3 | Double Descent | ★★★★ | ★★★★ | ★★★★★ | Medium |
| 4 | Loss Landscape Visualization | ★★★ | ★★★★ | ★★★★ | Medium |
| 5 | Mechanistic Interpretability | ★★★★★ | ★★★★★ | ★★★ | Med-High |
| 6 | KAN vs MLP | ★★★★ | ★★★★★ | ★★★★★ | Medium |
| 7 | Neural Collapse | ★★★★ | ★★★★ | ★★★★ | Medium |

**Decision**: These were considered too theoretical and math-focused for the team. The user explicitly asked for topics more on the "application side of things" rather than exploring unsolved mathematical problems.

## Round 2: Application-Oriented Topics

Seven new topics were proposed, all with practical real-world angles:

| # | Topic | Trending | Portfolio | Course Fit | Practical | Difficulty |
|---|-------|----------|-----------|------------|-----------|------------|
| 1 | Network Pruning / Lottery Ticket | ★★★★ | ★★★★ | ★★★★ | ★★★★★ | Low-Med |
| 2 | Transfer Learning Efficiency | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | Medium |
| 3 | Data Augmentation (Why It Works) | ★★★ | ★★★ | ★★★★ | ★★★★★ | Low-Med |
| 4 | ViT vs CNN Properties | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | Medium |
| 5 | Adversarial Robustness | ★★★★ | ★★★★★ | ★★★ | ★★★★★ | Medium |
| 6 | LoRA/QLoRA Properties | ★★★★★ | ★★★★★ | ★★★★ | ★★★★★ | Medium |
| 7 | Distribution Shift | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | Low-Med |

**Key insight from this round**: LoRA (#6) stood out because it directly connects to course material (matrix perturbation, over-parametrization) while being the most practically trending technique.

## Round 3: The Breakthrough — Real-World Experience

The user shared a personal experience:

> "I was using gemma-4-e4b on my phone... when I was asking it how to do emergency care on people who have things stuck to their throat, gemma-4-e4b told me to seek help and refused to instruct me on how to actually do it."

This led to exploring the intersection of:
1. **Over-refusal** — when safety guardrails cause harm
2. **Abliteration** — the technique used to "crack" open-source models
3. **Mechanistic interpretability** — understanding WHERE safety lives in the weights

The user also shared cracked model examples:
- `dealignai/Gemma-4-31B-JANG_4M-CRACK` — uses "CRACK v2" abliteration
- `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive` — uses "abliterated weights"

**Why this was the breakthrough**: It combined all preferences:
- Real-world motivation (emergency over-refusal = life and death)
- State-of-the-art (AI safety and mechanistic interpretability are peak trending)
- Experimental (clear experiments: activation extraction, ablation sweeps, weight diffs)
- Application-oriented (directly useful findings for model deployment)
- Strong course connection (over-parametrization, matrix perturbation, NTK linearization)
- Portfolio gold ("I reverse-engineered how LLM safety works at the weight level")

## Model Selection Analysis

### Gemma 4 E4B vs E2B

The primary mechanistic analysis model was chosen after analyzing the architecture:

**Gemma 4 E4B-it** (from `config.json`):
- 42 layers (35 sliding + 7 global attention)
- hidden_size = 2560
- NOT MoE — dense decoder with Per-Layer Embeddings (PLE)
- BF16: 15GB (too tight), 8-bit: 7.5GB (comfortable on 16GB GPU)

**Decision**: E4B at 8-bit quantization. Rationale:
- Better safety training than E2B → more interesting refusal behavior
- 8-bit quantization noise averages out when computing mean activations over 200+ prompts
- 7.5GB weights + 8.5GB headroom = comfortable fit
- E2B at BF16 serves as validation model

### Why Not MoE for Mechanistic Analysis?

Qwen3.5-35B-A3B (MoE, 256 experts) is too large for activation extraction on the GPU but perfect for CPU-based weight diff analysis on 100GB RAM. This naturally split the work: GPU for Gemma 4 mechanistic work, CPU for Qwen weight forensics.

## The Abliteration Technique — What We Learned

Investigation of the cracked models revealed that both use **abliteration**:

1. Collect activation pairs (refuse vs comply prompts)
2. Compute the "refusal direction" = mean(refuse_acts) - mean(comply_acts) 
3. Project this direction out of weight matrices: `W_new = W - d * d^T * W`

This is a **rank-1 perturbation** — modifying a single direction in a 2560-dimensional space removes all safety behavior. This means:
- Safety is encoded as approximately **one direction** in activation space
- Despite millions of RLHF training examples, the net effect is ~1-dimensional
- This connects to **over-parametrization theory**: in a massively overparameterized network, even complex behaviors can occupy tiny subspaces

## Final Topic: "The Geometry of Alignment"

**Full title**: *The Geometry of Alignment: How Safety Lives (and Dies) in Modern LLMs*

### Four parallel workstreams:
1. **Benchmark & Evaluation** (Person A): 300-400 prompt benchmark, cross-model refusal rate analysis
2. **Mechanistic Analysis** (Person B): Layer-by-layer activation probing, refusal direction computation, PCA rank analysis, UMAP visualization
3. **Abliteration & Selective Safety** (Person C): Reproduce abliteration, ablation sweeps, category-specific selective de-alignment
4. **Weight Diff Forensics** (Person D): SVD analysis of cracked vs original weights, MoE expert modification mapping, literature survey

### What makes this topic exceptional:
- **Novel angles**: MoE expert analysis, PLE investigation, sliding vs global attention comparison — barely explored in existing literature
- **Constructive contribution**: Selective safety (removing over-refusal while keeping genuine safety) is not just analysis but a proposed improvement
- **Compelling narrative**: Opens with a life-or-death scenario, progresses through technical investigation, concludes with a practical solution
- **Feasible in 1 month**: Clear work packages, fully parallelizable, fits available hardware
- **Professor approved**: Topic was presented and approved

## Key References for the Project

- Arditi et al. (2024) — "Refusal in LLMs Is Mediated by a Single Direction"
- Zou et al. (2023) — "Representation Engineering"
- Qi et al. (2023) — Fine-tuning attacks on alignment
- Christiano et al. (2017), Ouyang et al. (2022) — RLHF
- Rafailov et al. (2023) — DPO
- Rottger et al. (2024) — XSTest over-refusal benchmark
- Jacot et al. (2018) — Neural Tangent Kernel (course paper)
- Chizat and Bach (2018) — Lazy Training (course paper)

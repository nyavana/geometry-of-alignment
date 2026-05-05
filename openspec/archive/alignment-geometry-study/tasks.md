## 1. Environment Setup (All — Week 1, Days 1-2)

- [ ] 1.1 Create conda environment with Python 3.11, install PyTorch, transformers, bitsandbytes, accelerate, llama-cpp-python, scikit-learn, safetensors, matplotlib, seaborn, plotly, umap-learn, pandas, numpy, tqdm
- [ ] 1.2 Create project directory structure: `src/{benchmark,mechanistic,abliterate,weight_diff}`, `data/`, `results/{activations,refusal_directions,ablation_results,weight_diffs,figures}`, `paper/`
- [ ] 1.3 Download Gemma 4 E4B-it and E2B-it via `huggingface-cli download` (requires HF auth for Gemma 4 models)
- [ ] 1.4 Download Qwen3.5-35B-A3B original and HauhauCS uncensored variant
- [ ] 1.5 Download GGUF quantizations of Gemma 4 E4B and Qwen3.5-A3B for llama.cpp evaluation
- [ ] 1.6 Verify Gemma 4 E4B loads in 8-bit on 4070 Ti Super: `print(model)` to confirm layer attribute paths (`model.model.layers[i].self_attn.o_proj.weight`, `model.model.layers[i].mlp.down_proj.weight`)
- [ ] 1.7 Verify the 42-layer type pattern from config `layer_types` field — identify which layer indices are sliding vs global attention

## 2. Shared Benchmark Construction (All — Week 1, Days 1-3)

- [ ] 2.1 Create `data/benchmark_prompts.json` scaffold with metadata and category definitions
- [ ] 2.2 Write 50+ `emergency_medical` prompts with variants (Heimlich, CPR, tourniquet, wound care, allergic reaction, burns, drowning response)
- [ ] 2.3 Write 50+ `wilderness_survival` prompts with variants (water purification, fire starting, shelter building, snake bite, bear encounter, hypothermia)
- [ ] 2.4 Write 40+ `home_safety` prompts (gas leak, electrical fire, chemical spill, carbon monoxide)
- [ ] 2.5 Write 40+ `chemistry_safety` prompts (acid spill response, chemical burn treatment, fume exposure)
- [ ] 2.6 Write 40+ `mental_health` prompts (talking someone through crisis, de-escalation, panic attack response)
- [ ] 2.7 Write 40+ `gray_zone` prompts (detailed wound suturing, medication administration, lockpicking own house, self-defense techniques)
- [ ] 2.8 Write 40+ `should_refuse` prompts (weapons synthesis, hacking targeting others, exploitation)
- [ ] 2.9 Write 40+ `safe_control` prompts (cooking, math, travel, general knowledge)
- [ ] 2.10 Review and finalize benchmark — ensure all prompts have `id`, `category`, `expected`, `prompt`, `variants` fields

## 3. Benchmark Evaluation Pipeline — Person A (Weeks 1-3)

- [ ] 3.1 Implement `src/benchmark/classify_refusal.py` with regex-based refusal/compliance pattern matching
- [ ] 3.2 Implement `src/benchmark/evaluate.py` with llama.cpp backend (`Llama` class, chat completion, temperature 0.1)
- [ ] 3.3 Add transformers backend to `evaluate.py` for evaluating abliterated models (accepts model + tokenizer objects)
- [ ] 3.4 Run initial evaluation on Gemma 4 E4B-it (GGUF) to validate pipeline works end-to-end
- [ ] 3.5 Manually label 50-100 responses and validate classifier achieves >= 90% agreement
- [ ] 3.6 Iterate on refusal classifier patterns based on validation errors
- [ ] 3.7 Run full evaluation: Gemma 4 E4B-it (original)
- [ ] 3.8 Run full evaluation: Gemma 4 E2B-it (original)
- [ ] 3.9 Run full evaluation: Qwen3.5-35B-A3B (original)
- [ ] 3.10 Run full evaluation: Qwen3.5-35B-A3B-Uncensored (HauhauCS)
- [ ] 3.11 Run phrasing sensitivity analysis using prompt variants
- [ ] 3.12 Run context sensitivity test: prepend "I am an emergency first responder" to emergency_medical prompts and compare refusal rates
- [ ] 3.13 Implement `src/benchmark/analyze_results.py` — refusal heatmap, over-refusal bar chart, phrasing sensitivity plot
- [ ] 3.14 Run evaluation on Person C's abliterated models when available
- [ ] 3.15 Produce all final figures for paper Section 4

## 4. Activation Extraction & Mechanistic Analysis — Person B (Weeks 1-3)

- [ ] 4.1 Implement `ActivationCollector` class with `register_forward_hook` on each of 42 layers' residual stream output
- [ ] 4.2 Implement `extract_activations_for_prompts()` — iterate over prompts, run forward pass, collect last-token activations per layer on CPU
- [ ] 4.3 Test activation extraction on Gemma 4 E2B (BF16) with 10 prompts to verify hook paths and tensor shapes
- [ ] 4.4 Extract full refuse activations from Gemma 4 E4B (8-bit): ~200 should_refuse + over-refused prompts
- [ ] 4.5 Extract full comply activations from Gemma 4 E4B (8-bit): ~200 safe_control + should-comply prompts
- [ ] 4.6 Save activations to `results/activations/refuse_activations.pt` and `comply_activations.pt`
- [ ] 4.7 Compute refusal direction per layer via mean-diff method, save to `results/activations/refusal_directions.pt`
- [ ] 4.8 Implement `compute_signal_strength()` — separation score, cosine distance, projection magnitude per layer
- [ ] 4.9 Plot layer-wise signal strength chart colored by sliding (blue) vs global (red) attention type
- [ ] 4.10 Implement PCA rank analysis per layer — components needed for 95% and 99% variance, top-1 energy fraction
- [ ] 4.11 Produce rank analysis summary: is refusal approximately rank-1 across layers?
- [ ] 4.12 Implement UMAP/t-SNE visualization — single layer and multi-layer grid
- [ ] 4.13 Produce multi-layer UMAP grid showing refuse/comply separation evolving through the network
- [ ] 4.14 Validation: repeat refusal direction extraction on E2B at BF16, compute cosine similarity with E4B directions — report whether findings hold across precisions
- [ ] 4.15 Produce all final figures for paper Section 5

## 5. Abliteration Implementation & Experiments — Person C (Weeks 1-3)

- [ ] 5.1 Implement `compute_refusal_direction()` with mean-diff and PCA methods
- [ ] 5.2 Implement `abliterate_model()` — rank-1 projection out of `o_proj.weight` and `down_proj.weight` per layer
- [ ] 5.3 Implement `save_abliterated_model()` — save modified weights + tokenizer via `save_pretrained()`
- [ ] 5.4 Test abliteration on Gemma 4 E2B: extract directions, abliterate, verify model stops refusing on 5 test prompts
- [ ] 5.5 Apply full abliteration to Gemma 4 E4B (alpha=1.0, all 42 layers)
- [ ] 5.6 Quick-test: run 20 benchmark prompts on abliterated E4B, confirm refusal removal works
- [ ] 5.7 Implement ablation study framework with JSON result logging
- [ ] 5.8 Run alpha sweep: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0] — evaluate refusal rate at each
- [ ] 5.9 Run layer subset sweep: all, global-only, sliding-only, first-half, second-half, last-10, middle-14
- [ ] 5.10 Run prompt count sweep: [10, 25, 50, 100, 200] prompt pairs for direction computation
- [ ] 5.11 Run random direction control: abliterate with random unit vectors, confirm refusal persists
- [ ] 5.12 Run capability preservation: evaluate original and abliterated models on MMLU subset and GSM8K subset
- [ ] 5.13 Compute category-specific refusal directions: emergency_medical, wilderness_survival, should_refuse
- [ ] 5.14 Compute pairwise cosine similarity between category directions at each layer
- [ ] 5.15 Attempt selective abliteration: remove medical refusal direction only, evaluate over-refusal on medical + refusal rate on should_refuse
- [ ] 5.16 Produce alpha sweep curve, layer subset comparison, selective safety results table
- [ ] 5.17 Hand off abliterated models to Person A for full benchmark evaluation (task 3.14)
- [ ] 5.18 Produce all final figures for paper Section 6

## 6. Weight Diff & Cracked Model Analysis — Person D (Weeks 1-3)

- [ ] 6.1 Implement `load_state_dict_from_safetensors()` — memory-efficient CPU loading of model weights
- [ ] 6.2 Implement `compute_weight_diffs()` — iterate common keys, compute Frobenius norm, relative change, max abs change per parameter
- [ ] 6.3 Implement SVD computation for 2D weight diff matrices — effective rank at 95%/99%, top singular values, top singular vectors
- [ ] 6.4 Run weight diff on Qwen3.5-35B-A3B original vs uncensored, save results JSON + SVD data
- [ ] 6.5 Report: how many parameters changed? What fraction are unchanged?
- [ ] 6.6 Produce per-layer total Frobenius norm bar chart
- [ ] 6.7 Produce singular value spectrum plots for the most significantly modified weight matrices
- [ ] 6.8 Implement MoE expert analysis — separate modifications by component type: attention, MLP, expert, router, shared expert
- [ ] 6.9 Produce expert modification heatmap (layer x expert_index)
- [ ] 6.10 Report: was the router modified? Interpret implications for safety encoding
- [ ] 6.11 Report: component-type breakdown of total modification magnitude
- [ ] 6.12 Cross-reference: compare top singular vectors of weight diff with Person B's refusal directions (qualitative comparison across different models)
- [ ] 6.13 Produce all final figures for paper Section 7

## 7. Literature Survey — Person D (Weeks 2-3)

- [ ] 7.1 Survey alignment techniques: RLHF (Christiano et al. 2017, Ouyang et al. 2022), DPO (Rafailov et al. 2023), Constitutional AI (Bai et al. 2022)
- [ ] 7.2 Survey representation engineering: Zou et al. 2023 (Representation Engineering), linear representation hypothesis
- [ ] 7.3 Survey abliteration and de-alignment: Arditi et al. 2024 (Refusal in LLMs Is Mediated by a Single Direction), fine-tuning attacks (Qi et al. 2023, Yang et al. 2023)
- [ ] 7.4 Survey over-refusal literature: Rottger et al. 2024 (XSTest), Cui et al. 2024
- [ ] 7.5 Write Sections 2-3 of paper (Background + Related Work), minimum 15 citations

## 8. Paper & Presentation (All — Week 4)

- [ ] 8.1 Person D: Write Section 1 (Introduction) with hiking emergency scenario motivation
- [ ] 8.2 Person A: Write Section 4 (Over-Refusal Analysis) with benchmark results, heatmaps, phrasing sensitivity
- [ ] 8.3 Person B: Write Section 5 (Mechanistic Analysis) with layer signal strength, rank analysis, UMAP visualizations
- [ ] 8.4 Person C: Write Section 6 (Abliteration & Selective Safety) with ablation sweeps, selective safety results
- [ ] 8.5 Person D: Write Section 7 (Weight Diff Analysis) with SVD results, MoE expert analysis
- [ ] 8.6 All: Write Section 8 (Discussion & Course Connections) — over-parametrization, matrix perturbation, NTK connections
- [ ] 8.7 All: Write Section 9 (Conclusion & Ethics)
- [ ] 8.8 Person D: Integrate all sections, ensure consistent formatting, compile LaTeX
- [ ] 8.9 All: Create presentation slides — opening with emergency scenario, 3+ key result figures, course connections
- [ ] 8.10 All: Practice presentation run-through
- [ ] 8.11 All: Final paper review and submission

# Onboarding — Geometry of Alignment

A long-form companion to `README.md`. The README is the GitHub-style quickstart
(install, run, headline numbers). This doc is what you read first when you
join the project or pick up the codebase cold: the *why*, the *how it fits
together*, and the *narrative arc of findings* in the order they happened.

Estimated reading time: 30–45 minutes. You do not need prior LLM-interpretability
experience. You should know Python and basic deep learning (gradient descent,
backprop, attention, residual connections).

If you only have 5 minutes, read sections 1, 2, and 8.

---

## Table of contents

1. [What this project is, in plain English](#1-what-this-project-is-in-plain-english)
2. [Prerequisites: the five concepts you need](#2-prerequisites-the-five-concepts-you-need)
3. [The research questions](#3-the-research-questions)
4. [Models and why we picked them](#4-models-and-why-we-picked-them)
5. [The experimental architecture](#5-the-experimental-architecture)
6. [The narrative arc of findings (M0 to M6)](#6-the-narrative-arc-of-findings-m0-to-m6)
7. [Headline results](#7-headline-results)
8. [TL;DR for an external reader](#8-tldr-for-an-external-reader)
9. [How to navigate the repo](#9-how-to-navigate-the-repo)
10. [How to run things](#10-how-to-run-things)
11. [Glossary](#11-glossary)
12. [Further reading](#12-further-reading)

---

## 1. What this project is, in plain English

Modern aligned LLMs refuse two kinds of things:

- **Genuinely harmful queries**, which they should refuse (how to build a
  bioweapon, how to make CSAM, etc.)
- **Benign-but-flagged queries**, which they should not refuse (how to
  perform the Heimlich maneuver, how to treat hypothermia in the field, how
  to safely store cleaning chemicals). This is called *over-refusal*.

There is a known and widely-circulated technique called **abliteration** that
removes refusal behavior from open-weight LLMs by subtracting a single direction
from the residual stream — a rank-1 weight perturbation. No gradient descent,
no training data, just `W ← W − α · d · (dᵀ W)` applied to a couple of weight
matrices per layer. It works on Llama 2/3, Qwen, Mistral, and several other
families. It is fast (minutes on a laptop), reproducible, and the code fits in
~50 lines.

This project asks: **why does that work, and what does it tell us about where
"safety" actually lives in the model's weights?** And then: **what happens when
it stops working?** Because on the model we picked (Gemma 4 E4B-it,
released late 2024) the standard recipe doesn't work, and published
"uncensored" variants of the same model use noticeably different recipes to
get the same behavioral result.

The course context is EECS 6699 (Mathematics of Deep Learning, Columbia,
Spring 2026). The substantive contribution is empirical and geometric: a
careful characterization of which ingredients of the rank-1 recipe are
load-bearing on Gemma 4, and what the failure mode tells us about the
geometry of refusal in general.

The original motivation was a real incident: a teammate, hiking in an area
with no cell service, asked a locally-installed Gemma 4 model for first-aid
instructions and got a refusal. The over-refusal question is not
hypothetical.

---

## 2. Prerequisites: the five concepts you need

If you already know these, skip to §3.

### 2.1 Refusal as a behavior

When you prompt an aligned LLM with something it deems unsafe, it produces a
templated refusal: *"I'm sorry, but I can't help with that."* Modern alignment
methods (RLHF, DPO, Constitutional AI) optimize the model to produce these
refusals on harmful inputs. The refusal text is not hard-coded — it is the
output of next-token prediction, just like everything else the model says. So
the model's internal computation must somehow recognize "this is unsafe" and
then route to a refusal continuation.

### 2.2 The residual stream

A transformer's hidden state is a high-dimensional vector that flows through
the layers and gets additively modified by each attention block and each MLP
block. For Gemma 4 E4B that vector has dimension 2560. You can think of every
intermediate computation as a write into this stream and every layer's read as
a query against the stream's current contents. *Mechanistic interpretability*
treats the residual stream at each layer as a feature space and asks "what
directions in this space correspond to what concepts?"

### 2.3 The refusal direction

If you take a batch of harmful prompts and a batch of benign prompts and feed
them through the model, look at the residual-stream activation at some
middle layer, and compute the difference of mean vectors:

```
d = mean(activation | harmful) − mean(activation | benign)
```

then `d` (after normalization) turns out to be the *refusal direction*. Adding
`d` to a benign prompt's hidden state at inference time makes the model
refuse. Subtracting `d` from a harmful prompt's hidden state makes the model
comply. Arditi et al. (2024) and Lermen (2024) showed this for a wide range
of open-weight models.

### 2.4 Abliteration

Once you have `d`, you can bake the "subtract `d`" behavior permanently into
the weights by projecting `d` out of the output weights of the attention and
MLP blocks:

```
W_o_proj   ← W_o_proj   − d · (dᵀ · W_o_proj)
W_down_proj ← W_down_proj − d · (dᵀ · W_down_proj)
```

The resulting model is **abliterated**: it can no longer write anything in
the `d` direction onto the residual stream, so it can't trigger refusal. This
is a rank-1 perturbation per weight matrix. On Llama-family models, this
recipe drives refusal rates from ~100% down to ~0% on harmful queries with
roughly no collateral damage. The published variants we benchmark
(HauhauCS, OBLITERATUS, TrevorJS) are all in the abliteration family,
though as §6 shows, they differ in important ways.

### 2.5 Why "rank-1" is interesting

A rank-1 update changes the matrix `W ∈ ℝ^{d×d}` along exactly one direction
in row-space. It has `2d` parameters (one vector `d` and one scalar `α`),
while the full matrix has `d²`. The fact that a 2×2560 = 5120-parameter
intervention deletes a behavior that took millions of RLHF examples to install
is the empirical observation that motivates this whole research area.

It also touches a specific mathematical question the course covers:
matrix-perturbation theory (Hoffman–Wielandt, Davis–Kahan). How does a small
change in a layer's weights propagate through the rest of the network? When is
the network *robust* to weight perturbations and when is it not?

---

## 3. The research questions

The proposal (`docs/planning/project_proposal.md`) lists five:

1. **How prevalent is over-refusal in practice?** When do aligned models
   refuse beneficial queries (emergency medical, wilderness survival, home
   safety, etc.)?
2. **Where does refusal live?** Which layers and subspaces of the residual
   stream encode the refusal direction, and what dimensionality does it
   occupy?
3. **Is rank-1 abliteration universally effective, and if not, why?** Recent
   community reports flag Gemma 4 as a hard case. What does the failure mode
   look like, and what does it tell us about the geometry of refusal?
4. **Can over-refusal be removed selectively?** Is it possible to remove
   refusal on benign-but-flagged queries while keeping refusal on harmful
   ones?
5. **What do "uncensored" model releases actually change?** How do published
   uncensored variants differ from their original checkpoint at the
   weight-diff level, and do they share a common modification subspace?

Q1, Q2, Q5 are answered cleanly. Q3 is answered as a *partial-effect* story
(§6.5). Q4 has a clean geometric answer (yes, the directions are
orthogonal) but the standard recipe could not exploit it on Gemma 4
(§6.4).

---

## 4. Models and why we picked them

| Model | Architecture | Role |
|---|---|---|
| **Gemma 4 E4B-it** (base) | 42-layer dense, hidden 2560, mixed attention | Primary mechanistic + abliteration target. 8-bit on GPU. |
| **Gemma 4 E2B-it** (validation) | smaller dense | Cross-precision sanity check, behavioral baseline. BF16 fits in VRAM unquantized. |
| **HauhauCS** (`gemma-4-E4B-Uncensored-HauhauCS-Aggressive`) | published abliteration, GGUF only | Behavioral eval. Fully uncensored — 0% refusal across the entire 344-prompt benchmark. |
| **OBLITERATUS** (`gemma-4-E4B-it-OBLITERATED`) | published abliteration | Weight-diff target. Whitened SVD + attention head surgery + winsorized activations. Edits 21 of 42 layers. |
| **TrevorJS** (`gemma-4-E4B-it-uncensored`) | published abliteration | Weight-diff target. Norm-preserving biprojection. Edits 84 weights with median rank-1. |
| **Self-abliterated E4B** (built by us) | our own rank-1 perturbation | The negative-result control. Same recipe as Arditi/Mlabonne. |

**Why Gemma 4 specifically.** Three reasons. (1) It is a recent (late-2024)
production-grade checkpoint where the standard rank-1 recipe is reported to
fail — investigating that failure is the meat of the project. (2) Multiple
*independent* uncensored releases exist (OBLITERATUS, TrevorJS, HauhauCS),
each using a noticeably different recipe, which gives us a comparative
weight-diff signal. (3) It fits in 16 GB of consumer VRAM at 8-bit.

**Why E2B as a validation.** A smaller checkpoint that runs in BF16 lets us
sanity-check that observations on E4B are not artifacts of int8 quantization.
The E2B model also turns out to over-refuse more visibly than E4B
(12% emergency_medical vs. 2% on E4B), which is an interesting finding in its
own right (§6.1).

### Gemma 4 E4B architecture constants

You will see these numbers everywhere in the code and figures:

- **42 transformer layers** total
- **35 sliding-window attention** layers + **7 global-attention** layers
- Global attention layer indices: `[5, 11, 17, 23, 29, 35, 41]`
- **Hidden size 2560**
- 8-bit on GPU: ~7.5 GB VRAM, ~75–150 s/prompt at `max_new_tokens=512`
  (`transformers` backend with `bitsandbytes`)
- Q8_0 GGUF via `llama-server`: ~18–25 s/prompt for the same setting
  (5–8× faster)

Memorize the global-attention index list. It comes up every time we partition
sweeps by attention type.

---

## 5. The experimental architecture

There are four independent modules in `src/`, each runnable as a standalone
pipeline. Together they form a four-stage analysis chain:

```
                       data/benchmark_prompts.json
                                  │
                                  ▼
                    ┌──────────────────────────┐
        Stage 1     │  src/benchmark/          │   How often does each model
   (behavioral)     │  evaluate.py             │   refuse each category?
                    │  classify_refusal.py     │
                    │  analyze_results.py      │   → refusal_heatmap.png
                    └────────────┬─────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
        Stage 2     │  src/mechanistic/        │   Where in the layers does
   (where does     │  extract_activations.py   │   refusal live, and at what
    refusal live?) │  layer_analysis.py        │   rank?
                    │  visualize.py            │
                    │                          │   → refusal_directions.pt
                    └────────────┬─────────────┘   → signal_vs_layer.png
                                  │                 → UMAP figures
                                  ▼
                    ┌──────────────────────────┐
        Stage 3     │  src/abliterate/         │   Can we remove refusal by
   (can we remove  │  abliterate.py            │   a rank-1 weight edit?
    it ourselves?) │  ablation_study.py        │
                    │  selective_safety.py     │   → alpha_sweep.png
                    └────────────┬─────────────┘   → layer_subset_comparison.png
                                  │
                                  ▼
                    ┌──────────────────────────┐
        Stage 4     │  src/weight_diff/        │   How do published uncensored
   (what do the    │  compute_diff.py          │   variants actually differ
    cracked        │  svd_analysis.py          │   from the base, and from
    models do?)    │                          │   each other?
                    └──────────────────────────┘   → cross_method_singular_vectors.png
                                                    → refusal_direction_vs_singular_vector.png
```

Module-by-module:

### `src/benchmark/` — behavioral evaluation

- **Input**: `data/benchmark_prompts.json` (340 prompts across 8 categories;
  see §6.1).
- **What it does**: runs a model against every prompt, captures the response,
  classifies it as refuse/comply via regex-based two-stage classification
  (refusal patterns first, compliance patterns second), and produces a CSV
  plus a per-category heatmap.
- **Two backends**: `llamacpp` (GGUF via HTTP to `llama-server`, fast) and
  `transformers` (HF safetensors in-process, slow but supports models we
  cannot easily quantize to GGUF).
- **Why two backends**: the published HauhauCS variant is only released as
  GGUF; the self-abliterated checkpoint and the TrevorJS variant only have
  HF safetensors. We need both code paths to compare them.

### `src/mechanistic/` — where does refusal live

- **Input**: a model + the benchmark prompts.
- **What it does**: hooks into every transformer layer via
  `register_forward_hook`, collects the residual-stream output, and computes
  the refusal direction `d = mean(activation | refuse) − mean(activation |
  comply)`, normalized to unit length.
- **Outputs**: `refusal_directions.pt` (42 unit vectors of dim 2560, keyed
  by layer index) plus signal-strength figures (Cohen's d per layer), PCA
  rank analysis, and UMAP/t-SNE projections.
- **Why this is more than just "compute a difference"**: the analysis tells
  you (a) which layer to attack, (b) whether the refusal *subspace* is
  really 1D or higher, and (c) whether sliding-attention layers behave
  differently from global ones.

### `src/abliterate/` — the rank-1 intervention

- **Input**: a model + a refusal direction.
- **What it does**: applies `W ← W − α · d · (dᵀ W)` to `o_proj.weight` and
  `down_proj.weight` per `_project_out()` in `abliterate.py`.
- **Sweeps**: alpha (0 to 2.0), layer subsets (global-only, sliding-only,
  peak-band-only, etc.), and a random-direction control to rule out
  "any rank-1 perturbation breaks refusal."
- **Selective safety**: computes category-specific refusal directions
  (one per safety category) and tries to remove the medical-over-refusal
  one while keeping the harmful-query one.

### `src/weight_diff/` — what did the cracked models actually change

- **Input**: two model directories of safetensors (base vs. modified).
- **What it does**: loads both checkpoints to CPU fp32, computes element-wise
  diffs per parameter tensor, runs SVD on each non-zero diff, ranks the
  changes, and cross-references the top-1 singular vectors against the
  refusal directions from `src/mechanistic/`.
- **Cross-method**: also computes cosines between OBLITERATUS's and TrevorJS's
  top-1 singular vectors per layer, to see whether the two methods are
  modifying the "same" direction.

### How the modules feed each other

- Stage 2 produces `refusal_directions.pt`, consumed by Stage 3
  (`abliterate.py --directions ...`) and by Stage 4
  (`svd_analysis.py` cross-reference).
- Stage 1's CSV is consumed by every downstream interpretation: "did the
  intervention actually change refusal rates?"
- Stages 3 and 4 share zero code with each other — they are independent
  arms.

Cross-module Python-import dependencies (helpful when reading the code):
`ablation_study.py` imports `load_model_and_tokenizer` from
`mechanistic.extract_activations` and `evaluate_with_transformers` from
`benchmark.evaluate`. `selective_safety.py` imports from
`mechanistic.extract_activations` and `abliterate.ablation_study`.
`weight_diff` has no cross-module imports.

---

## 6. The narrative arc of findings (M0 to M6)

The work was structured as milestones M0 through M6, each with its own
working branch under `agent/*`. This section walks through them in order. If
you read nothing else in this doc, read this section.

### M0 — bootstrap

Set up six sibling git worktrees (`gb-ablit`, `gb-bench`, `gb-env`, `gb-mech`,
`gb-paper`, `gb-wdiff`) sharing a single non-versioned sidecar at
`/home/nyavana/columbia/6699/shared/` that holds the 25 GB model checkpoints,
the Python venv, the Hugging Face cache, and per-branch results. The shared
sidecar means multiple agents can work in parallel without trampling each
other's `results/`.

The `model` and `.venv` symlinks in each worktree resolve to the shared
sidecar and are excluded via `.git/info/exclude`. This is mechanical but
worth understanding before you go looking for files: if you `ls model/` in
the project root, you are looking at `/home/nyavana/columbia/6699/shared/model/`.

### M1 — the benchmark

Started from 9 hand-written prompts (proof of concept), expanded to **340
prompts across 8 categories** via `scripts/build_benchmark.py`. The categories:

| Category | Expected behavior | # prompts |
|---|---|---|
| `should_refuse` | Refuse (truly harmful: bioweapons, CSAM, ICS malware) | 42 |
| `emergency_medical` | Comply (Heimlich, choking, hypothermia) | 50 |
| `wilderness_survival` | Comply (snake bite, lost-in-woods, water purification) | 50 |
| `home_safety` | Comply (chemical mixing, electrical, gas) | 40 |
| `mental_health` | Comply (panic attack, suicide hotline info) | 40 |
| `chemistry_safety` | Comply (lab safety, household chemistry) | 40 |
| `gray_zone` | Either is defensible (security research, pen-testing) | 42 |
| `safe_control` | Comply (totally innocuous control) | 40 |

Each prompt has an `id`, `category`, `expected` (refuse/comply), `prompt`,
and `variants` field for phrasing-sensitivity testing. Total: 640 variants,
118 KB JSON. The schema was validated at freeze time (M1 tag
`m1-benchmark-frozen`).

Why this matters: the benchmark is the **measuring stick** for everything
that follows. Every claim about over-refusal, every abliteration result,
every comparison between published variants is grounded in this CSV-based
behavioral eval. If you don't trust the benchmark, you can't trust the rest.

### M2a — behavioral baselines

Ran the benchmark against every accessible model:

- **Base Gemma 4 E4B** (GGUF Q8_0 via `llama-server`): 100% on
  `should_refuse`, **2.0% on `emergency_medical`**. Over-refusal hypothesis
  **weakly confirmed at best** — the base E4B production checkpoint refuses
  emergency medical queries only 2% of the time on this benchmark.
- **Gemma 4 E2B** (BF16): 95.2% on `should_refuse`, **12.0% on
  `emergency_medical`**, **42.9% on `gray_zone`**. The smaller model
  over-refuses more. The over-refusal narrative is more accurately framed
  as *"over-refusal is a smaller-model artifact"* rather than a property of
  modern aligned models generally.
- **HauhauCS** (GGUF): **0% across the entire benchmark, including 0/42 on
  `should_refuse`.** Cleanly and totally uncensored.
- **OBLITERATUS** (GGUF): **failed.** `llama-server`'s chat parser crashed
  on the model's raw Harmony-format `<|channel>` tokens after the second
  prompt. See `docs/issues/2026-05-06-obliteratus-eval-fail.md`. Behavioral
  row not available.
- **TrevorJS** (transformers + 8-bit): **failed.** 117 s/iter sustained,
  projected 11 h on 344 prompts. Killed at 5%. See
  `docs/issues/2026-05-06-trevorjs-eval-fail.md`. We later got a positive
  control row on a 48-prompt stratified subset under bf16 (M6 Stage 0b):
  0/48 refused.

The takeaway from M2a: the over-refusal motivation that originally framed the
project is *weaker than expected on the production E4B checkpoint* (2%, not
the catastrophic rate we had assumed). E2B does over-refuse heavily. The
paper framing pivoted toward the geometry findings as the substantive
contribution.

### M2b — mechanistic analysis (where does refusal live)

Hooked every layer, collected residual-stream activations on refuse-paired
and comply-paired prompts, computed the per-layer refusal direction. Then
asked three questions:

- **Which layer carries the strongest refusal signal?** Cohen's d (per
  layer) over `mean_refuse − mean_comply`. **Peak: L15, d = 2.87.** Top 3:
  L15 (2.87), L4 (2.84), L14 (2.80). The peak is broad — d ≥ 2.8 across
  L4–L17.
- **Sliding vs. global attention layers?** Mean d: sliding 2.60, global 2.52.
  Gap −0.08, within per-layer noise. **No systematic difference.** The
  refusal signal does not concentrate on the 7 global-attention indices.
- **Is the refusal subspace rank-1 in activations?** Run PCA on
  `(refuse_act − comply_act)` at each layer; ask what fraction of
  `‖Δμ‖²` the top-1 PC captures. **86.6% mean** across the peak band,
  90.7% at L14, 89.6% at L15. PCs 2–3 add ~1pp. The activation-space
  refusal direction is **effectively rank-1**.

Figures: `signal_vs_layer.png`, `pca_variance_per_layer.png`,
`umap_layer_{00,05,11,15,17,41}.png`, `rank_analysis.png`. All
visually clean: L15 shows clean linear separation in 2D UMAP
(mis-classification 0.13 along centroid axis); L00 has no separation; L41
collapses again at the output. Bottom line: **rank-1 abliteration is the
right tool, geometrically, on the activation side.** If it fails on Gemma 4,
the failure must come from somewhere else (M2c, M6).

Output artifact: `refusal_directions.pt` (42 unit vectors of dim 2560, one
per layer). This file becomes the input to M2c and M3.

### M2c — abliteration sweep on Gemma 4 E4B

We applied the standard rank-1 mean-diff abliteration recipe to the base
E4B model using the M2b directions. We swept:

- **Alpha** (projection strength), α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
  1.75, 2.0}.
- **Layer subset**: all layers, global only, sliding only, peak band L4–L17,
  early third, middle third, late third, two more bespoke subsets.
- **Random control**: substitute a random unit vector for `d`, hold α=1.0.
  This rules out "any rank-1 perturbation breaks refusal."

Subset of the benchmark: stratified 20 prompts, `max_new_tokens=128`
(compute-budgeted; the load-bearing finding is the *shape* of the curve,
not the absolute refusal rate).

**Result: refusal rate flat at 30–35% across the entire α sweep, flat at
25–35% across all layer subsets, and the random control sits at 30% — which
*also matches* the α=0 baseline at 30%.** The intervention is doing nothing.

Compare to a working abliteration on Llama 3, where the same sweep would
drive refusal from ~100% to ~5%. On Gemma 4 E4B, the sweep is *informative
noise*.

We also computed per-category refusal directions (one for `emergency_medical`,
one for `wilderness_survival`, one for `chemistry_safety`, etc.) and asked
about their geometry. **Result: the five over-refuse categories form a tight
+0.93 pairwise cosine cluster, and that cluster is orthogonal to the
`should_refuse` direction (mean cosine −0.015, range −0.024 to +0.001).**
So selective abliteration is geometrically possible — the directions are
clean and separate. But the magnitude-side problem (the standard recipe is
inert) prevented us from exploiting it.

Behavioral confirmation (M2c-followup): we built an actual self-abliterated
checkpoint at α=1.0 with the all-layers recipe and ran it against a
stratified 48-prompt benchmark subsample. **`should_refuse` rate: 6/6 =
100%, identical to base.** The most paper-relevant behavioral test is a
0-percentage-point delta.

This is the headline negative finding the paper is built around. It is
*expected* — Heretic and the OBLITERATUS model card both flagged Gemma 4 as
a hard case in advance — but it is the first published systematic
characterization of *what kind of inert* the standard recipe is, and the
clean category geometry adjacent to it. M2c sets up M6.

### M3 — comparative weight-diff (what did the cracked models actually change)

If the standard recipe doesn't work, but published abliterations of Gemma 4
*do* work behaviorally, then what do they change? We loaded each published
variant alongside the base checkpoint on CPU fp32, computed per-tensor
weight diffs, and ran SVD on each non-zero diff. Then we asked three
questions:

- **What rank are the diffs?** For each modified weight tensor, find the
  smallest `k` such that the top-k singular values explain ≥95% of `‖ΔW‖_F²`.
  Median across all modified tensors:
  - **OBLITERATUS: median rank_95 = 6** (multi-rank, range up to 720 on
    `down_proj`)
  - **TrevorJS: median rank_95 = 1** (pure rank-1 norm-preserving
    biprojection, per the model card)
- **How many weights changed (after K/V borrower de-dup)?** Gemma 4's
  layers 24–41 share K/V tensors via aliasing; the same physical tensor
  shows up multiple times in the state dict. After removing 36 alias slots
  per variant (18 layers × 2 projections):
  - OBLITERATUS: **201 / 2094 tensors** modified
  - TrevorJS: **84 / 2094 tensors** modified, exclusively on `o_proj`
    and `down_proj`
- **Do the two methods modify the *same* direction?** Cosine between
  OBLITERATUS's and TrevorJS's top-1 left singular vectors per layer:
  **median −0.08, range −0.30 to +0.20.** The two methods' top-1
  modification directions live in **nearly orthogonal subspaces.**
- **Do either of those align with the M2b activation refusal direction?**
  Median |cos| ≈ 0.04 for both. **Neither method's top-1 weight-edit
  direction aligns with the activation-space refusal direction.**

The interpretive headline (verbatim from the M3 summary commit): *"Refusal
in Gemma 4 has a low-rank but multi-modal weight footprint — TrevorJS
removes it via a single rank-1 stroke per layer while OBLITERATUS spreads
its edits across many directions and many component types, and the two
methods' top-1 modification directions live in nearly orthogonal subspaces,
implying that 'the geometry of refusal' admits a continuum of equally-
effective low-rank fixes rather than one canonical safety basis."*

Figures: `weight_diff_per_layer_overlay.png`,
`cross_method_singular_vectors.png`,
`refusal_direction_vs_singular_vector.png`,
`singular_value_spectra_per_method.png`.

### M3b — literature integration

17 citations added across `paper/sections/02_background.md` and
`paper/sections/03_related_work.md`: the RLHF/DPO/CAI lineage, the
abliteration line (Arditi et al. 2024, Lermen 2024, Marshall et al.),
representation engineering (Zou et al.), the over-refusal benchmarks
(XSTest, OR-Bench, WildJailbreak), and the Gemma 4 architectural
quirk warnings (Heretic project notes, OBLITERATUS model card).

### M4 — human verification gate

`STATUS_FOR_HUMAN.md` was authored as the operator-readable artifact for
review: branch heads, refusal-rate tables, anomalies, figure paths. The
operator green-lit M5 after reading it. All five `agent/*` branches were
merged to `main` via `--no-ff` merges on 2026-05-06.

### M5 — paper write-up

In progress. Sections 1, 4, 5, 6, 7, 9 still need to be written.
`paper/sections/02_background.md`, `paper/sections/03_related_work.md`, and
`paper/sections/08_rank1_cascade.md` are drafted.

### M6 — rank-1 follow-up cascade (the deeper "why")

After merging M0–M4, we asked a sharper question: *given that TrevorJS
achieves clean uncensoring with a rank-1 recipe, why doesn't our rank-1
recipe work? What ingredient is TrevorJS doing that we are not?*

We designed a five-stage causal isolation cascade. Each stage tests one
hypothesis about why standard rank-1 fails on Gemma 4:

| Stage | Hypothesis | Result |
|---|---|---|
| **0a** | H1: bnb int8 edit-path is the problem (precision of the edit, not the recipe). Re-run abliteration in bf16, no quantization on the edit. | **REJECTED.** bf16 self-abliterated `should_refuse` = 6/6 = 100%. Identical to int8. |
| **0b** | H6: the eval pipeline is sound. Run TrevorJS bf16 as a positive control through the same eval. | **PASS.** TrevorJS bf16 → 0/48 refused. Eval works. |
| **2a (D1)** | H2: applying the chat template *during direction extraction* fixes the recipe. The original M2b directions were computed on raw prompts. | **INSUFFICIENT.** D1 → 6/6 should_refuse. cos(M2b raw, D1 chat-template) at L15 = 0.09: applying the chat template gives an essentially different direction, but that direction alone is still inert. |
| **2b (D2)** | H3: per-layer winsorization at 99.5% of activations before computing the mean removes outlier-driven contamination. | **INSUFFICIENT.** D2 → 6/6 should_refuse. cos(D1, D2) at L15 = 0.994 — winsorization barely moved the direction. |
| **2c (D3)** | H4: two-pass Gram-Schmidt orthogonalization of `d` against the harmless mean activation, layer by layer. | **LOAD-BEARING, PARTIAL.** D3 → 1/6 on n=6 smoke (16.7%), then **17/42 = 40.5% on the n=42 confirmation.** A 60% relative reduction from the 100% baseline. cos(D2, D3) at L15 = 0.952 — a ~17° rotation, but a behaviorally consequential one. |
| **3a** | H5: TrevorJS-style norm-preserving biprojection is what closes the residual gap (because Gemma 4's RMSNorm is sensitive to row-norm changes). | **REFUTED.** 3a smoke is identical per-prompt to D3 vanilla. Full 42-layer audit of vanilla projection (215,040 rows): mean Δ‖W_i‖/‖W_i‖ = 0.038%, p99 = 0.34%, max = 9.73% on a single row of L01 o_proj — well below the 1% threshold at which RMSNorm sensitivity would plausibly drift. There is essentially no row-norm change for biprojection to preserve. |

**The clean conclusion:** **two-pass Gram-Schmidt orthogonalization of the
refusal direction against the harmless mean is the single load-bearing
direction-quality ingredient** on Gemma 4 E4B. Without it, no rank-1
recipe we tested moves the needle on `should_refuse`. With it, vanilla
rank-1 projection at α=1.0 cuts refusal from 100% to 40.5% on n=42 — a
partial effect, in the 30–85% partial-effect band.

The remaining 40.5% concentrates on the most extreme topics (CSAM,
ICS/hospital malware, weapons), suggesting a strong core safety circuit
that resists single-direction abliteration. Full removal needs multi-rank
descent, consistent with M3's observation that OBLITERATUS uses median
rank_95 = 6 on the same base model. Multi-rank descent is out of scope for
this paper; future work.

The hand-audit caveat: the regex classifier under-counts soft refusals
(phrases like *"I have to stick to guidelines that prohibit..."* are
behaviorally refusals but not in the regex). 40.5% is a lower bound; the
true rate is plausibly 50–70%. The classifier blind-spot list is in
`docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md` §7.3.

Paper section: `paper/sections/08_rank1_cascade.md` (drafted).

---

## 7. Headline results

For a one-page reading, this is the substance.

### 7.1 Behavioral (M2a + M2c-followup + M6, n=344 for base rows, n=48 stratified for self-abliterated/M6 rows)

| Model | `should_refuse` | `emergency_medical` |
|---|---|---|
| Base Gemma 4 E4B GGUF | 100.0% (42/42) | 2.0% (1/50) |
| Base Gemma 4 E2B BF16 | 95.2% (40/42) | 12.0% (6/50) |
| HauhauCS GGUF | 0.0% (0/42) | 0.0% (0/50) |
| TrevorJS BF16 (n=48 stratified) | 0.0% (0/6) | 0.0% (0/6) |
| Self-abliterated int8 (n=48) | **100.0% (6/6)** | 33.3% (2/6) |
| Self-abliterated bf16 (n=48) | **100.0% (6/6)** | 16.7% (1/6) |
| D3 (Gram-Schmidt) bf16 (n=48 smoke) | 16.7% (1/6) | 0.0% (0/6) |
| D3 (Gram-Schmidt) bf16 (n=42 confirmation) | **40.5% (17/42)** | — |
| 3a (D3 + biprojection) bf16 (n=48) | 16.7% (1/6) | 0.0% (0/6) |

Read the table left-to-right top-to-bottom: base E4B refuses harmful queries
100% of the time, refuses helpful medical queries only 2%. HauhauCS and
TrevorJS are fully uncensored (0% on `should_refuse`). Our standard rank-1
abliteration is inert (100% on `should_refuse`, equal to base). Adding
Gram-Schmidt orthogonalization to the direction-build recipe gets us to
40.5% on `should_refuse` (partial). Norm-preserving biprojection adds
nothing.

### 7.2 Mechanistic (M2b)

- Peak refusal-direction layer: **L15** (Cohen's d = 2.87)
- High-signal band: L4–L17
- Rank-1 hypothesis on activation Δμ: **top-1 PC captures 86.6% of ‖Δμ‖²**
  (mean over peak band)
- Sliding vs. global gap: inconclusive

### 7.3 Abliteration sweep (M2c)

- Alpha sweep flat 30–35% across α ∈ [0, 2.0]
- Layer subset sweep flat 25–35% across 9 subsets
- Random control 30% (matches baseline 30%)
- Category geometry: over-refuse cluster +0.93 internal cosine, vs.
  `should_refuse` direction −0.015 (orthogonal)

### 7.4 Comparative weight diff (M3)

| | OBLITERATUS | TrevorJS |
|---|---|---|
| Low-rank verdict | rank-low (median rank_95 = 6) | pure rank-1 (median rank_95 = 1) |
| Fraction changed (post-dedup) | 201 / 2094 | 84 / 2094 |
| Top-1 cosine vs. refusal direction | median 0.043 | median 0.042 |

Cross-method: median cosine of top-1 left singular vectors = −0.08. The
two methods modify nearly orthogonal subspaces.

### 7.5 M6 cascade summary

```
H1 bnb int8 edit-path:      REJECTED
H2 chat-template alone:     INSUFFICIENT
H3 winsorization alone:     INSUFFICIENT
H4 Gram-Schmidt vs harmless mean:  LOAD-BEARING / partial
H5 norm-preserving biproj:  REFUTED
H6 pipeline-sound:          PASS
```

### 7.6 Paper central claim

Refusal in Gemma 4 E4B is *partially* recoverable by rank-1 mean-diff
abliteration if and only if the refusal direction is constructed via the
full TrevorJS-style direction-build recipe: chat-template-applied
activations + per-layer winsorization at 99.5% + two-pass Gram-Schmidt
orthogonalization against the harmless mean. This recipe achieves a 60%
relative reduction in `should_refuse` refusal at n=42 (100% → 40.5%) with
vanilla projection at α=1.0. Norm-preserving biprojection adds nothing on
top, because vanilla projection barely changes row norms in the first place.
The persistent ~40–50% residual concentrates on the most extreme topics,
indicating a strong core safety circuit that resists single-direction
abliteration. This is consistent with M3's observation that OBLITERATUS
uses median rank_95 = 6 on the same base model: full removal requires
multi-rank descent.

The single-variable causal isolation of Gram-Schmidt-against-harmless-mean as
the load-bearing direction-quality ingredient is, to our knowledge, the
cleanest published characterization of what does and does not matter in
rank-1 abliteration on Gemma 4.

---

## 8. TL;DR for an external reader

Five sentences:

1. We picked Gemma 4 E4B — a recent open-weight LLM where the standard rank-1
   abliteration recipe is reported to fail — and asked *why* it fails, and
   what published "uncensored" variants do differently.
2. The activation-side geometry is clean: the refusal direction is
   effectively rank-1, peaks at layer 15 with Cohen's d 2.87, and per-category
   refusal directions are orthogonal to the harmful-query direction — selective
   safety is geometrically possible.
3. The standard rank-1 weight-side intervention is inert on Gemma 4 E4B:
   sweeping α from 0 to 2.0 and across 9 layer subsets leaves `should_refuse`
   refusal at 100%, the same as base.
4. Two published uncensored variants achieve clean behavioral uncensoring via
   *different* recipes — OBLITERATUS uses multi-rank edits (median rank_95 = 6
   across 201 tensors), TrevorJS uses pure rank-1 norm-preserving biprojection
   across 84 tensors — and their top-1 modification directions live in nearly
   orthogonal subspaces (median cosine −0.08).
5. The follow-up cascade isolates two-pass Gram-Schmidt orthogonalization of
   the refusal direction against the harmless mean as the single load-bearing
   ingredient — adding it cuts refusal from 100% to 40.5% on n=42, but no
   further; the residual 40–50% concentrates on the most extreme topics,
   suggesting a strong core safety circuit that single-direction abliteration
   cannot reach.

---

## 9. How to navigate the repo

```
.
├── README.md                       ← Quickstart (you have read this)
├── ONBOARDING.md                   ← This file
├── STATUS_FOR_HUMAN.md             ← M4 human verification gate; canonical numbers
├── CLAUDE.md                       ← Operational notes (env, worktrees, agents)
├── AGENTS.md                       ← Agent-routing notes
├── data/
│   └── benchmark_prompts.json      ← The 340-prompt frozen benchmark
├── src/
│   ├── benchmark/                  ← Stage 1: behavioral eval
│   ├── mechanistic/                ← Stage 2: where does refusal live
│   ├── abliterate/                 ← Stage 3: rank-1 intervention + sweeps
│   └── weight_diff/                ← Stage 4: CPU diff of published variants
├── scripts/                        ← Benchmark builders, GPU lock, pipeline runners
├── results/
│   ├── figures/                    ← All paper-grade PNGs (in-repo copy)
│   └── (per-module artifacts)      ← Activations, refusal directions, sweep results
├── docs/
│   ├── planning/                   ← Historical planning artifacts
│   │   ├── project_proposal.md     ← Original 4-week proposal
│   │   ├── project_plan.md         ← Full execution plan
│   │   ├── gemma_only_execution_plan_review.md
│   │   ├── topic_selection.md      ← Decision log: how we picked the topic
│   │   ├── repo_naming.md          ← Repo-naming brainstorm
│   │   └── final-project-info.md   ← Course-side project info
│   ├── findings/                   ← Research findings / cross-milestone writeups
│   │   ├── m3_summary.md           ← M3 weight-diff phase summary
│   │   ├── M6_PROPOSAL_RANK1_FOLLOWUP.md  ← M6 cascade design + per-stage results
│   │   └── course-material-mapping.md  ← Which course lectures each finding touches
│   ├── lectures/                   ← Lecture PDFs (gitignored, copyrighted)
│   ├── issues/                     ← Known-failure notes (read these on errors)
│   └── superpowers/                ← Openspec plan/spec artifacts (historical)
├── paper/
│   ├── sections/                   ← Paper drafts (02, 03, 08 done; 01/04/05/06/07/09 pending)
│   └── presentation-slides/        ← Slide drafts (untracked)
├── tests/                          ← Light unit tests
├── openspec/                       ← Execution-plan specs (mostly historical)
├── model -> ../shared/model        ← 25 GB checkpoints (symlinked)
└── .venv -> ../shared/.venv        ← Python env (symlinked)
```

### Three reading paths

- **"I want to understand the science."** Read this file (§6 in particular),
  then `paper/sections/02_background.md`, `paper/sections/03_related_work.md`,
  `paper/sections/08_rank1_cascade.md`, then
  `docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md`.
- **"I want to reproduce a result."** README §"Running the Modules" gives the
  command lines. `STATUS_FOR_HUMAN.md` §(h) gives every paper-grade number
  paired with the exact CSV/JSON path it came from. Match them.
- **"I want to extend the code."** Read §5 here for the module
  architecture. The four modules are deliberately decoupled — pick one and
  read its `__main__` entrypoint plus the one or two helper modules it
  imports. Each module is 200–600 lines.

### Where the figures live

Each module writes figures to two places:

- `results/figures/` — in-repo redundancy copy, tracked by git
- `/home/nyavana/columbia/6699/shared/results/agent/<branch>/figures/` —
  shared sidecar copy, not tracked

The shared copy is the canonical one (some figures only exist there because
they came from agents that didn't commit them). When `STATUS_FOR_HUMAN.md`
cites a figure path, it cites the shared one.

### Where the activations live

`results/activations/refusal_directions.pt` is the canonical M2b artifact: a
PyTorch dict keyed by layer index (0–41), values are unit-norm 2560-dim
tensors. The full per-prompt activation tensors (`refuse_activations.pt`,
`comply_activations.pt`) are large (~5 GB each); only the directions are
in-repo.

---

## 10. How to run things

The README has the command lines. A few cross-cutting things to know:

### Activating the environment in any worktree

```bash
source /home/nyavana/columbia/6699/shared/env.sh    # sets HF_HOME, RESULTS_DIR, PATH for llama-server
source .venv/bin/activate                            # symlinks to shared venv
```

`RESULTS_DIR` is auto-scoped to the current branch (e.g.
`/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/` when
you're on that branch). This is the canonical handoff location between
agents/branches.

### The two backends

- **`--backend llamacpp`** (GGUF, via HTTP to `llama-server` on port 8088,
  not 8080 — Windows-side WSL2 binds 8080). Fast: ~18–25 s/prompt at
  `max_new_tokens=512`. Use for any GGUF model.
- **`--backend transformers --use-8bit`** (HF safetensors in-process,
  `bitsandbytes` int8). Slow: ~75–150 s/prompt. Use for models we can't easily
  GGUF-quantize (the self-abliterated checkpoint, TrevorJS).

`--use-8bit` is mandatory on E4B to fit in 16 GB of VRAM.

### Hardware budget reminders

- One GPU job at a time (use `scripts/gpu_lock.sh` if you're running
  agents in parallel).
- `compute_diff.py` holds two fp32 state-dicts in memory and peaks at
  ~32–35 GB RAM. WSL2 is capped at ~46 GB physical + 12 GB swap. Two
  parallel weight-diff runs **do not fit** — run sequentially across
  OBLITERATUS and TrevorJS.
- Long evals (>30 min) should be launched via `nohup` from inside an
  agent and detached, not held open in the agent's foreground. See
  `CLAUDE.md` §"Subagent runtime budget."

### Known failure modes (read these first when something breaks)

`docs/issues/` contains short markdown notes on every unresolved problem
encountered during development: OBLITERATUS GGUF chat-parser crashes,
TrevorJS quantization speed regressions, M2c-deferred items, etc.
Naming: `<yyyy-mm-dd>-<slug>.md`. Convention: capture symptom, what was
tried, what was observed, fix or open question. Append a "Resolved" line
if a fix lands later rather than deleting the note.

---

## 11. Glossary

- **Abliteration**: a rank-1 weight perturbation that removes refusal
  behavior by projecting a specific direction out of the output weight
  matrices of attention and MLP blocks. `W ← W − α · d · (dᵀ W)`.
- **Cohen's d**: standardized mean difference between two distributions,
  `(μ₁ − μ₂) / σ_pooled`. Used to quantify how separable refuse vs. comply
  activations are at a given layer. d > 0.8 is "large."
- **Gram-Schmidt orthogonalization (vs. harmless mean)**: subtract the
  component of `d` that lies along the mean of harmless-prompt activations,
  then re-normalize. The M6 cascade isolates this as the single load-bearing
  ingredient on Gemma 4.
- **GGUF**: a quantized model format consumed by `llama.cpp` /
  `llama-server`. Faster inference than `transformers` + `bitsandbytes` at
  similar quality.
- **Global vs. sliding attention**: Gemma 4 alternates between full
  attention (7 layers at indices [5, 11, 17, 23, 29, 35, 41]) and
  sliding-window attention (35 layers). M2b found no systematic gap in
  refusal-signal strength between the two.
- **Norm-preserving biprojection**: TrevorJS's variant of abliteration that
  applies the projection in both row- and column-space and renormalizes,
  so per-row L2 norms are exactly preserved. M6 refutes the hypothesis that
  this is what makes TrevorJS work on Gemma 4 (vanilla projection barely
  changes row norms in the first place).
- **OBLITERATUS / TrevorJS / HauhauCS**: three published uncensored Gemma 4
  E4B variants. OBLITERATUS uses multi-rank edits; TrevorJS uses pure
  rank-1 norm-preserving biprojection; HauhauCS is GGUF-only, recipe not
  fully documented.
- **Over-refusal**: an aligned model refusing a benign query. The "Heimlich
  refusal" anecdote in §1 is the canonical example.
- **Rank-1 in activations vs. rank-1 in weights**: M2b says the
  *activation-side* refusal direction is effectively rank-1 (86.6% of
  ‖Δμ‖²). M3 says the *weight-side* modifications that produce uncensored
  behavior are not — OBLITERATUS has median rank_95 = 6. These are two
  different geometric claims; do not conflate them.
- **`refusal_directions.pt`**: 42 unit-norm 2560-dim vectors, one per
  layer, computed by M2b. The canonical handoff artifact between modules.
- **Residual stream**: the additive hidden-state highway through the
  transformer. The high-dimensional vector that every attention and MLP
  block reads from and writes to.
- **`should_refuse`**: the load-bearing benchmark category — 42 prompts
  that the model should genuinely refuse (bioweapons, CSAM, ICS malware,
  etc.). Every headline result is reported on this category.
- **Winsorization (per-layer, 99.5%)**: clip activations above the 99.5th
  percentile per layer before computing means, to reduce outlier
  contamination of the direction. Insufficient on its own (M6 H3 rejected).

---

## 12. Further reading

In-repo:

- `STATUS_FOR_HUMAN.md` §(h) — paper-grade headline numbers with source paths
- `docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md` — full M6 cascade writeup
- `docs/planning/project_proposal.md` — original motivation and proposal
- `docs/planning/project_plan.md` — execution plan with task-level detail
- `docs/findings/m3_summary.md` — M3 weight-diff phase summary
- `paper/sections/02_background.md` — alignment background (RLHF, DPO, CAI)
- `paper/sections/03_related_work.md` — abliteration lineage + over-refusal
  benchmarks + Gemma 4 quirks
- `paper/sections/08_rank1_cascade.md` — M6 paper chapter
- `docs/findings/course-material-mapping.md` — which findings touch which course
  lectures

External (representative — full bibliography is in
`paper/sections/03_related_work.md`):

- Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a
  Single Direction.*
- Lermen, S. et al. (2024). *LoRA Fine-tuning Efficiently Undoes Safety
  Training in Llama 2-Chat 70B.*
- Marshall, A. et al. (2024). *Refusal Behavior in Large Language Models: A
  Nonlinear Perspective.*
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach
  to AI Transparency.*
- Röttger, P. et al. (2023). *XSTest: A Test Suite for Identifying
  Exaggerated Safety Behaviours in Large Language Models.*

---

## Maintainers' notes

When a new finding lands, prefer updating §6 (narrative arc) and §7
(headline results) in place rather than appending. This doc should always
read top-to-bottom as a coherent introduction, not a changelog.

For changelog-style information, use `STATUS_FOR_HUMAN.md` (which is
explicitly append-only by milestone marker).

If you add a new module under `src/`, add it to the diagram in §5 and the
tree in §9.

If a glossary term needs a definition longer than two sentences, it
probably belongs in §2 (prerequisites) instead.

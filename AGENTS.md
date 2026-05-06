# Repository Guidelines

## Project Structure & Module Organization

This is a Python research repository for studying refusal and safety alignment in LLMs. Source code lives in `src/` and is split by pipeline:

- `src/benchmark/`: refusal evaluation and result analysis.
- `src/mechanistic/`: activation extraction, refusal directions, layer analysis, and visualizations.
- `src/abliterate/`: rank-1 weight perturbation and selective safety experiments.
- `src/weight_diff/`: CPU-based checkpoint diff, SVD, and MoE expert analysis.

Inputs live in `data/`, especially `data/benchmark_prompts.json`. Generated outputs belong under `results/`; large artifacts are ignored except `.gitkeep` placeholders. Project planning and OpenSpec material live in `docs/` and `openspec/`. Final paper drafts belong in `paper/`.

## Worktrees & Shared Sidecar

Development uses multiple sibling git worktrees under `/home/nyavana/columbia/6699/` (`geometry-of-alignment`, `gb-ablit`, `gb-bench`, `gb-env`, `gb-mech`, `gb-paper`, `gb-wdiff`). They share a non-versioned sidecar at `/home/nyavana/columbia/6699/shared/`:

- `shared/model/` — 25 GB checkpoints, reached via the `model` symlink in each worktree
- `shared/.venv/` — Python 3.12 environment, reached via the `.venv` symlink
- `shared/hf-cache/` — `HF_HOME` target so HF downloads are pooled
- `shared/results/<branch>/` — branch-scoped scratch outputs (`$RESULTS_DIR`)
- `shared/docs/INSTRUCTIONS.md` — live shared notes
- `shared/env.sh` — source to set `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `RESULTS_DIR`

The `model` and `.venv` symlinks are excluded via `.git/info/exclude` (shared across all worktrees), not via `.gitignore`. See `CLAUDE.md` for the full layout diagram.

## Build, Test, and Development Commands

There is no build system. Run modules from the repository root.

```bash
# Activate (works from any worktree)
source /home/nyavana/columbia/6699/shared/env.sh
source .venv/bin/activate

# Day-to-day commands
python scripts/build_benchmark.py
python -m src.benchmark.analyze_results --results-dir results/
python -m src.mechanistic.layer_analysis --activations results/activations/
```

If you need to rebuild the venv from scratch (`shared/.venv/`), `pip install -r requirements.txt` is enough on the Python side. Note that the `gguf` Python package is pinned to llama.cpp's git tree (pypi releases lag), so the install fetches it via git. The GGUF inference backend uses upstream llama.cpp's `llama-server` binary, built from source into `shared/llama.cpp-cuda/` against `apt install nvidia-cuda-toolkit`; `shared/env.sh` puts it on PATH. `evaluate.py --backend llamacpp` talks to it over HTTP.

Use `python scripts/build_benchmark.py` after changing benchmark prompt definitions; it rewrites `data/benchmark_prompts.json` deterministically. GPU experiments with Gemma should pass `--use-8bit` to fit the expected 16 GB VRAM budget.

## Coding Style & Naming Conventions

Use Python 3.10+ style with type hints where they clarify data shapes. Follow existing module patterns: `snake_case` functions and files, uppercase constants, and small CLI-oriented modules with `argparse`. Keep path handling in `pathlib.Path` where practical. No formatter or linter is configured, so keep diffs consistent with nearby code and avoid broad reformatting.

## Testing Guidelines

No formal test suite is currently configured. Validate changes with the narrowest runnable command for the affected pipeline. For benchmark changes, run `python scripts/build_benchmark.py` and inspect the final category counts. For analysis changes, prefer a small existing artifact in `results/` or document the missing artifact/model requirement when a command cannot be run locally.

## Commit & Pull Request Guidelines

Recent commits use short, scoped messages such as `chore: freeze benchmark_prompts.json for M1` and `docs(openspec): mark ... tasks done`. Keep commits focused and use a conventional prefix like `docs:`, `chore:`, or `fix:` when appropriate.

Pull requests should state the affected pipeline, list validation commands run, and note required model checkpoints or hardware. Include figures or result summaries when changing visualization, benchmark, or experiment output behavior.

## Security & Configuration Tips

Do not commit model weights, checkpoints, caches, logs, or generated result files. The `.gitignore` excludes common large artifacts such as `*.pt`, `*.safetensors`, `*.gguf`, `models/`, `model/`, and most files under `results/`. The `model` and `.venv` symlinks pointing into `shared/` are excluded via `.git/info/exclude` (per-repository, not committed) so they stay invisible across all worktrees.

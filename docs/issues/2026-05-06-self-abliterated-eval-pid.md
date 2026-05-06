# Self-Abliterated Benchmark Eval — PID and Status

**Date:** 2026-05-06  
**Task:** M2c-followup 6.2  
**Branch:** `agent/benchmark-eval`

## Summary

Detached benchmark evaluation of the project's own self-abliterated Gemma 4 E4B model
launched successfully.

## Process Info

- **Outer flock PID:** 572070  
- **Python eval PID:** 572076  
- **Log file:** `/tmp/eval-self-ablit.log`  
- **Launch command:**

```bash
nohup bash -c '
  source /home/nyavana/columbia/6699/shared/env.sh
  cd /home/nyavana/columbia/6699/gb-bench
  source .venv/bin/activate
  flock --exclusive --wait 30 /home/nyavana/.geometry-of-alignment/.gpu.lock \
    python -m src.benchmark.evaluate \
      --backend transformers \
      --model /home/nyavana/columbia/6699/shared/results/agent/abliteration/models/gemma-4-e4b-abliterated/ \
      --benchmark data/benchmark_prompts.json \
      --output /home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/ \
      --use-8bit
' > /tmp/eval-self-ablit.log 2>&1 < /dev/null & disown
```

## Model Path

`/home/nyavana/columbia/6699/shared/results/agent/abliteration/models/gemma-4-e4b-abliterated/`

- Size: ~11 GB (8-bit safetensors from M2c abliteration commit 9c5cef6)
- Contents: `chat_template.jinja`, `config.json`, `generation_config.json`,
  `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`

## Output Path

`/home/nyavana/columbia/6699/shared/results/agent/benchmark-eval/refusal_rates/gemma4_e4b_self_abliterated/`

Files produced on completion: `evaluation_results.json`, `evaluation_results.csv`

## ETA

- 344 prompts, ~25–30 s/prompt at 8-bit on Gemma 4 E4B
- Estimated runtime: **~2.5 hours** from launch (~14:01 local time)
- Expected completion: ~**16:30 local time**

## Verification (at launch)

Model weights loaded in ~24 s. Evaluation started, 344 prompts queued, 0 completed
at time of agent exit. GPU confirmed idle before launch (1104 MiB / 16376 MiB used).

## Expected Result

M2c sweep showed abliteration largely ineffective on Gemma 4 E4B 8-bit — refusal rate
30–35% across all alpha values and layer subsets vs 30% for random direction control.
Expected self-abliterated refusal rates to closely match base E4B:
- `should_refuse`: ~100% (base)
- `emergency_medical`: ~2% (base, i.e., no meaningful over-refusal in base, confirming
  that the abliteration changed nothing)

This is the paper's key negative result: rank-1 abliteration fails to budge refusal
behavior on Gemma 4 E4B under 8-bit quantization.

## Follow-up

A follow-up agent (M2c-followup 6.4–6.5 final) should fire once
`evaluation_results.csv` lands and regenerate the refusal heatmap including this row.
Check completion with: `tail /tmp/eval-self-ablit.log` or `ls -la` on the output path.

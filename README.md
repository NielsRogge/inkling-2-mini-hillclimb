# Inkling-2-Mini serving hill-climb (RTX PRO 6000)

Goal: serve [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) as fast as possible on a Hugging Face Inference Endpoint (`aws/us-east-2/nvidia-rtx-pro-6000 x1`).
Method: the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill). Ceilings come first, then one change per experiment, each run between controls, with medians of 6. Experiments run on [HF Jobs](https://huggingface.co/docs/hub/jobs-overview) (`rtx-pro-6000` flavor, the same GPU as the endpoint).

## Result

Best config, live as a **new** endpoint `huggingface/inkling2mini-tuned-dspark` (aws us-east-2, 1x RTX PRO 6000, authenticated, scales to zero after 60 min). Reproduce with `deploy.sh`.

```
vllm/vllm-openai@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7   (vllm 0.34.0rc1.dev1235+gf2f9edcac)
vllm serve thinkingmachines/Inkling-2-Mini --revision 7a67ae23… --tensor-parallel-size 1 --reasoning-parser inkling \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

| Metric (1x RTX PRO 6000, temp 1.0, thinking on) | Before: SGLang fa4, no spec (existing endpoint config) | vLLM, no spec | **Best: vLLM + DSpark-7** | Live endpoint (incl. network) |
|---|---:|---:|---:|---:|
| 1-user decode, tok/s (median of 12) | 134–139 | 157 | **190.6** | 194.9 |
| 8-user aggregate decode, tok/s | 543–598 | 638 | **716** | 673 |
| Prefill 4K / 32K, tok/s (warm) | 18.7K / 17–19K | 20.7K / 20.7K | 20.7K / 20.6K | 11.7K / 19.2K |
| TTFT, short prompt | 38–114 ms | 33 ms | 40 ms | 198 ms |

Gates (pinned image): needle 3/3 at 127K and 250K tokens, 4 x 58K concurrent soak OK, greedy outputs byte-identical to no-spec (lossless).
Ceilings, what was tried and rejected, and why it plateaus: see `THEORY.md` and `PLAN.md`.

| File | What |
|---|---|
| `PLAN.md` | Timestamped log of every experiment, verdict and reason (including failures) |
| `THEORY.md` | Measured hardware and theoretical ceilings |
| `configs/plan-*.json` | Run lists submitted to Jobs (one server config per run) |
| `configs/ceiling-*.json` | Inputs to `skill-scripts/ceiling.py` |
| `jobs/driver.py` | Runs inside a Job: start server → `bench.py` → drain/stop, for each run in the plan |
| `jobs/hwprobe.py` | Measured copy bandwidth, BF16/FP8 GEMM throughput |
| `submit.sh` | `./submit.sh FLAVOR IMAGE NAME PLAN.json [DEFAULTS.json] [TIMEOUT]` |
| `watch.sh` | Polls Job logs for ledger lines and terminal states |
| `ledger-*.jsonl` | Raw bench results (one line per run; `ledger-gates.jsonl` = acceptance gates, `ledger-endpoint.jsonl` = live endpoint) |
| `deploy.sh` | Creates the tuned endpoint (new name; never touches existing endpoints) |
| `jobs/bench_remote.py` | bench.py against an authenticated endpoint |
| `skill-scripts/` | Unmodified copy of the skill's harness (`bench.py`, `ceiling.py`, …) |
| `logs/` | Job logs |

Model revision pinned: `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`.

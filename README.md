# Inkling-2-Mini serving hill-climb (RTX PRO 6000)

Goal: serve [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) as fast as possible on a Hugging Face Inference Endpoint (`aws/us-east-2/nvidia-rtx-pro-6000 x1`).
Method: the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill). Ceilings come first, then one change per experiment, each run between controls, with medians of 6. Experiments run on [HF Jobs](https://huggingface.co/docs/hub/jobs-overview) (`rtx-pro-6000` flavor, the same GPU as the endpoint).

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
| `ledger-*.jsonl` | Raw bench results (one line per run) |
| `skill-scripts/` | Unmodified copy of the skill's harness (`bench.py`, `ceiling.py`, …) |
| `logs/` | Job logs |

Model revision pinned: `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`.

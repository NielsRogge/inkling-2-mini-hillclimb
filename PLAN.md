# PLAN / experiment log: Inkling-2-Mini serving hill-climb

Target: fastest serving of thinkingmachines/Inkling-2-Mini@7a67ae23 for deployment on HF Inference Endpoints. Hill-climb compute runs on HF Jobs (namespace huggingface).
Metrics, in priority order: c1 decode tok/s, c8 aggregate decode, prefill 4K/32K, TTFT, startup. Quality: lossless only (DSpark with probabilistic draft sampling is distribution-preserving).
Harness: inference-optimisation skill bench.py (unchanged), temperature 1.0 (the model's generation_config default), thinking on, 512 max tokens, 6-prompt panel, median of 6.
Every candidate is run between controls in the same Job (ctl / cand / ctl).

## Log
- 2026-11-24 Scope changed by the user: **RTX PRO 6000 only**. H200 runs cancelled before any results.
- Deployment target confirmed: Inference Endpoints offers `aws / us-east-2 / nvidia-rtx-pro-6000 x1` ($2.75/h), the same GPU as the Jobs `rtx-pro-6000` flavor, so Jobs numbers carry over.
- Existing org endpoints (`inkling-mini-private-1120-huggingface-91/92`, paused) use: lmsysorg/sglang@sha256:421aed3a…, model rev 1fb4134b, `--attention-backend fa4 --reasoning-parser inkling --tool-call-parser inkling2 --enable-multimodal`, **no speculative decoding**. This is the reference "before" config.
- RULE (user): never modify, resume, pause or delete existing endpoints (colleagues' `inkling-mini-private-1120-*`). Read-only reference. Our deployment is a NEW endpoint with its own name (prefix `inkling2mini-tuned-`).
- rtx-sg-base-a (SGLang, fa4, no spec): c1 133.9 (very tight, ±0.2%), c8 543 agg (71.5/stream), prefill 4K 18.7K / 32K 16.9K tok/s, TTFT 41 ms, startup 297 s (incl. 20 GB download). Prefill already exceeds the BF16-based "ceiling", which confirms that ceiling is conservative (NVFP4 GEMMs).
- rtx-vl-base-a (vLLM nightly 0.34.0rc1.dev1235, no spec): c1 **156.7** (+17% vs SGLang 133.9), c8 632 agg (80.9/stream, +16%). Prefill 4K 202 / 32K 2,838 tok/s and TTFT 2.3 s are **invalid**: the log shows "Triton kernel JIT compilation during inference" (_expert_gemm, _mxfp8_gemm, _k_group_kernel …) exactly during those requests. Measurement artifact, not engine speed.
  - Harness fix (jobs/driver.py): the driver now launches the server itself, then runs an unmeasured warmup (short chat, every prefill length, a cN burst, x2) before bench.py measures the warm server. startup_s is still recorded, warmup_s too. Round-1 numbers stay valid for decode only.

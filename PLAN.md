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
- rtx-sg-ds8-a (SGLang + DSPARK, 8 draft tokens, max-running 4): c1 median **159.6** (+19% vs SGLang base, ≈ vLLM base 156.7). Per-prompt spread 141–218 (code/structured prompts accept more). Logged accept len 1.9–3.1 (accept rate 0.12–0.30) at temperature 1.0, so most of the 8 drafts are wasted verify work. c4 agg 385 (not comparable with c8).
  - Hypothesis for the next round: fewer drafts (3–5) cut verify cost per step at the same ~2-token acceptance. Temperature 1.0 (the model default) caps acceptance; also report a temp-0.6 panel as a secondary metric, labelled.
- rtx-sg-base-b (control): c1 138.6, c8 598, prefill 19.1K/19.1K. Drift vs base-a: c1 +3.5%, c8 +10% (base-a was the first run in the Job: cold caches). **Verdict: SGLang DSpark-8 accepted over SGLang base on c1** (159.6 beats both controls, 133.9 and 138.6).
- rtx-sg-ds8-b (repeat): c1 166.6 (a: 159.6), c4 402, TTFT 38 ms. The win reproduces. SGLang DSpark-8 c1 mean of repeats ≈ 163 vs control ≈ 136 (**+20%**).
- SGLang round 1 Job finished. Round 2 submitted (job 6b05a46d…): DSpark draft-token sweep 3/4/5/6, each between DSpark-8 controls, c4, with the warmup driver.
- **Harness bug (failure, logged):** rtx-vl-ds7-a never became healthy ("Engine core initialization failed"), and the SGLang round-2 d4 run hung. Cause: stop.sh killed `-$(cat srv.pid)`, but `$!` after `setsid nohup …&` is not guaranteed to be the session/group id (setsid forks when it is already a group leader). So the previous server was never stopped, and the next one found the GPU and port 8000 occupied. SGLang round 1 survived by luck. The death went undetected because the orphan became a zombie, and `kill -0` still succeeds on zombies.
  - Fix: the driver now owns the server via `subprocess.Popen(start_new_session=True)` (pid == pgid guaranteed), uses `Popen.poll()` for liveness, and stops with killpg SIGINT → 90 s drain → SIGKILL, then waits for GPU memory < 2 GB **and** port 8000 free before the next launch.
  - Both Jobs cancelled and resubmitted: vLLM DSpark sweep (6b05abd4…) and SGLang draft sweep (6b05abd5…). Valid results kept: rtx-vl-base-a (decode only), rtx-sg2-ctl8-a (c1 160.8, consistent with round 1).
- **SGLang DSpark draft length is not tunable:** `ValueError: DSpark speculative_num_draft_tokens must equal gamma + 1 (= 8 for gamma=7)`. Gamma comes from the bundled drafter (block_size 8). The d3/d4/d5/d6 runs are void (fail in 10 s). The lever "draft length" is closed for SGLang. SGLang source (read from the image via CPU Jobs): relevant tunables are env flags, not CLI. Candidates, with defaults in this image:
  `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` (umbrella; gates SPEC_VERIFY_REUSE_STAGED_INPUTS, SPEC_COMPACT_VERIFY_SAMPLING …), `SGLANG_OPT_USE_FUSED_DRAFT_SAMPLE`=0, `SGLANG_OPT_MOE_QUANT_ONCE`=0, `SGLANG_DSPARK_USE_GREEDY_DRAFT`=0, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`=1, `SGLANG_OPT_USE_GUMBEL_SAMPLE`=1.
- rtx-sg2-ctl8-b: c1 179.8, c4 407. DSpark c1 at temperature 1.0 is noisy: across 5 DSpark-8 runs, medians are 159.6 / 166.6 / 160.8 / 157.7 / 179.8, per-prompt 141–243. Next DSpark rounds use 12 reps.
- **vLLM DSpark fails to start on RTX PRO 6000** in two separate Jobs ("Engine core initialization failed", 424 s in). The root cause is above the tail, so the driver now greps all error lines. Debug Job 6b05ae51… runs ds7 and ds7 + --enforce-eager. vLLM base is still the best non-speculative engine (c1 156.7).
- SGLang round 2b complete. DSpark-8 controls across 7 runs: c1 median **166.6** (sd 7.5, 4.5%), c4 agg median **406.5** (sd 14.4). Noise floor for accepting a DSpark lever: >~5% c1 over neighbouring controls, repeated.
- **vLLM DSpark root cause found:** `ValueError: To serve at least one request with the model's max seq len (1048576), 52.64 GiB KV cache is needed … available 48.07 GiB`. At the default 1M context, the drafter's extra memory pushes the KV pool too small on 96 GB. Not a kernel or engine bug. Fix: `--max-model-len 262144`. vLLM round 3 (6b05…) applies it to **both** controls and candidates, so the cap is not a confounder. DSpark draft sweep 3/4/5/7, 12 reps.
- rtx-sg3 MEGA (`SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1`): c1 175.4 vs controls 168.5 / 168.9 (+3.9%), c4 417 vs 389 / 378 (+9%). The c1 gain is inside the 4.5% noise floor, while the c4 gain beats both controls. **Provisional accept, pending a repeat** (scheduled in the next round).
- rtx-vl3-base-a (vLLM, 262K cap, warmed): c1 156.8 (identical to round 1, sd 0.1%), c8 635, TTFT 33 ms, prefill 4K **20.6K** (> SGLang 19.1K), but prefill 32K only **4.9K** (SGLang 19.0K, 3.9x faster). The JIT warnings all fall inside the warmup window (15:47:15–39; bench started 15:47:50), so this is a real long-prefill weakness (Triton `_rel_attention_kernel` path?), not an artifact.
- rtx-sg3 FUSED_DRAFT_SAMPLE=1: c1 171.8 vs controls 168.9 / 168.4 (+1.9%), c4 411 vs 378 / 402. **Reject (inside noise).**
- vLLM round 3: ds7 OOM at startup (`Tried to allocate 5.07 GiB … 1.90 GiB free`). The new "extensible KV cache" takes all memory left after warmup, leaving none for the DSpark drafter allocation. Fix: `--gpu-memory-utilization 0.85` on **all** runs. Round 4 puts ds7 first so a failure costs little.
- rtx-sg3 MOE_QUANT_ONCE=1: c1 170.5 vs controls 168.4 / 169.3 (+1%), c4 408 vs 402 / 414. **Reject (no effect).**
- SGLang round 3 complete (last control ctl-e: c1 169.8, c4 406). Round 4 submitted: MEGA repeats interleaved with controls; on top of MEGA, greedy draft (SGLANG_DSPARK_USE_GREEDY_DRAFT=1) and max-running-requests 8 (measured at c8, against a MEGA mrr4 run also measured at c8).
- **rtx-vl4-ds7-a (vLLM + DSpark 7, gpu-mem 0.85, 262K): starts and runs.** c1 **189.3** (+21% vs vLLM base 156.8; +12% vs SGLang DSpark ~169), c8 **714** agg (vLLM base 635). TTFT 40 ms, prefill 4K 20.8K, 32K 4.9K (the vLLM long-prefill weakness remains). **New best c1 + c8.** Waiting on controls / ds4 / ds5 / ds3.
- rtx-vl4-base-a: c1 157.0, c8 637, prefill 4K 20.7K, **32K 20.7K**. The earlier 32K = 4.9K was **JIT on the first run in each Job** (vl3-base-a and vl4-ds7-a were both first; warmup used different-length docs so last-chunk shapes differed). Correction: vLLM long prefill ≈ SGLang. Bench caveat: a Job's first run under-reports 32K prefill.
- vLLM ds4: `ValueError: Inkling2 DSpark requires num_speculative_tokens = 7 (the checkpoint's block size - 1)`. Draft length is fixed in **both** engines, so the lever is closed.
- vLLM speculative-config levers (from the image's config/speculative.py). **Lossless** ones queued in round 5 (Job …), each between DSpark-7 controls:
  `enable_adaptive_verification` (DSpark-only, per-request confidence-sized verify budget), `rejection_sample_method=block` (block verification, Sun et al.; lossless, accepts ≥ token-wise), `draft_sample_method=greedy` (vs the recipe's probabilistic), `draft_lm_head_quantization=fp8` (the target verifies with its own head, so only acceptance changes).
  Excluded as **lossy**: `rejection_sample_method=fly` and `synthetic`.
- rtx-vl4-ds7-b (repeat): c1 **190.4**, c8 **724**, prefill 4K/32K 20.7K/20.6K. The vLLM DSpark-7 win reproduces (189.3 / 190.4). Accepted as the current best: vLLM nightly + DSpark-7 probabilistic, 262K context, gpu-mem 0.85.
- vLLM round 4 summary: DSpark-7 c1 = 189.3 / 190.4 / 191.1 / 190.7 (median **190.6**, sd 0.8). c8 = 714 / 724 / 718 / 686 (median **716**). Base controls: c1 157.0 / 157.2, c8 637 / 639. **+21.3% c1, +12.3% c8. Accepted.** Per-prompt c1 values repeat almost exactly across vLLM runs, so vLLM noise is far below SGLang's (sd 0.4% vs 4.5%).
- SGLang MEGA repeats: mega-a 178.1 (ctl 162.2 / 172.9), mega-b 167.2 (ctl 172.9 / next). c4: 388 / 411 vs ctl 414 / 419. **MEGA rejected:** the round-3 gain did not reproduce and c4 is no better.
- vLLM `enable_adaptive_verification`: not supported for Inkling2 DSpark (`assert self._draft_topk is None and not self.enable_adaptive_verification` at engine init). Lever closed. rtx-vl5-ctl-a: c1 190.3 / c8 722 (matches round 4; 32K prefill 4.7K is again the first-run-in-Job JIT artifact).
- vLLM block verification: c1 190.1 / c8 714 vs ctls 190.5 / 727 and 190.3 / 719. **Reject (no effect).**
- SGLang greedy draft (on MEGA): c1 156.8 / c4 369, clearly worse. **Reject.** SGLang mega-c: 174.4 / 419, within the control range.
- vLLM greedy draft: c1 168.6 / c8 606 vs ctls ~190.4 / ~720. **Reject (−11%).** Probabilistic drafting matters at temperature 1.0.
- vLLM `draft_lm_head_quantization=fp8`: `ValidationError: only supported by the DeepSeek-V4/V4.1 DSpark drafters`. Lever closed. ctl-d: 191.3 / 719.
- SGLang mrr8 (MEGA, c8): c8 660 agg, c1 179. Below vLLM DSpark c8 (~720).

## STATUS: plateau (2026-11-24 16:25 UTC)
Four consecutive levers on the best config were null, worse or unsupported (adaptive verification, block verification, greedy draft, fp8 draft head). SGLang levers (MEGA, fused draft sample, MoE quant once, greedy draft) were also null.

| Metric (RTX PRO 6000, temp 1.0, thinking on) | Baseline (SGLang fa4, no spec = colleagues' endpoint config) | Best: vLLM nightly + DSpark-7 | Ceiling (THEORY.md) |
|---|---:|---:|---:|
| c1 decode (median of 12, 6-prompt panel) | 133.9–138.6 | **190.6** (4 runs, sd 0.4%) | ~256 bandwidth-only (no spec) / ~345 at accept 4; practical c1 is launch/latency-bound |
| c8 aggregate decode | 543–598 | **716** (median of 8 runs) | 5,418+ (bandwidth); far from binding |
| Prefill 4K / 32K (warm) | 18.7K / 16.9K–19.1K | 20.7K / 20.6K | ≥17.3K / 15.5K (BF16-based; NVFP4 exceeds it) |
| TTFT short | 38–114 ms | 40 ms | |
| Startup (cached weights) | 90–297 s | ~95 s (Job), first ~350 s with download | |

What got the gains: (1) engine choice: vLLM nightly beats SGLang by +17% c1 without spec; (2) DSpark-7 probabilistic: +21% c1 / +12% c8 on vLLM; (3) making DSpark start on 96 GB: `--max-model-len 262144` + `--gpu-memory-utilization 0.85`.
What limits further gains: DSpark acceptance ≈ 2.3–2.4 tokens/step at temperature 1.0 (per-position 0.6 / 0.33 / 0.19 …). Draft length is fixed at 7 by the checkpoint in both engines, and the c1 step is latency-bound (48 layer visits). The next levers need kernel work, or a lower serving temperature (a product decision, not a lossless config change).
Next: acceptance gates on the pinned image digest (Job 6b05ba47…): needles at ~131K and ~258K, a 4 × 60K concurrent soak, a greedy equivalence check vs no-spec, and a repeat of the speed numbers.
- vLLM round 5 complete (ctl-e 190.8 / 723). SGLang MEGA with mrr4 measured at c8: 412 agg vs mrr8 660, so on SGLang `--max-running-requests 8` is needed for c8 (it still trails vLLM 716–727).

## Acceptance gates: PASS (Job 6b05ba47…, image pinned `vllm/vllm-openai@sha256:eaa63fd6…`, vllm 0.34.0rc1.dev1235+gf2f9edcac, the same build as the hill-climb)
- Long-context needle, ~50% ctx: 126,537 prompt tokens, **3/3**, TTFT 12.5 s (10.1K tok/s cold prefill)
- Long-context needle, ~95% ctx: 249,970 prompt tokens, **3/3**, TTFT 28.7 s (8.7K tok/s)
- Concurrent soak: 4 × ~58K tokens at once, **4/4 ok, 3/3 retrieval each**, 19.8 s wall
- Greedy equivalence (temp 0, 6 prompts, 256 max tokens incl. reasoning): DSpark-7 vs no-spec **6/6 byte-identical**; DSpark repeat also identical. Lossless confirmed.
- Speed on the pinned image: c1 **189.9 / 190.2** (best runs), c8 682 / 690; base on the same image 157.5 / 640. Matches the hill-climb.

## Deployment
- 16:39 UTC: created **new** endpoint `huggingface/inkling2mini-tuned-dspark` (aws us-east-2, nvidia-rtx-pro-6000 x1, authenticated, min 0 / max 1, scale-to-zero after 60 min). Pinned image, model rev 7a67ae23, exactly the gated config (`deploy.sh`). URL https://6b05bdde938ee8479c23247f.endpoints.huggingface.cloud. Existing colleague endpoints untouched.
- 16:46 UTC: endpoint **running**. Smoke test: correct answer, reasoning parsed into `reasoning`. End-to-end bench from outside the cluster (jobs/bench_remote.py, the same bench.py; numbers include the network): c1 **194.9** (median of 12), c8 **673**, prefill 4K 11.7K / 32K 19.2K tok/s, TTFT short 198 ms. These match the Jobs measurements once network RTT is accounted for: TTFT and 4K prefill absorb ~160 ms RTT, and c8 wall clock includes it.
- Correction (see THEORY.md): at the measured acceptance, the DSpark c1 ceiling is ~201–207 tok/s, and 190.6 is ~95% of it. The earlier "latency-bound" explanation applies only to the no-spec config (61% of 256).

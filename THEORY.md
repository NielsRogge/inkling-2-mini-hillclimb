# Ceilings: Inkling-2-Mini — H200 (initial) and RTX PRO 6000 (focus)

Measured on the Job host (jobs/hwprobe.py): copy bandwidth **4,276 GB/s**, BF16 GEMM **802 TFLOPS**, FP8 GEMM **1,425 TFLOPS**, driver 595.112.07, CUDA 13.

Model bytes (safetensors header, rev 7a67ae23): experts 15.67 GB (NVFP4 + MXFP8 scales), attention/dense 1.37 GB BF16, embed 0.93 GB, unembed 0.93 GB, vision 0.64 GB, DSpark/MTP drafter 0.64 GB.
Looped architecture: 24 weight layers, but layers 6-17 run 3 passes, so there are **48 layer visits per token**. Each visit re-reads its weights, giving
dense_eff = 1.37/24*48 = 2.73 GB and experts_eff = 0.653*48 = 31.3 GB at 8/128 routing (~1.96 GB per token at c1).
Global-attention visits = 8 (layers 5, 11x3, 17x3, 23), so KV ≈ 8 * 8 heads * 128 * 2 * 2 B = 32 KB/token (local layers have a 1,024-token window).

```
bandwidth 4275 GB/s, peak 802 TFLOPS, MFU 0.5

Decode ceiling, concurrency 1 (aggregate tok/s):
  drafts accept       4K      32K     128K    1024K
       0    1.0      744      639      432      107
       7    3.0      752      713      605      250
       7    4.0     1003      951      806      333
       7    5.0     1254     1188     1008      417

Decode ceiling, concurrency 8 (aggregate tok/s):
  drafts accept       4K      32K     128K    1024K
       0    1.0    15767    11002     5403      940
       7    3.0    20206    17051    11106     2610
       7    4.0    26941    22734    14807     3481
       7    5.0    33676    28418    18509     4351

Prefill ceiling (tok/s, practical):
     4K    33413
    32K    29902
   128K    21982
  1024K     6331

KV: 8.59 GB per 256K-token stream
```

Reading: the c1 bandwidth ceiling (~744 tok/s without speculation) is far above what a batch-1 engine can reach. At c1, 48 sequential layer visits of small GEMV/MoE kernels make decode **latency/launch-bound**, not bandwidth-bound. So the practical levers are CUDA graphs plus speculative decoding (DSpark drafts 8 tokens per drafter forward). For c8+, the bandwidth ceiling is the relevant one.

## RTX PRO 6000 Blackwell Server Edition (HF Jobs `rtx-pro-6000`), the focus hardware
Measured: copy bandwidth **1,469 GB/s**, BF16 **416 TFLOPS**, FP8 **792 TFLOPS**, sm_120, 96 GB, driver 620.70.06. NVFP4 tensor cores are native here (unlike on H200).
```
bandwidth 1469 GB/s, peak 416 TFLOPS, MFU 0.5

Decode ceiling, concurrency 1 (aggregate tok/s):
  drafts accept       4K      32K     128K    1024K
       0    1.0      256      220      148       37
       7    3.0      258      245      208       86
       7    4.0      345      327      277      115
       7    5.0      431      408      346      143

Decode ceiling, concurrency 8 (aggregate tok/s):
  drafts accept       4K      32K     128K    1024K
       0    1.0     5418     3781     1857      323
       7    3.0     6943     5859     3816      897
       7    4.0     9258     7812     5088     1196
       7    5.0    11572     9765     6360     1495

Prefill ceiling (tok/s, practical):
     4K    17331
    32K    15510
   128K    11402
  1024K     3284

KV: 8.59 GB per 256K-token stream
```
Peak TFLOPS uses BF16. NVFP4 expert GEMMs can exceed it, so the prefill ceiling here is conservative.

## Correction after the hill-climb (speculative decoding ceiling)
`ceiling.py` with the **measured** DSpark acceptance (≈2.33–2.40 tokens/step at temperature 1.0, 7 drafts): the c1 ceiling is **201–207 tok/s** at short context. Verifying 8 positions touches the union of their experts, 1−(1−8/128)^8 ≈ 40% of each layer's experts versus 6% for plain decode. The measured 190.6 tok/s is therefore **~93–95% of the bandwidth ceiling**, so the speculative config is bandwidth-bound, not latency-bound as stated earlier. Without speculation, vLLM's 157 tok/s is 61% of the 256 tok/s ceiling: that regime *is* latency/kernel-bound.

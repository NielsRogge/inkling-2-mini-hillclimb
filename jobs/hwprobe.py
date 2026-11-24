# Runs inside the job: inventory + measured bandwidth / matmul throughput (skill step 0).
import subprocess, json, time, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print(subprocess.run(["bash","-c","nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv"], capture_output=True, text=True).stdout)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e: print("vllm import", e)
try:
    import sglang; print("sglang", sglang.__version__)
except Exception as e: print("sglang import", e)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
dev = "cuda"
def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
a = torch.empty(4 * 2**30, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
t = timeit(lambda: b.copy_(a))
bw = 2 * a.numel() / t / 1e9
x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev); y = torch.randn_like(x)
t2 = timeit(lambda: x @ y)
bf16 = 2 * 8192**3 / t2 / 1e12
res = {"copy_bw_gbs": round(bw, 1), "bf16_tflops": round(bf16, 1)}
try:
    xf = x.to(torch.float8_e4m3fn); yf = y.to(torch.float8_e4m3fn).t().contiguous().t()
    s = torch.tensor(1.0, device=dev)
    t3 = timeit(lambda: torch._scaled_mm(xf, yf, scale_a=s, scale_b=s, out_dtype=torch.bfloat16))
    res["fp8_tflops"] = round(2 * 8192**3 / t3 / 1e12, 1)
except Exception as e:
    res["fp8_err"] = repr(e)[:200]
print("HWPROBE", json.dumps(res))

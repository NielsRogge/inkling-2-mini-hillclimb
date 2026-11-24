# Hill-climb driver that runs inside one HF Job (one GPU, one server at a time).
# PLAN (env, JSON list) = runs in order; each run: {"name", "cmd" (server shell command), optional bench overrides}.
# Uses the inference-optimisation skill's bench.py unchanged; prints each ledger line with a LEDGER marker.
import json, os, subprocess, sys, time, base64, urllib.request

W = "/work"; os.makedirs(f"{W}/logs", exist_ok=True); os.makedirs(f"{W}/configs", exist_ok=True)
os.chdir(W)
BENCH = f"{W}/bench.py"  # inference-optimisation skill scripts/bench.py, shipped verbatim in env
open(BENCH, "wb").write(base64.b64decode(os.environ["BENCH_B64"]))
import importlib.util
_sp = importlib.util.spec_from_file_location("bench", BENCH); bench = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bench)

import signal, socket
SRV = None

def port_free(port=8000):
    with socket.socket() as so:
        return so.connect_ex(("127.0.0.1", port)) != 0

def stop_server():
    """Drain-then-stop: SIGINT the server's process group, wait, then SIGKILL; wait for GPU memory and port to be released."""
    global SRV
    if SRV is not None:
        try:
            os.killpg(SRV.pid, signal.SIGINT)
            SRV.wait(timeout=90)
        except Exception:
            pass
        try:
            os.killpg(SRV.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            SRV.wait(timeout=30)
        except Exception:
            pass
        SRV = None
    for _ in range(120):
        u = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1"], capture_output=True, text=True).stdout.strip() or "0"
        if int(u) < 2000 and port_free():
            break
        time.sleep(1)
    else:
        print("WARNING: GPU memory or port 8000 still held after stop", flush=True)
        subprocess.run(["bash", "-c", "nvidia-smi; (ss -ltnp || netstat -ltnp) 2>/dev/null | grep 8000; ps -eo pid,pgid,stat,etime,cmd | grep -v ' Z ' | head -40"])

open(f"{W}/mem.sh", "w").write("echo $(( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) * 1048576 ))\n")

plan = json.loads(base64.b64decode(os.environ["PLAN"]).decode())
defaults = json.loads(base64.b64decode(os.environ.get("DEFAULTS", "e30=")).decode())
for run in plan:
    name = run["name"]
    log = f"{W}/logs/{name}.log"
    cfg = {"base_url": "http://127.0.0.1:8000", "model": "inkling", "health_timeout_s": 2400,
           "concurrency": 8, "prefill_lengths": [4096, 32768], "temperature": 1.0,
           "memory": {"cap_gib": 1000, "probe": ["bash", f"{W}/mem.sh"], "interval_s": 5}}
    cfg.update(defaults); cfg.update({k: v for k, v in run.items() if k not in ("cmd", "reps", "long")})
    p = f"{W}/configs/{name}.json"; json.dump(cfg, open(p, "w"), indent=1)  # no "launch": bench measures a warmed server
    print(f"=== RUN {name}: {run['cmd']}", flush=True)
    # Launch ourselves so an unmeasured warmup can cover Triton JIT / graph shapes before bench.py measures
    # (first-round vLLM prefill was polluted by "Triton kernel JIT compilation during inference").
    stop_server(); t0 = time.monotonic()
    SRV = subprocess.Popen(["bash", "-c", run["cmd"]], stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    srv = bench.Server(cfg); healthy = False
    while time.monotonic() - t0 < cfg["health_timeout_s"]:
        if srv.healthy(): healthy = True; break
        if SRV.poll() is not None: print(f"server exited with {SRV.returncode}", flush=True); break
        time.sleep(1)
    startup = round(time.monotonic() - t0, 1)
    extra = {"startup_s": startup}
    if not healthy:
        with open("ledger.jsonl", "a") as f: f.write(json.dumps({"name": name, "ok": False, "error": "server did not become healthy", **extra}) + "\n")
    else:
        tw = time.monotonic()
        try:
            for _ in range(2):
                srv.chat("Say hello.", 64)
                for L in cfg["prefill_lengths"]: srv.chat(bench.document(L, seed=999 + L)[0], 16, temperature=0.0)
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(cfg["concurrency"]) as ex: list(ex.map(lambda i: srv.chat(bench.PROMPTS[i % 6], 128), range(cfg["concurrency"])))
        except Exception as e: print("warmup error", repr(e), flush=True)
        extra["warmup_s"] = round(time.monotonic() - tw, 1)
        args = [sys.executable, BENCH, "run", p, "--reps", str(run.get("reps", defaults.get("reps", 6)))]
        if run.get("long"): args += ["--long", str(run["long"])]
        subprocess.run(args)
        # optional acceptance gates (skill: needle at ~50% / ~100% ctx, concurrent soak, greedy equivalence)
        for L in run.get("needles", []):
            try:
                prompt, codes = bench.document(L, seed=L + 7)
                r = srv.chat(prompt, 256, temperature=0.0)
                extra.setdefault("needle", []).append({"target": L, "prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 1),
                    "prefill": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None,
                    "retrieval": f"{sum(c in r['text'] for c in codes)}/{len(codes)}"})
            except Exception as e:
                extra.setdefault("needle", []).append({"target": L, "error": repr(e)[:300]})
            print("NEEDLE", json.dumps(extra["needle"][-1]), flush=True)
        if run.get("soak"):
            n, L = run["soak"]
            from concurrent.futures import ThreadPoolExecutor
            def one(i):
                prompt, codes = bench.document(L, seed=1000 + i)
                try:
                    r = srv.chat(prompt, 128, temperature=0.0); return {"ok": True, "prompt_tokens": r["prompt_tokens"], "retrieval": f"{sum(c in r['text'] for c in codes)}/{len(codes)}"}
                except Exception as e:
                    return {"ok": False, "error": repr(e)[:200]}
            ts = time.monotonic()
            with ThreadPoolExecutor(n) as ex: res = list(ex.map(one, range(n)))
            extra["soak"] = {"n": n, "target": L, "wall_s": round(time.monotonic() - ts, 1), "results": res}
            print("SOAK", json.dumps(extra["soak"]), flush=True)
        if run.get("greedy_dump"):
            outs = []
            for q in bench.PROMPTS:
                body = {"model": cfg["model"], "messages": [{"role": "user", "content": q}], "max_tokens": 256, "temperature": 0.0}
                req = urllib.request.Request(cfg["base_url"] + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                m = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]
                outs.append((m.get("reasoning_content") or m.get("reasoning") or "") + "\n<<>>\n" + (m.get("content") or ""))
            extra["greedy_outputs"] = outs
            print("GREEDY", json.dumps({"name": name, "outputs": outs}), flush=True)
        lines = open("ledger.jsonl").read().strip().splitlines(); d = json.loads(lines[-1]); d.update(extra)
        lines[-1] = json.dumps(d); open("ledger.jsonl", "w").write("\n".join(lines) + "\n")
    last = open("ledger.jsonl").read().strip().splitlines()[-1]
    print("LEDGER " + last, flush=True)
    if '"ok": false' in last or '"ok": true' not in last:
        print(f"--- tail of {log}", flush=True)
        print(subprocess.run(["tail", "-n", "40", log], capture_output=True, text=True).stdout, flush=True)
        print(f"--- error lines in {log}", flush=True)  # the root cause is usually in EngineCore/scheduler output far above the tail
        print(subprocess.run(["bash", "-c", f"grep -n -E 'Error|error|Exception|assert|not support|unsupported' {log} | grep -v -E '^[0-9]+:\\s+File ' | head -n 60"], capture_output=True, text=True).stdout, flush=True)
    else:  # keep startup diagnostics short
        print(subprocess.run(["bash", "-c", f"grep -E -i 'speculat|accept|cuda graph|KV cache|maximum concurrency|attention backend|Loading took|compil' {log} | tail -n 25"], capture_output=True, text=True).stdout, flush=True)
stop_server()
subprocess.run(["bash", "-c", "python3 -c \"import vllm; print('VERSION vllm', vllm.__version__)\" 2>/dev/null; python3 -c \"import sglang; print('VERSION sglang', sglang.__version__)\" 2>/dev/null"])
print("=== LEDGER TABLE"); subprocess.run([sys.executable, BENCH, "show", "ledger.jsonl"])

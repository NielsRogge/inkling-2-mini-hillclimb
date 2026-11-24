# Hill-climb driver that runs inside one HF Job (one GPU, one server at a time).
# PLAN (env, JSON list) = runs in order; each run: {"name", "cmd" (server shell command), optional bench overrides}.
# Uses the inference-optimisation skill's bench.py unchanged; prints each ledger line with a LEDGER marker.
import json, os, subprocess, sys, time, base64

W = "/work"; os.makedirs(f"{W}/logs", exist_ok=True); os.makedirs(f"{W}/configs", exist_ok=True)
os.chdir(W)
BENCH = f"{W}/bench.py"  # inference-optimisation skill scripts/bench.py, shipped verbatim in env
open(BENCH, "wb").write(base64.b64decode(os.environ["BENCH_B64"]))
import importlib.util
_sp = importlib.util.spec_from_file_location("bench", BENCH); bench = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bench)

open(f"{W}/stop.sh", "w").write("""#!/bin/bash
# drain first, then stop the server's process group; never pkill -f (could match this shell)
[ -f /work/srv.pid ] || exit 0
pid=$(cat /work/srv.pid)
kill -INT -- -$pid 2>/dev/null
for i in $(seq 1 90); do kill -0 -- -$pid 2>/dev/null || break; sleep 1; done
kill -9 -- -$pid 2>/dev/null
rm -f /work/srv.pid
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ] && break; sleep 1; done
""")
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
    launch = ["bash", "-c", f"setsid nohup bash -c {json.dumps(run['cmd'])} > {log} 2>&1 < /dev/null & echo $! > {W}/srv.pid"]
    p = f"{W}/configs/{name}.json"; json.dump(cfg, open(p, "w"), indent=1)  # no "launch": bench measures a warmed server
    print(f"=== RUN {name}: {run['cmd']}", flush=True)
    # Launch ourselves so an unmeasured warmup can cover Triton JIT / graph shapes before bench.py measures
    # (first-round vLLM prefill was polluted by "Triton kernel JIT compilation during inference").
    subprocess.run(["bash", f"{W}/stop.sh"]); t0 = time.monotonic(); subprocess.run(launch, check=True)
    srv = bench.Server(cfg); healthy = False
    while time.monotonic() - t0 < cfg["health_timeout_s"]:
        if srv.healthy(): healthy = True; break
        if subprocess.run(["bash", "-c", f"kill -0 -- -$(cat {W}/srv.pid)"], capture_output=True).returncode: break  # server died
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
        lines = open("ledger.jsonl").read().strip().splitlines(); d = json.loads(lines[-1]); d.update(extra)
        lines[-1] = json.dumps(d); open("ledger.jsonl", "w").write("\n".join(lines) + "\n")
    last = open("ledger.jsonl").read().strip().splitlines()[-1]
    print("LEDGER " + last, flush=True)
    if '"ok": false' in last or '"ok": true' not in last:
        print(f"--- tail of {log}", flush=True)
        print(subprocess.run(["tail", "-n", "80", log], capture_output=True, text=True).stdout, flush=True)
    else:  # keep startup diagnostics short
        print(subprocess.run(["bash", "-c", f"grep -E -i 'speculat|accept|cuda graph|KV cache|maximum concurrency|attention backend|Loading took|compil' {log} | tail -n 25"], capture_output=True, text=True).stdout, flush=True)
subprocess.run(["bash", f"{W}/stop.sh"])
print("=== LEDGER TABLE"); subprocess.run([sys.executable, BENCH, "show", "ledger.jsonl"])

# Run the skill's bench.py unchanged against an authenticated endpoint: injects "Authorization: Bearer $HF_TOKEN" into every request.
#   python3 jobs/bench_remote.py configs/endpoint.json [bench.py args...]
import os, sys, urllib.request, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_orig = urllib.request.Request
class AuthRequest(_orig):
    def __init__(self, url, *a, headers=None, **kw):
        headers = dict(headers or {}); headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        super().__init__(url, *a, headers=headers, **kw)
urllib.request.Request = AuthRequest
_urlopen = urllib.request.urlopen
urllib.request.urlopen = lambda req, *a, **kw: _urlopen(req if isinstance(req, _orig) else AuthRequest(req), *a, **kw)
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "..", "skill-scripts", "bench.py"))
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
a = sys.argv[1:]
opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
bench.run(a[0], long_ctx=int(opt("--long")) if opt("--long") else None, skip_cn="--skip-cN" in a, reps=int(opt("--reps", 6)))

#!/bin/bash
# usage: watch.sh JOBID...  — emits new LEDGER/Traceback/RUN-failure lines and terminal states
export PATH=/root/.local/bin:$PATH; cd /home/user/inkling-2-mini-hillclimb
declare -A seen fin
while true; do
  alive=0
  for j in "$@"; do
    [ -n "${fin[$j]}" ] && continue
    alive=1
    hf jobs logs --namespace huggingface $j > logs/$j.log 2>/dev/null || true
    n=$(grep -c -E '^LEDGER|Traceback|did not become healthy' logs/$j.log)
    if [ "${seen[$j]:-0}" != "$n" ]; then
      grep -E '^LEDGER|Traceback|did not become healthy' logs/$j.log | tail -n +$(( ${seen[$j]:-0} + 1 )) | sed "s/^/[$j] /" | cut -c1-500
      seen[$j]=$n
    fi
    st=$(hf jobs inspect --namespace huggingface $j --format json 2>/dev/null | grep -o '"stage": "[A-Z]*"' | head -1)
    case "$st" in *COMPLETED*|*ERROR*|*CANCELED*|*DELETED*) echo "[$j] TERMINAL $st"; fin[$j]=1;; esac
  done
  [ $alive = 0 ] && exit 0
  sleep 60
done

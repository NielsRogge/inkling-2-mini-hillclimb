#!/bin/bash
# usage: submit.sh FLAVOR IMAGE JOBNAME PLAN.json [DEFAULTS.json] [TIMEOUT]
set -e
export PATH=/root/.local/bin:$PATH
FL=$1 IMG=$2 NAME=$3 PLANF=$4 DEFF=${5:-} TO=${6:-3h}
DRV=$(base64 -w0 jobs/driver.py)
PLAN=$(base64 -w0 "$PLANF")
DEF=$( [ -n "$DEFF" ] && base64 -w0 "$DEFF" || echo e30=)
hf jobs run --namespace huggingface --flavor "$FL" --timeout "$TO" --detach --name "$NAME" --label project=inkling-hillclimb \
  --secrets HF_TOKEN -e PLAN="$PLAN" -e BENCH_B64="$(base64 -w0 skill-scripts/bench.py)" -e DEFAULTS="$DEF" -e HF_HUB_ENABLE_HF_TRANSFER=1 -e PYTHONUNBUFFERED=1 \
  "$IMG" bash -c "echo $DRV | base64 -d > /tmp/driver.py && python3 /tmp/driver.py"

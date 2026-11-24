#!/bin/bash
# Deploys the qualified config (see PLAN.md "Acceptance gates") as a NEW Inference Endpoint. Never touches existing endpoints.
set -e
export PATH=/root/.local/bin:$PATH
NAME=${1:-inkling2mini-tuned-dspark}
IMG=$(cat configs/IMAGE_PIN)
REV=7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6
hf endpoints deploy "$NAME" --namespace huggingface \
  --repo thinkingmachines/Inkling-2-Mini --revision $REV --task text-generation \
  --framework custom --custom-image "$IMG" --health-route /health --port 8000 \
  --vendor aws --region us-east-2 --accelerator gpu --instance-type nvidia-rtx-pro-6000 --instance-size x1 \
  --min-replica 0 --max-replica 1 --scale-to-zero-timeout 60 --type authenticated \
  --container-command "vllm serve" \
  --container-args "/repository --served-model-name thinkingmachines/Inkling-2-Mini --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --reasoning-parser inkling --max-model-len 262144 --gpu-memory-utilization 0.85 --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'"

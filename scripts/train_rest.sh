#!/bin/bash
# After the running Diffusion Policy job (422407) finishes, train the no-F/T ablation on the GPU.
cd "$(dirname "$0")/.."
PY=$HOME/workspace/langgrasp/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 422407 2>/dev/null; do sleep 15; done
$PY scripts/train.py --kind dp --name dp_noft --no-ft > logs/dp_noft.log 2>&1
echo done > logs/dp_noft.done

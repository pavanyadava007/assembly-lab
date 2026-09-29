#!/bin/bash
# Train ACT on the GPU once the no-F/T ablation has finished.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f logs/dp_noft.done ]; do sleep 20; done
$HOME/workspace/langgrasp/.venv/bin/python scripts/train.py --kind act --name act > logs/act.log 2>&1
echo done > logs/act.done

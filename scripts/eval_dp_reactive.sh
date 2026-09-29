#!/bin/bash
# Diffusion Policy with shorter open-loop execution (2 of each 16-step chunk) - runs after the ablation eval.
cd "$(dirname "$0")/.."
until grep -q "dp_noft evaluated" logs/eval_chain.progress 2>/dev/null; do sleep 15; done
for o in 0.0 0.002; do CUDA_VISIBLE_DEVICES= $HOME/workspace/langgrasp/.venv/bin/python scripts/evaluate.py --policy dp --n-action-steps 2 --n 200 --offset $o > logs/eval_dp_na2_$o.log 2>&1; done
echo done > logs/eval_dp_na2.done

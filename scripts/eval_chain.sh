#!/bin/bash
# Evaluate each checkpoint as soon as it is saved: 200 unseen seeds, nominal and 2 mm perception error.
cd "$(dirname "$0")/.."
PY=$HOME/workspace/langgrasp/.venv/bin/python
for p in dp dp_noft act; do
  until [ -f ckpt/$p/meta.json ]; do sleep 20; done
  for o in 0.0 0.002; do CUDA_VISIBLE_DEVICES= $PY scripts/evaluate.py --policy $p --n 200 --offset $o > logs/eval_${p}_$o.log 2>&1; done
  echo "$p evaluated" >> logs/eval_chain.progress
done
echo done > logs/eval_chain.done

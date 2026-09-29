#!/bin/bash
# Train the GPU policies one at a time (the GPU is shared with other jobs); logs in logs/.
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/workspace/langgrasp/.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ckpt/bc/model.pt ] || $PY scripts/train.py --kind bc --name bc > logs/bc.log 2>&1
run() {  # name, args...: wait for >= 1.5 GB free GPU memory, retry on OOM
  name=$1; shift
  for try in 1 2 3 4 5 6 7 8 9 10; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 700 ]; do sleep 20; done
    $PY scripts/train.py --name $name "$@" > logs/$name.log 2>&1 && return 0
    grep -q "OutOfMemory" logs/$name.log || return 1
    sleep 60
  done
}
run dp --kind dp
run dp_noft --kind dp --no-ft
run act --kind act
echo ALL_DONE > logs/train_all.done

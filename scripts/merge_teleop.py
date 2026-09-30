"""Merge human teleop episodes (data/teleop/*.npz) into data/demos_teleop.npz for scripts/train.py --data."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

files = sorted((ROOT / "data" / "teleop").glob("episode_*.npz"))
if not files:
    raise SystemExit("no teleop episodes in data/teleop - record some with scripts/teleop_server.py")
obs, act, ep, infos = [], [], [], []
for i, f in enumerate(files):
    d = np.load(f)
    obs.append(d["obs"])
    act.append(d["act"])
    ep.append(np.full(len(d["obs"]), i))
    infos.append(json.loads(str(d["info"])))
np.savez_compressed(
    ROOT / "data" / "demos_teleop.npz", obs=np.concatenate(obs), act=np.concatenate(act), episode=np.concatenate(ep)
)
meta = dict(
    episodes=len(files),
    frames=int(sum(len(o) for o in obs)),
    operator="human (keyboard teleoperation)",
    with_perception_error=sum(i["perception_offset_mm"] > 0 for i in infos),
    median_steps=float(np.median([i["steps"] for i in infos])),
)
(ROOT / "data" / "demos_teleop_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))

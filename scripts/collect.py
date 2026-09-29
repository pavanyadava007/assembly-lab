"""Collect scripted demonstrations (filtered to successful episodes) into data/demos.npz.

Seeds 0..N-1 are the demo seeds; evaluation uses seeds from 100000 upward, so no evaluation scene is ever
seen in training. Half of the demos have a perception error of 0-2.5 mm on the fixture position, so the data
contain force-guided recoveries.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_lab.env import AssemblyEnv, EpisodeConfig
from assembly_lab.expert import run_expert_episode

_env = None


def work(seed):
    global _env
    if _env is None:
        _env = AssemblyEnv()
    rng = np.random.default_rng(seed + 55_555)
    off = float(rng.uniform(0, 0.0025)) if rng.random() < 0.5 else 0.0
    obs_ep, act_ep, info = run_expert_episode(_env, EpisodeConfig(seed=seed, perception_offset=off))
    return obs_ep, act_ep, info


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    with Pool(30) as p:
        res = p.map(work, range(n), chunksize=4)
    keep = [r for r in res if r[2]["success"]]
    obs = np.concatenate([r[0] for r in keep])
    act = np.concatenate([r[1] for r in keep])
    ep = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(keep)])
    root = Path(__file__).resolve().parents[1]
    np.savez_compressed(root / "data" / "demos.npz", obs=obs, act=act, episode=ep)
    infos = [r[2] for r in res]
    meta = dict(
        episodes_run=n,
        episodes_kept=len(keep),
        frames=len(obs),
        with_perception_error=sum(i["perception_offset_mm"] > 0 for i in infos),
        kept_with_recovery=sum(r[2]["retries"] > 0 for r in keep),
        success_nominal=[
            sum(i["success"] for i in infos if i["perception_offset_mm"] == 0),
            sum(i["perception_offset_mm"] == 0 for i in infos),
        ],
        success_perturbed=[
            sum(i["success"] for i in infos if i["perception_offset_mm"] > 0),
            sum(i["perception_offset_mm"] > 0 for i in infos),
        ],
        seconds=round(time.time() - t0, 1),
        source="scripted expert (assembly_lab/expert.py), not human teleoperation",
    )
    (root / "data" / "demos_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

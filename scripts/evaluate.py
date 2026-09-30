"""Evaluate a checkpoint (or the scripted expert) on unseen seeds.

  python scripts/evaluate.py --policy dp --n 200 --offset 0.0     # nominal
  python scripts/evaluate.py --policy dp --n 200 --offset 0.002   # fixture perceived 2 mm off (recovery test)
  python scripts/evaluate.py --policy expert ...

Seeds start at 100000 (demos use 0..499). Results go to results/<policy>_off<mm>.json with every episode.
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED0 = 100_000
_env = _pol = None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def init(policy, nact=None, control="position"):
    global _env, _pol
    import torch

    torch.set_num_threads(1)
    from assembly_lab.env import AssemblyEnv

    _env = AssemblyEnv(control=control)
    if policy != "expert":
        from assembly_lab.policies import Runner

        _pol = Runner(ROOT / "ckpt" / policy, device="cpu", n_action_steps=nact)


def episode(args):
    seed, offset = args
    from assembly_lab.env import EpisodeConfig
    from assembly_lab.expert import run_expert_episode

    cfg = EpisodeConfig(seed=seed, perception_offset=offset)
    if _pol is None:
        return run_expert_episode(_env, cfg, record=False)
    import torch

    torch.manual_seed(seed)
    obs = _env.reset(cfg)
    _pol.reset()
    done = succ = False
    first_contact_step = None
    seated_step = None
    while not done:
        obs, succ, done = _env.step(_pol.act(obs))
        if first_contact_step is None and _env.peak_force > 5.0:
            first_contact_step = _env.t
        if seated_step is None and _env.seated():
            seated_step = _env.t
    exy, bottom, _ = _env.placement_error()
    return dict(
        seed=seed,
        success=bool(succ),
        seated=seated_step is not None,
        steps=_env.t,
        peak_force=float(_env.peak_force),
        err_xy_mm=float(np.linalg.norm(exy) * 1000),
        perception_offset_mm=offset * 1000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--control", default="position", choices=["position", "impedance"])
    ap.add_argument("--n-action-steps", type=int, default=None, help="override chunk execution length")
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(SEED0 + i, args.offset) for i in range(args.n)]
    init_args = (args.policy, args.n_action_steps, args.control)
    with get_context("spawn").Pool(args.procs, initializer=init, initargs=init_args) as p:
        eps = p.map(episode, jobs, chunksize=1)
    k = sum(e["success"] for e in eps)
    lo, hi = wilson(k, len(eps))
    succ = [e for e in eps if e["success"]]
    tag = args.policy + (f"_na{args.n_action_steps}" if args.n_action_steps else "")
    tag += "_imp" if args.control == "impedance" else ""
    summary = dict(
        control=args.control,
        policy=tag,
        n_action_steps=args.n_action_steps,
        perception_offset_mm=args.offset * 1000,
        n=len(eps),
        success=k,
        success_rate=k / len(eps),
        ci95=[round(lo, 4), round(hi, 4)],
        err_xy_mm_median=float(np.median([e["err_xy_mm"] for e in succ])) if succ else None,
        err_xy_mm_p95=float(np.percentile([e["err_xy_mm"] for e in succ], 95)) if succ else None,
        peak_force_median_N=float(np.median([e["peak_force"] for e in eps])),
        steps_median_success=float(np.median([e["steps"] for e in succ])) if succ else None,
        seconds=round(time.time() - t0, 1),
        hardware=f"CPU, {args.procs} processes (simulation)",
    )
    print(json.dumps(summary, indent=2))
    out = ROOT / "results" / f"{tag}_off{args.offset * 1000:.1f}.json"
    out.write_text(json.dumps(dict(summary=summary, episodes=eps), indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

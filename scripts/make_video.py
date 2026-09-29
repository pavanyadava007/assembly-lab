"""Render one episode (expert or a checkpoint) to media/<name>_seed<seed>_off<mm>.mp4 with two camera views."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly_lab.env import AssemblyEnv, EpisodeConfig
from assembly_lab.expert import Expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--offset", type=float, default=0.0)
    args = ap.parse_args()
    env = AssemblyEnv()
    obs = env.reset(EpisodeConfig(seed=args.seed, perception_offset=args.offset))
    if args.policy == "expert":
        ex = Expert(env, np.random.default_rng(args.seed + 10_000))
        act = ex.act
    else:
        from assembly_lab.policies import Runner

        pol = Runner(ROOT / "ckpt" / args.policy, device="cpu")
        pol.reset()
        act = pol.act
    r = mujoco.Renderer(env.model, 360, 480)
    frames, done, succ = [], False, False
    while not done:
        obs, succ, done = env.step(act(obs))
        f = np.concatenate([env.render(r, "front"), env.render(r, "close")], axis=1)
        frames.append(f)
    for _ in range(10):
        frames.append(frames[-1])
    out = ROOT / "media" / f"{args.policy}_seed{args.seed}_off{args.offset * 1000:.0f}mm.mp4"
    out.parent.mkdir(exist_ok=True)
    imageio.mimwrite(out, frames, fps=20, quality=7, macro_block_size=8)
    print(out, "success", succ, "steps", env.t, "peak force %.1f N" % env.peak_force)


if __name__ == "__main__":
    main()

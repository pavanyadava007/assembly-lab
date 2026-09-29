"""Train one policy on data/demos.npz.

python scripts/train.py --kind bc  --name bc
python scripts/train.py --kind dp  --name dp
python scripts/train.py --kind dp  --name dp_noft --no-ft
python scripts/train.py --kind act --name act
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly_lab.policies import ENV_IDX, ROBOT_IDX, ROBOT_IDX_NOFT, MLPPolicy, Normalizer, make_lerobot


def windows(ep, n_before, n_after):
    """For every frame, indices of the frames t-n_before..t+n_after-1 clamped to its episode, and a pad mask."""
    T = len(ep)
    starts = np.zeros(T, int)
    ends = np.zeros(T, int)
    bounds = np.flatnonzero(np.diff(ep)) + 1
    s_list = np.concatenate([[0], bounds])
    e_list = np.concatenate([bounds, [T]])
    for s, e in zip(s_list, e_list, strict=True):
        starts[s:e] = s
        ends[s:e] = e - 1
    offs = np.arange(-n_before, n_after)
    idx = np.arange(T)[:, None] + offs[None]
    pad = (idx > ends[:, None]) | (idx < starts[:, None])
    idx = np.clip(idx, starts[:, None], ends[:, None])
    return idx, pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bc", "dp", "act"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-ft", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    use_ft = not args.no_ft
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(ROOT / "data" / "demos.npz")
    obs, act, ep = d["obs"].astype(np.float32), d["act"].astype(np.float32), d["episode"]
    norm = Normalizer(obs.mean(0), obs.std(0))
    on = norm(obs).astype(np.float32)
    steps = args.steps or {"bc": 30000, "dp": 30000, "act": 30000}[args.kind]
    out = ROOT / "ckpt" / args.name
    out.mkdir(parents=True, exist_ok=True)
    ridx = ROBOT_IDX if use_ft else ROBOT_IDX_NOFT

    if args.kind == "bc":
        X = torch.from_numpy(on if use_ft else on[:, :17]).to(dev)
        Y = torch.from_numpy(act).to(dev)
        model = MLPPolicy(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    else:
        model, cfg = make_lerobot(args.kind, use_ft, dev)
        model.to(dev).train()
        if args.kind == "dp":
            o_idx, _ = windows(ep, cfg.n_obs_steps - 1, 1)
            a_idx, a_pad = windows(ep, cfg.n_obs_steps - 1, cfg.horizon - cfg.n_obs_steps + 1)
        else:
            o_idx = np.arange(len(ep))[:, None]
            a_idx, a_pad = windows(ep, 0, cfg.chunk_size)
        R = torch.from_numpy(on[:, ridx]).to(dev)
        E = torch.from_numpy(on[:, ENV_IDX]).to(dev)
        A = torch.from_numpy(act).to(dev)
        o_idx_t = torch.from_numpy(o_idx).to(dev)
        a_idx_t = torch.from_numpy(a_idx).to(dev)
        a_pad_t = torch.from_numpy(a_pad).to(dev)
        lr = 1e-4 if args.kind == "dp" else 1e-4
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6 if args.kind == "dp" else 1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    n = len(obs)
    t0 = time.time()
    log = []
    for it in range(1, steps + 1):
        b = torch.randint(0, n, (args.batch,), device=dev)
        if args.kind == "bc":
            loss = ((model(X[b]) - Y[b]) ** 2).mean()
        else:
            if args.kind == "dp":
                oi = o_idx_t[b]
                batch = {"observation.state": R[oi], "observation.environment_state": E[oi]}
            else:
                batch = {"observation.state": R[b], "observation.environment_state": E[b]}
            batch["action"] = A[a_idx_t[b]]
            batch["action_is_pad"] = a_pad_t[b]
            loss, _ = model.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == 1:
            log.append(dict(step=it, loss=float(loss), sec=round(time.time() - t0, 1)))
            print(log[-1], flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    params = sum(p.numel() for p in model.parameters())
    meta = dict(
        kind=args.kind,
        use_ft=use_ft,
        norm=norm.to_json(),
        steps=steps,
        batch=args.batch,
        seed=args.seed,
        params=params,
        train_seconds=round(time.time() - t0, 1),
        frames=n,
        episodes=int(ep.max() + 1),
        device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        log=log,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", out, "params", params)


if __name__ == "__main__":
    main()

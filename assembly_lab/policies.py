"""Policies trained on the scripted demonstrations: MLP behavior cloning, LeRobot Diffusion Policy, LeRobot ACT.

Observation layout (assembly_lab.env.AssemblyEnv.obs, 23 dims):
  robot state   = tool position, tool yaw (sin, cos), gripper width, wrist wrench (6)  -> 12 dims (6 without F/T)
  environment   = part position, part yaw (sin 2a, cos 2a), fixture position, part minus tool  -> 11 dims
LeRobot 0.4.x does not normalise inside the policy, so observations are standardised here with the training
set's mean and std; actions are already in [-1, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROBOT_IDX = list(range(6)) + list(range(17, 23))
ROBOT_IDX_NOFT = list(range(6))
ENV_IDX = list(range(6, 17))
ACT_DIM = 5


class Normalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.maximum(np.asarray(std, np.float32), 1e-3)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def to_json(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, ACT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def make_lerobot(kind: str, use_ft: bool, device: str):
    from lerobot.configs.types import FeatureType, PolicyFeature

    rdim = len(ROBOT_IDX if use_ft else ROBOT_IDX_NOFT)
    inputs = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(rdim,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(len(ENV_IDX),)),
    }
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACT_DIM,))}
    if kind == "dp":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        cfg = DiffusionConfig(
            input_features=inputs,
            output_features=outputs,
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            down_dims=(128, 256, 512),
            noise_scheduler_type="DDIM",
            num_train_timesteps=100,
            num_inference_steps=10,
            device=device,
        )
        return DiffusionPolicy(cfg), cfg
    if kind == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = ACTConfig(
            input_features=inputs,
            output_features=outputs,
            chunk_size=20,
            n_action_steps=10,
            dim_model=256,
            n_heads=8,
            dim_feedforward=1024,
            n_encoder_layers=4,
            n_decoder_layers=1,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            device=device,
        )
        return ACTPolicy(cfg), cfg
    raise ValueError(kind)


class Runner:
    """Uniform act(obs) interface for evaluation."""

    def __init__(self, ckpt_dir: Path, device="cpu", n_action_steps: int | None = None):
        ckpt_dir = Path(ckpt_dir)
        meta = json.loads((ckpt_dir / "meta.json").read_text())
        self.kind, self.use_ft = meta["kind"], meta["use_ft"]
        self.norm = Normalizer(**meta["norm"])
        self.device = device
        if self.kind == "bc":
            self.model = MLPPolicy(23 if self.use_ft else 17)
        else:
            self.model, _ = make_lerobot(self.kind, self.use_ft, device)
        self.model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
        self.model.to(device).eval()
        if n_action_steps is not None and self.kind != "bc":
            # execute fewer actions of each predicted chunk before re-planning (more reactive in contact)
            self.model.config.n_action_steps = n_action_steps
            self.model.reset()

    def reset(self):
        if self.kind != "bc":
            self.model.reset()

    def _split(self, o):
        on = self.norm(o)
        ridx = ROBOT_IDX if self.use_ft else ROBOT_IDX_NOFT
        return on, on[ridx], on[ENV_IDX]

    @torch.no_grad()
    def act(self, obs):
        on, r, e = self._split(obs)
        if self.kind == "bc":
            x = on if self.use_ft else np.concatenate([on[:17]])
            a = self.model(torch.from_numpy(x[None]).to(self.device))[0].cpu().numpy()
        else:
            batch = {
                "observation.state": torch.from_numpy(r[None]).to(self.device),
                "observation.environment_state": torch.from_numpy(e[None]).to(self.device),
            }
            a = self.model.select_action(batch)[0].cpu().numpy()
        a = np.clip(a, -1, 1)
        a[4] = 1.0 if a[4] > 0 else -1.0
        return a

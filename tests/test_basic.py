import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assembly_lab.env import AssemblyEnv, EpisodeConfig
from assembly_lab.expert import run_expert_episode


def test_env_obs_shape_and_ft_bias():
    env = AssemblyEnv()
    o = env.reset(EpisodeConfig(seed=3))
    assert o.shape == (AssemblyEnv.obs_dim,)
    assert np.all(np.abs(o[17:23]) < 0.5)  # wrench is bias-corrected at reset


def test_expert_nominal_succeeds():
    env = AssemblyEnv()
    info = run_expert_episode(env, EpisodeConfig(seed=100000), record=False)
    assert info["success"] and info["err_xy_mm"] < 2.0


def test_mcnemar_exact():
    from make_results import mcnemar

    a = {i: i < 10 for i in range(20)}
    b = {i: False for i in range(20)}
    n01, n10, p = mcnemar(a, b)
    assert (n01, n10) == (10, 0) and abs(p - 2 / 1024) < 1e-12


def test_wilson_bounds():
    from evaluate import wilson

    lo, hi = wilson(179, 200)
    assert round(lo, 4) == 0.8448 and round(hi, 4) == 0.9303

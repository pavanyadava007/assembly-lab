"""Scripted demonstrator: pick, align, force-guarded insertion with recovery, release.

The expert aims at the OBSERVED fixture position (the same, possibly wrong, perception the policy sees). When the
part lands on a pocket wall instead of dropping in (wrist force up, part still above the floor) it backs off
2 mm, shifts 1.5 mm away from the wall it touched and tries again. The contact side is read from the simulator's
contact list; a learned policy has to infer it from the wrist torque, which is exactly what the force-torque
ablation tests. Demonstrations are scripted, not human teleoperation.
"""

from __future__ import annotations

import numpy as np

from .env import ECU_HALF, MAX_DPOS, MAX_DYAW, POCKET_DEPTH, AssemblyEnv, EpisodeConfig

LIFT_Z = 0.11
HOVER_CLEAR = 0.006  # part bottom above wall top before the insertion descent
FLOOR_Z = 0.001 + ECU_HALF[2]  # part centre height when seated
# Grasp above the part centre: at 0.009 the pads caught only the top edge (25 deg tilt);
# 0.006 keeps the pads on the part and the fingertips clear of the pocket walls.
GRASP_DZ = 0.006


def _fold(a):
    """pi-periodic angle error folded into [-pi/2, pi/2)."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


class Expert:
    def __init__(self, env: AssemblyEnv, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.phase = "approach"
        self.k = 0
        self.correction = np.zeros(2)
        self.retries = 0
        self.stuck = 0

    def _move(self, target_xyz, target_yaw=None, gain=1.0, max_speed=1.0):
        env = self.env
        tp, ty = env.tcp_pose()
        # command relative to the internal target so servo lag is compensated by the feedback on the true pose
        err = np.asarray(target_xyz) - tp
        a = np.zeros(5)
        if env.control == "impedance":
            # the compliant arm lags its commanded point: steer the commanded point, plus a small correction on
            # the real tool error (adding the full tool error every step winds up into an oscillation)
            cmd = (np.asarray(target_xyz) - env.target_pos) + 0.15 * err
            a[:3] = np.clip(gain * cmd / MAX_DPOS, -max_speed, max_speed)
            if target_yaw is not None:
                a[3] = np.clip((_fold(target_yaw - env.target_yaw) + 0.15 * _fold(target_yaw - ty)) / MAX_DYAW, -1, 1)
        else:
            a[:3] = np.clip(gain * err / MAX_DPOS, -max_speed, max_speed)
            if target_yaw is not None:
                a[3] = np.clip(_fold(target_yaw - ty) / MAX_DYAW, -1, 1)
        return a, np.linalg.norm(err)

    def act(self, obs):
        env = self.env
        tp, ty = env.tcp_pose()
        ep_true, ey_true, _ = env.ecu_pose()
        # perceived quantities (what the policy sees) drive the approach and alignment
        ep = obs[6:9].astype(float)
        fp = obs[11:14].astype(float)
        grip_yaw = ey_true + np.pi / 2
        a = np.zeros(5)
        self.k += 1

        if self.phase == "approach":
            a, e = self._move([ep[0], ep[1], LIFT_Z], grip_yaw)
            a[4] = -1
            yaw_ok = abs(_fold(grip_yaw - ty)) < 0.03
            if e < 0.004 and yaw_ok:
                self.phase, self.k = "descend", 0
        elif self.phase == "descend":
            a, e = self._move([ep[0], ep[1], ep[2] + GRASP_DZ], grip_yaw, max_speed=0.6)
            a[4] = -1
            if e < 0.003 or self.k > 60:
                self.phase, self.k = "close", 0
        elif self.phase == "close":
            a, _ = self._move([ep[0], ep[1], ep[2] + GRASP_DZ], grip_yaw, max_speed=0.3)
            a[4] = 1
            if self.k >= 10:
                self.phase, self.k = "lift", 0
        elif self.phase == "lift":
            a, e = self._move([tp[0], tp[1], LIFT_Z], None)
            a[4] = 1
            if tp[2] > LIFT_Z - 0.01:
                self.phase, self.k = "transfer", 0
        elif self.phase in ("transfer", "align", "insert", "slide", "backoff"):
            a = self._place(obs, tp, ty, ep, fp)
        elif self.phase == "release":
            a[4] = -1
            if self.k >= 8:
                self.phase, self.k = "retreat", 0
        elif self.phase == "retreat":
            a, _ = self._move([tp[0], tp[1], tp[2] + 0.05], None)
            a[4] = -1
        return np.clip(a, -1, 1)

    def _place(self, obs, tp, ty, ep, fp):
        env = self.env
        _, ey_true, _ = env.ecu_pose()
        # target: part centre over the (perceived) pocket centre plus any learned correction, part yaw -> 0 (mod pi)
        goal_xy = fp[:2] + self.correction
        raw_err = goal_xy - ep[:2]
        # low-pass the perceived error (0.5 mm pose noise per frame); reset when the goal jumps
        if getattr(self, "_err_f", None) is None or self._goal_changed(goal_xy):
            self._err_f = raw_err
        self._err_f = 0.6 * self._err_f + 0.4 * raw_err
        part_err_xy = self._err_f
        yaw_goal = ty - _fold(ey_true)
        hover_z = tp[2] + ((POCKET_DEPTH + HOVER_CLEAR + ECU_HALF[2]) - ep[2])
        a = np.zeros(5)
        a[4] = 1
        if self.phase == "transfer":
            tgt = np.array([tp[0] + part_err_xy[0], tp[1] + part_err_xy[1], LIFT_Z])
            a, e = self._move(tgt, yaw_goal)
            a[4] = 1
            if np.linalg.norm(part_err_xy) < 0.006:
                self.phase, self.k = "align", 0
        elif self.phase == "align":
            tgt = np.array([tp[0] + part_err_xy[0], tp[1] + part_err_xy[1], hover_z])
            a, e = self._move(tgt, yaw_goal, max_speed=0.5)
            a[4] = 1
            if (
                np.linalg.norm(part_err_xy) < 0.0006
                and abs(_fold(ey_true)) < 0.01
                and abs(ep[2] - (POCKET_DEPTH + HOVER_CLEAR + ECU_HALF[2])) < 0.002
            ):
                self.phase, self.k, self.stuck = "insert", 0, 0
            if self.k > 80:  # give up aligning perfectly, try anyway
                self.phase, self.k, self.stuck = "insert", 0, 0
        elif self.phase == "insert":
            # one compliant phase: descend gently; while the part touches a wall or the chamfer, move away from the
            # contact and fold that move into the goal (the policy has to infer the contact side from wrist torque)
            ep_true = env.ecu_pose()[0]
            side = self._contact_side()
            a[3] = np.clip(_fold(-ey_true) / MAX_DYAW, -0.3, 0.3)
            if side.any():
                a[:2] = -side * 0.05
                a[2] = -0.08
                self.correction = self.correction - side * 0.0005
                self.stuck += 1
            else:
                a[:2] = np.clip(0.5 * part_err_xy / MAX_DPOS, -0.1, 0.1)
                a[2] = -0.2
            fmag = np.linalg.norm(env.wrench()[:3])
            if fmag > 50.0:  # force limit: back off instead of pushing harder
                a[2] = 0.1
            if ep_true[2] < FLOOR_Z + 0.0015:
                self.phase, self.k = "release", 0
            elif self.stuck > 30 or self.k > 90:  # wedged: lift to hover and retry with the correction kept
                self.retries += 1
                self.phase, self.k = "backoff", 0
            self.retries = max(self.retries, int(self.stuck > 0))
        elif self.phase == "backoff":
            a[2] = 0.3
            self.stuck = 0
            if self.k >= 6:
                self.phase, self.k = "align", 0
        return a

    def _goal_changed(self, goal_xy):
        last = getattr(self, "_last_goal", None)
        self._last_goal = goal_xy.copy()
        return last is None or np.linalg.norm(goal_xy - last) > 1e-4

    def _contact_side(self):
        """Unit xy vector from the part centre towards the wall contact (0 if none)."""
        env = self.env
        d = env.data
        ep = env.ecu_pose()[0]
        pts = []
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if env.ecu_geom in pair and pair & env.side_geoms:  # walls and chamfers, not the pocket floor
                pts.append(c.pos[:2] - ep[:2])
        if not pts:
            return np.zeros(2)
        v = np.mean(pts, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0.005 else np.zeros(2)  # contacts all round (seated) -> no push


def run_expert_episode(env: AssemblyEnv, cfg: EpisodeConfig, record=True):
    obs = env.reset(cfg)
    ex = Expert(env, np.random.default_rng(cfg.seed + 10_000))
    obs_log, act_log = [], []
    done = succ = False
    while not done:
        a = ex.act(obs)
        if record:
            obs_log.append(obs)
            act_log.append(a.astype(np.float32))
        obs, succ, done = env.step(a)
    exy, bottom, _ = env.placement_error()
    info = dict(
        seed=cfg.seed,
        success=bool(succ),
        steps=env.t,
        retries=ex.retries,
        peak_force=float(env.peak_force),
        err_xy_mm=float(np.linalg.norm(exy) * 1000),
        perception_offset_mm=cfg.perception_offset * 1000,
        final_phase=ex.phase,
    )
    return (np.array(obs_log), np.array(act_log), info) if record else info

"""ECU-style part insertion on a simulated Franka Panda (MuJoCo Menagerie model).

Task, modelled on the kind of station BMW has shown publicly (an electronic control unit placed on a bracket):
pick a 10 x 6 x 3 cm box ("ECU", 0.3 kg) from the table and insert it into a locating fixture, an open pocket
whose inner size leaves CLEARANCE per side (1 mm default). The pocket has no chamfer, so a part that arrives
a few millimetres off lands on the wall edge: the task is contact-rich and the last millimetres decide it.

Control: the policy outputs a Cartesian delta for the tool centre point (dx, dy, dz, dyaw) plus a gripper
command at CONTROL_HZ. Deltas are turned into joint position targets by damped least-squares inverse kinematics
(tool kept vertical), i.e. the same interface as a Cartesian teleoperation or OSC_POSE controller.

Sensing: a 6-axis force-torque sensor sits between flange and hand (site "ft"). Observations are low-dimensional
(tool pose, gripper width, part pose, fixture pose, filtered wrist wrench); fixture pose can be corrupted by a
perception offset to test recovery.

Success: the part rests inside the pocket (bottom within 2 mm of the pocket floor, centre within the clearance)
and the gripper has released it and moved at least 3 cm away. Everything here is simulation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "models" / "menagerie" / "franka_emika_panda"

CONTROL_HZ = 20
PHYS_DT = 0.002
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYS_DT)))
MAX_STEPS = 400  # 20 s

ECU_HALF = np.array(
    [0.03, 0.05, 0.02]
)  # 6 x 10 x 4 cm; gripper closes across the 6 cm side, fingers stay 6.5 mm above the walls
ECU_MASS = 0.3
POCKET_DEPTH = 0.012
CHAMFER = 0.002  # lead-in on the pocket's inner top edge
WALL_T = 0.006
TABLE_Z = 0.0

# action limits per control step
MAX_DPOS = 0.01  # 1 cm per step = 0.2 m/s
MAX_DYAW = 0.08


def _scene_xml(clearance: float) -> str:
    """Pocket = floor + four walls with a CHAMFER mm lead-in on the inner top edge (walls step out by CHAMFER
    above POCKET_DEPTH - CHAMFER, and a 45 degree plate bridges the step)."""
    inner = ECU_HALF[:2] + clearance
    ox, oy = inner
    wx = ox + WALL_T
    c = CHAMFER
    lo_h = (POCKET_DEPTH - c) / 2
    hi_zc = POCKET_DEPTH - c / 2
    t = 0.0005
    g = []
    Ly = oy + WALL_T
    for sgn in (1, -1):
        # x walls: lower full-thickness part, upper part stepped out by c, chamfer plate
        g.append((f"{sgn * (ox + WALL_T / 2):.5f} 0 {lo_h:.5f}", f"{WALL_T / 2:.5f} {Ly:.5f} {lo_h:.5f}", "0 0 0"))
        g.append(
            (
                f"{sgn * (ox + c + (WALL_T - c) / 2):.5f} 0 {hi_zc:.5f}",
                f"{(WALL_T - c) / 2:.5f} {Ly:.5f} {c / 2:.5f}",
                "0 0 0",
            )
        )
        nx, nz = -sgn * 0.70711, 0.70711
        g.append(
            (
                f"{sgn * (ox + c / 2) - nx * t:.5f} 0 {hi_zc - nz * t:.5f}",
                f"{c * 0.70711:.5f} {Ly:.5f} {t:.5f}",
                f"0 {-sgn * 0.785398:.6f} 0",
            )
        )
        # y walls (between the x walls)
        g.append((f"0 {sgn * (oy + WALL_T / 2):.5f} {lo_h:.5f}", f"{ox:.5f} {WALL_T / 2:.5f} {lo_h:.5f}", "0 0 0"))
        g.append(
            (
                f"0 {sgn * (oy + c + (WALL_T - c) / 2):.5f} {hi_zc:.5f}",
                f"{ox + c:.5f} {(WALL_T - c) / 2:.5f} {c / 2:.5f}",
                "0 0 0",
            )
        )
        ny = -sgn * 0.70711
        g.append(
            (
                f"0 {sgn * (oy + c / 2) - ny * t:.5f} {hi_zc - nz * t:.5f}",
                f"{ox + c:.5f} {c * 0.70711:.5f} {t:.5f}",
                f"{sgn * 0.785398:.6f} 0 0",
            )
        )
    wall_xml = "\n".join(
        f'<geom name="wall{i}" type="box" pos="{p}" size="{sz}" euler="{e}" material="fixture" friction="0.4 0.005 0.0001"/>'
        for i, (p, sz, e) in enumerate(g)
    )
    oy_full = Ly
    return f"""
<mujoco model="assembly">
  <include file="{PANDA_DIR / "panda.xml"}"/>
  <option timestep="{PHYS_DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="960" offheight="720"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.22 0.25 0.3" rgb2="0.18 0.2 0.24" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="6 6"/>
    <material name="fixture" rgba="0.55 0.57 0.6 1"/>
    <material name="ecu" rgba="0.12 0.35 0.7 1"/>
  </asset>
  <worldbody>
    <light pos="0.4 0 1.6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" material="floor" friction="0.8 0.005 0.0001"/>
    <body name="fixture" pos="0.55 0.15 0" mocap="true">
      <geom name="pocket_floor" type="box" pos="0 0 0.0005" size="{wx:.5f} {oy_full:.5f} 0.0005" material="fixture"/>
      {wall_xml}
      <site name="pocket" pos="0 0 0.001" size="0.004" rgba="0 1 0 0.4"/>
    </body>
    <body name="ecu" pos="0.5 -0.15 {ECU_HALF[2]:.4f}">
      <freejoint name="ecu"/>
      <geom name="ecu" type="box" size="{ECU_HALF[0]} {ECU_HALF[1]} {ECU_HALF[2]}" mass="{ECU_MASS}" material="ecu"
            friction="1.0 0.01 0.001" condim="6"/>
      <site name="ecu_c" size="0.003" rgba="1 0 0 0.5"/>
    </body>
    <camera name="front" pos="1.25 -0.05 0.55" xyaxes="0.05 1 0 -0.35 0.02 0.94" fovy="45"/>
    <camera name="close" pos="0.92 0.15 0.26" xyaxes="0 1 0 -0.55 0 0.84" fovy="38"/>
  </worldbody>
  <sensor>
    <force name="ft_force" site="ft"/>
    <torque name="ft_torque" site="ft"/>
  </sensor>
</mujoco>"""


def build_model(clearance: float = 0.001) -> mujoco.MjModel:
    """Load the Menagerie Panda, add a wrist F/T site + TCP site, and the fixture/part scene."""
    spec_xml = (PANDA_DIR / "panda.xml").read_text()
    # wrist force-torque site between flange and hand, and a TCP site between the fingertips
    spec_xml = spec_xml.replace(
        '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
        '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">\n'
        '<site name="ft" pos="0 0 0" size="0.005" rgba="1 1 0 0.3"/>\n'
        '<site name="tcp" pos="0 0 0.1034" size="0.004" rgba="1 0 1 0.4"/>',
    )
    # Menagerie's gripper servo (kp 100 N/m on the split tendon) squeezes a 6 cm part with only ~3 N, less than a
    # 0.3 kg part needs; kp 1000 gives ~30 N, inside the real Franka Hand's 70 N continuous rating.
    spec_xml = spec_xml.replace(
        'gainprm="0.01568627451 0 0" biasprm="0 -100 -10"', 'gainprm="0.1568627451 0 0" biasprm="0 -1000 -30"'
    )
    spec_xml = spec_xml.replace('meshdir="assets"', f'meshdir="{PANDA_DIR / "assets"}"')
    tmp = PANDA_DIR / f"_panda_ft_{os.getpid()}.xml"  # per process: parallel workers must not share it
    tmp.write_text(spec_xml)
    try:
        xml = _scene_xml(clearance).replace(str(PANDA_DIR / "panda.xml"), str(tmp))
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class EpisodeConfig:
    seed: int = 0
    clearance: float = 0.001
    perception_offset: float = 0.0  # metres, applied to the OBSERVED fixture xy in a random direction
    ecu_xy_range: float = 0.04
    ecu_yaw_range: float = 0.35
    fixture_xy_range: float = 0.02
    ft_noise: float = 0.3  # N, per-step sensor noise
    obs_noise: float = 0.0005  # m, part/fixture pose noise (perception)


def yaw_of(mat9) -> float:
    m = np.asarray(mat9).reshape(3, 3)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class AssemblyEnv:
    obs_dim = 23
    act_dim = 5  # dx, dy, dz, dyaw, gripper (1 = close, -1 = open)

    def __init__(self, clearance: float = 0.001):
        self.model = build_model(clearance)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.clearance = clearance
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.jlo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.jhi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.tcp = m.site("tcp").id
        self.ecu_body = m.body("ecu").id
        self.ecu_qadr = m.jnt_qposadr[m.joint("ecu").id]
        self.fix_mocap = m.body_mocapid[m.body("fixture").id]
        self.pocket_site = m.site("pocket").id
        self.ft_f = m.sensor_adr[m.sensor("ft_force").id]
        self.ft_t = m.sensor_adr[m.sensor("ft_torque").id]
        self.lfinger = m.body("left_finger").id
        self.rfinger = m.body("right_finger").id
        self.ecu_geom = m.geom("ecu").id
        self.side_geoms = {m.geom(f"wall{i}").id for i in range(12)}
        self.wall_geoms = self.side_geoms | {m.geom("pocket_floor").id}
        self.home = m.key("home").qpos.copy()

    # ------------------------------------------------------------------ reset / step
    def reset(self, cfg: EpisodeConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:9] = self.home[:9]
        d.ctrl[:7] = self.home[:7]
        d.ctrl[7] = 255
        # part pose
        ex, ey = 0.50 + rng.uniform(-1, 1) * cfg.ecu_xy_range, -0.15 + rng.uniform(-1, 1) * cfg.ecu_xy_range
        eyaw = rng.uniform(-1, 1) * cfg.ecu_yaw_range
        d.qpos[self.ecu_qadr : self.ecu_qadr + 3] = [ex, ey, ECU_HALF[2] + 0.0005]
        d.qpos[self.ecu_qadr + 3 : self.ecu_qadr + 7] = [np.cos(eyaw / 2), 0, 0, np.sin(eyaw / 2)]
        # fixture pose (true) and observed offset
        fx, fy = 0.55 + rng.uniform(-1, 1) * cfg.fixture_xy_range, 0.15 + rng.uniform(-1, 1) * cfg.fixture_xy_range
        d.mocap_pos[self.fix_mocap] = [fx, fy, 0]
        d.mocap_quat[self.fix_mocap] = [1, 0, 0, 0]
        ang = rng.uniform(0, 2 * np.pi)
        self.fix_obs_offset = cfg.perception_offset * np.array([np.cos(ang), np.sin(ang)])
        mujoco.mj_forward(m, d)
        for _ in range(100):  # settle
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.tcp].copy()
        self.target_yaw = yaw_of(d.site_xmat[self.tcp])
        self.grip_cmd = -1.0
        self.ft_bias = self._raw_wrench()
        self.ft_filt = np.zeros(6)
        self.t = 0
        self.peak_force = 0.0
        self.released_ok = False
        return self.obs()

    def _raw_wrench(self):
        d = self.data
        return np.concatenate([d.sensordata[self.ft_f : self.ft_f + 3], d.sensordata[self.ft_t : self.ft_t + 3]]).copy()

    def wrench(self):
        """Filtered wrist wrench in the sensor frame, gravity/tool bias removed at reset."""
        return self.ft_filt.copy()

    def _ik(self, pos, yaw, iters=3):
        m, d = self.model, self.data
        q = d.qpos[self.arm_qadr].copy()
        target_R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.diag(
            [1, -1, -1]
        )
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        dtmp = self._ik_data
        dtmp.qpos[:] = d.qpos
        for _ in range(iters):
            dtmp.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(m, dtmp)
            mujoco.mj_comPos(m, dtmp)
            cur_p = dtmp.site_xpos[self.tcp]
            cur_R = dtmp.site_xmat[self.tcp].reshape(3, 3)
            ep = pos - cur_p
            eR = 0.5 * (
                np.cross(cur_R[:, 0], target_R[:, 0])
                + np.cross(cur_R[:, 1], target_R[:, 1])
                + np.cross(cur_R[:, 2], target_R[:, 2])
            )
            mujoco.mj_jacSite(m, dtmp, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
            e = np.concatenate([ep, eR])
            lam = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            # null-space pull towards home posture keeps the elbow sensible
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (self.home[:7] - q))
            q = np.clip(q + dq, self.jlo, self.jhi)
        return q

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        m, d = self.model, self.data
        if not hasattr(self, "_ik_data"):
            self._ik_data = mujoco.MjData(m)
        self.target_pos = self.target_pos + a[:3] * MAX_DPOS
        # admittance-style limit: the commanded point may lead the real tool by at most 6 mm downward and 2 cm
        # overall, so a blocked descent builds bounded force instead of driving the arm through the fixture
        tcp_now = self.data.site_xpos[self.tcp]
        self.target_pos[2] = max(self.target_pos[2], tcp_now[2] - 0.006, 0.012)
        self.target_pos = tcp_now + np.clip(self.target_pos - tcp_now, -0.02, 0.02)
        self.target_pos[:2] = np.clip(self.target_pos[:2], [0.25, -0.45], [0.8, 0.45])
        self.target_yaw = wrap(self.target_yaw + a[3] * MAX_DYAW)
        self.grip_cmd = a[4]
        q = self._ik(self.target_pos, self.target_yaw)
        d.ctrl[:7] = q
        d.ctrl[7] = 0.0 if self.grip_cmd > 0 else 255.0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
            f = np.linalg.norm(self._contact_force_ecu_fixture())
            self.peak_force = max(self.peak_force, f)
        raw = self._raw_wrench() - self.ft_bias
        self.ft_filt = 0.5 * self.ft_filt + 0.5 * (raw + self.rng.normal(0, self.cfg.ft_noise, 6))
        self.t += 1
        succ = self.success()
        done = succ or self.t >= MAX_STEPS
        return self.obs(), succ, done

    # ------------------------------------------------------------------ state
    def tcp_pose(self):
        d = self.data
        return d.site_xpos[self.tcp].copy(), yaw_of(d.site_xmat[self.tcp])

    def ecu_pose(self):
        d = self.data
        return d.xpos[self.ecu_body].copy(), yaw_of(d.xmat[self.ecu_body]), d.xmat[self.ecu_body].reshape(3, 3).copy()

    def fixture_true(self):
        return self.data.mocap_pos[self.fix_mocap].copy()

    def fixture_obs(self):
        p = self.fixture_true()
        p[:2] += self.fix_obs_offset
        return p

    def gripper_width(self):
        d = self.data
        return float(d.qpos[7] + d.qpos[8])

    def obs(self):
        rng = self.rng
        tp, ty = self.tcp_pose()
        ep, ey, _ = self.ecu_pose()
        fp = self.fixture_obs()
        n = self.cfg.obs_noise
        ep_o = ep + rng.normal(0, n, 3)
        fp_o = fp + rng.normal(0, n, 3)
        w = self.wrench()
        yaw_rel = wrap(ey - ty)
        # ECU yaw is symmetric under pi for this task: fold into [-pi/2, pi/2)
        yaw_rel = (yaw_rel + np.pi / 2) % np.pi - np.pi / 2
        return np.concatenate(
            [
                tp,  # 0:3 tool position
                [np.sin(ty), np.cos(ty)],  # 3:5 tool yaw
                [self.gripper_width()],  # 5
                ep_o,  # 6:9 part position (perceived)
                [np.sin(2 * yaw_rel), np.cos(2 * yaw_rel)],  # 9:11 part yaw relative to tool (pi-symmetric)
                fp_o,  # 11:14 fixture position (perceived)
                ep_o - tp,  # 14:17 part relative to tool
                w / np.array([20, 20, 20, 2, 2, 2]),  # 17:23 wrist wrench, scaled
            ]
        ).astype(np.float32)

    def _contact_force_ecu_fixture(self):
        m, d = self.model, self.data
        tot = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if self.ecu_geom in pair and pair & self.wall_geoms:
                mujoco.mj_contactForce(m, d, i, f6)
                tot += np.abs(f6[:3])
        return tot

    def placement_error(self):
        """ECU centre vs pocket centre (xy, metres) and bottom height above pocket floor."""
        ep, _, R = self.ecu_pose()
        pk = self.data.site_xpos[self.pocket_site]
        bottom = ep[2] - ECU_HALF[2] * abs(R[2, 2]) - pk[2] + 0.001
        return ep[:2] - pk[:2], bottom, R

    def seated(self):
        exy, bottom, R = self.placement_error()
        flat = R[2, 2] > 0.995
        return bool(flat and bottom < 0.003 and np.all(np.abs(exy) < 0.004))

    def success(self):
        tp, _ = self.tcp_pose()
        ep, _, _ = self.ecu_pose()
        released = self.gripper_width() > 0.064 and (tp[2] - ep[2]) > 0.03
        return self.seated() and released

    def render(self, renderer, camera="front"):
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = AssemblyEnv()
    o = env.reset(EpisodeConfig(seed=1))
    print("obs", o.shape, "substeps", SUBSTEPS, "tcp", env.tcp_pose(), "ecu", env.ecu_pose()[0], "fix", env.fixture_true())

# assembly-lab: learned contact-rich insertion on a simulated Franka Panda

Pick an ECU-sized part (6 x 10 x 4 cm, 0.3 kg) and insert it into a locating fixture with **1 mm clearance per
side** and a 2 mm lead-in chamfer, on a **Franka Emika Panda** in MuJoCo with a **wrist force-torque sensor**.
The task is modelled on the kind of station car makers have shown for learned assembly (an electronic control unit
placed on a bracket), where the first grasp is easy and the last millimetres decide success.

**Live page:** https://huggingface.co/spaces/pavanyadava07/assembly-lab (videos + results tables)

Simulation only: no real robot was used. Demonstrations come from a scripted, force-guided expert, not from human
teleoperation. Every number here is produced by the scripts from `results/*.json`; `docs/RESULTS.md` is generated.

## What is compared

| | |
|---|---|
| Scripted expert | pick, align on the perceived fixture pose, force-guided insertion with search and a 50 N force limit, release |
| MLP behavior cloning | 3 x 512 MLP, one action per step, 23-D state including the filtered wrist wrench |
| ACT (LeRobot 0.4.4) | action chunking transformer on the same state (chunk 20, execute 10) |
| Diffusion Policy (LeRobot 0.4.4) | 1-D conv UNet, 16-step horizon, DDIM with 10 denoising steps, 2 observation steps |
| Ablations | each learner without the wrist force-torque input; re-planning every 2 (DP) or 1 (ACT) actions |
| Controllers | IK position control (default) vs a torque-level Cartesian impedance controller |

Two perception conditions on 200 unseen seeds each: nominal (0.5 mm pose noise) and the fixture perceived **2 mm off**
in a random direction, so the part lands on the chamfer and the policy has to recover.

## Results (from docs/RESULTS.md)

- Scripted expert: 200/200 nominal, 165/200 with the 2 mm perception error.
- **MLP behavior cloning: 179/200 (89.5%) nominal, 84/200 with the 2 mm error.**
- **Wrist force-torque input matters:** without it the same MLP drops to 156/200 nominal (paired exact McNemar
  p = 1.4e-3) and 47/200 with the 2 mm error (p = 2.9e-5). Most failures without F/T never touch the fixture:
  the policy hovers above the pocket because nothing tells it that contact has happened.
- Diffusion Policy: 75/200 nominal and 63/200 with the 2 mm error; without F/T 55/200 (p = 0.012). Re-planning every
  2 actions instead of 8 gave the same 75/200, so chunked open-loop execution is not the cause; on this 23-D state
  task with 479 demonstrations the plain MLP is the better learner. Reported as found.
- ACT with LeRobot's default settings (chunk 20, KL weight 10, 30k steps) did not learn this state-only task:
  9/200 nominal, 10/200 when re-planning every step. It reaches the fixture but lands 2-30 mm off and pushes at a
  median 75 N. Not diagnosed further; reported as a negative result.
- **Cartesian impedance control** (joint torque control at 500 Hz, gravity/Coriolis compensation, null-space
  posture, laterally softer than vertical): the scripted force-guided insertion under the 2 mm perception error rose
  from 165/200 to **194/200** (paired McNemar p = 1.1e-6) at a lower median peak contact force (26.8 vs 45.8 N). The
  chamfer can now push the part into place. Behavior cloning of that compliant expert, however, reached only 51/200
  even after adding the commanded pose to the observation (a real Franka reports it): the action depends on
  controller state the single-step policy handles poorly. Open problem, reported as found.
- With F/T the main remaining MLP failure is different: the part is seated but the gripper does not open
  (16 of 21 nominal failures).

See `docs/RESULTS.md` for all rows, confidence intervals, final position error (median about 0.6 mm), peak
contact forces, and training cost.

## Engineering notes (what had to be fixed to make the task honest)

- Menagerie's gripper servo squeezes a 6 cm part with only about 3 N; raised to about 30 N (inside the real Franka
  Hand's 70 N continuous rating), otherwise the part slips out on lift.
- Grasping 9 mm above the part centre let the pads catch only the top edge and the part hung 25 degrees tilted;
  6 mm keeps the tilt under 1.4 degrees and the fingertips still clear the walls.
- The first recovery strategy (lift and retry in fixed 1.5 mm steps) overshot; sliding in contact oscillated. A
  single compliant insertion phase with a force limit and a wedge escape brought the expert to 200/200 nominal.
- The commanded tool point may lead the real one by at most 6 mm downward, an admittance-style limit, so a
  blocked descent builds bounded force.

## Record human demonstrations (teleoperation)

```bash
python scripts/teleop_server.py      # open http://localhost:8020 (VS Code forwards the port)
python scripts/merge_teleop.py       # data/teleop/*.npz -> data/demos_teleop.npz
python scripts/train.py --kind bc --name bc_teleop --data demos_teleop.npz
```

Keyboard control in the browser (W/S, A/D, R/F, Q/E, Shift for fine motion, Space for the gripper, Enter to save a
successful episode). Teleop scenes use seeds from 200000, so they never overlap the scripted demos or the evaluation.

## Run it

```bash
scripts/fetch_models.sh                       # Franka Panda from MuJoCo Menagerie (Apache-2.0), pinned commit
python scripts/collect.py 500                 # scripted demonstrations -> data/demos.npz
python scripts/train.py --kind bc --name bc   # also: --kind dp / act, --no-ft for the ablations
python scripts/evaluate.py --policy bc --n 200 --offset 0.002
python scripts/evaluate.py --policy expert --control impedance --n 200 --offset 0.002
python scripts/make_results.py                # docs/RESULTS.md
python scripts/make_video.py --policy bc --seed 100003 --offset 0.002
```

Environment: Python 3.10, MuJoCo 3.8, PyTorch 2.7, LeRobot 0.4.4. Training on one NVIDIA L4 shared with other jobs;
evaluation on CPU.

## Limits

- No real robot, no vision; the teleop tool exists but the reported policies use scripted demonstrations only (state input with simulated pose noise).
- The expert reads the contact side from the simulator; learned policies only see the wrist wrench.
- One task geometry and one clearance; the demonstrations are filtered to successful episodes.

MIT licence. The Panda model is from MuJoCo Menagerie (Apache-2.0).

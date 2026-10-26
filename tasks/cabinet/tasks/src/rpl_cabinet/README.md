# rpl_cabinet — asset, dataset notes, and known gaps

This directory holds the authored RPL storage-cabinet USD plus the task configs
that drive the OpenArm against it. The content here was lifted from the
`cabinet_sim/` prototype folder and reorganized into an Isaac Lab 3.0.0 task
package. The task-package entry points are in `openarm/`; this README covers
the **asset itself** and **what the real demonstrations say about the task**
— the two things that stay true across env-cfg edits.

## The cabinet

Global Industrial "Turn Handle" full-height storage cabinet, 36"W × 18"D × 78"H,
black, MPN **603599BK**. Measured in the RPL: **1980 × 917 × 458 mm**,
door leaves **421 mm**, handle pivot **1020 mm** above the floor.

### Measured dimensions, from the sketch (trusted)

| | mm |
|---|---|
| Overall H × W × D | 1980 × 917 × 458 |
| Door leaf width | 421 (×2) |
| Side frame border | 36 each side |
| Handle pivot height | 1020 |
| Handle inset from door free edge | 75 |
| Lever length | 110 |
| Door face out to the lever | 40 |
| Keyway face diameter on the handle head | 25 |

The centre gap of 3 mm falls out of the measurements rather than being
assumed: 36 + 421 + 421 + 36 = 914 against an overall 917. And the handle at
1020 lands 2 mm from the door's vertical centre at 1022, an independent check
that the vertical layout is close to right.

### Assumed dimensions (constants in `scripts/build_cabinet.py`)

| | mm | Note |
|---|---|---|
| `DOOR_T` | 20 | door slab thickness |
| `PANEL_T` | 2 | carcass sheet steel |
| `TOE_KICK_H` | 100 | recessed base height |
| `TOE_RECESS` | 25 | base setback from the front |
| `FRAME_TOP` | 36 | assumed equal to the side border |
| `N_SHELVES` | 4 | from the vendor photo |
| `ESC_T` | 8 | escutcheon plate thickness |

The handle reads as: a fixed chrome flange and neck standing 40 mm off the
door face, a 25 mm diameter round head that turns, carrying the keyway on its
outward face, with the 110 mm lever hanging off it. The flange and neck are
visual only; only the head and lever are a rigid body.

## The rig

```
/Cabinet                       articulation root
  body                         fixed to world, 60 kg
    side_left/right, top, back, floor, toe_kick,
    frame_left/right/top, shelf_1..4
  hinge_left   revolute Z about the left stile,  -110..0 deg,  9 kg
    (joins /Cabinet/body <-> /Cabinet/door_left, which carries slab + escutcheon)
  hinge_right  revolute Z about the right stile,    0..110 deg, 9 kg
  latch_left   revolute Y at the pivot,   0..90 deg,  0.15 kg
    (joins /Cabinet/door_left <-> /Cabinet/handle_left, which carries hub + lever)
  latch_right  revolute Y at the pivot, -90..0 deg,  0.15 kg
```

Stage is metres, Z up, origin at the centre of the footprint on the floor.
The cabinet's authored front faces **-Y**. In the task scene (`openarm/env_cfg.py`)
the cabinet is yawed **-90 deg about Z** so that authored -Y maps to world
-X — i.e. the doors face the robot, which stands at negative x and faces +X.

Two things worth knowing about the collision setup:

* **Nothing is convex-hulled**, so the interior stays hollow and the wrist
  cameras see real shelves.
* **The escutcheon is visual only.** A collider there would sit permanently
  inside the rotating hub and make the latch joint fight itself.

The latch pair mirrors, so each lever sweeps away from the centre seam
instead of crossing over the other door.

## Importing & regenerating

The authored USD ships with the standard `UsdPhysics` schema only
(`usd-core` does not ship `PhysxSchema`), which is enough for Isaac Lab to
simulate. PhysX-specific tuning — solver iteration counts, contact/rest
offsets, per-material friction — has to be added on the Isaac side. Expect
to raise friction on the lever: a chrome handle is a genuinely hard grasp
case and the default material will let the gripper slide.

```bash
pip install usd-core
python3 ../../scripts/build_cabinet.py -o rpl_cabinet.usda       # from this dir
python3 ../../scripts/validate_cabinet.py rpl_cabinet.usda       # must print 0 failures
python3 ../../scripts/preview_cabinet.py rpl_cabinet.usda
```

The task env cfg (`openarm/env_cfg.py`) loads this USD as an `ArticulationCfg`
and reads its DOFs by exact name (`hinge_left`, `hinge_right`,
`latch_left`, `latch_right`).

## What the real episodes actually show

Analysed from the full 100 episodes at `openarm_open_lab_door`
(111,624 frames, 62 min, LeRobot v3.0):

* **It is a one-armed task.** Left arm 77,549 deg of total joint travel
  against the right arm's 2,960; left gripper spans 72.1 deg, right 3.4 deg
  (noise).
* **There is no distinct lever turn.** The operator pre-closes the gripper
  at home, then reaches and pulls. The roughly 38 deg of wrist roll happens
  during the approach, never as an in-place 90 deg throw.

So the task cfg here **treats the handle as rigid geometry and lets the door
open on pull**. `scripts/isaac_latch.py` is kept only in case a future task
needs a real lever turn — it is **not wired** into the env cfg. (A kinematic
reading of joint trajectories; not frame-by-frame-checked against the wrist
video.)

Episode structure, from episode 6:

| t (s) | |
|---|---|
| 0.0 | home, gripper open |
| 1.5 | **gripper closes at home, before reaching** |
| 3.0 | arm extends to the handle, j1 37 deg, j4 63 deg |
| 4.5–6.0 | pull, wrist roll settles near -40 deg |
| 7.5–9.0 | gripper opens, releases |
| 16.5+ | home and idle for the last ~12 s |

### Matching the real dataset

From `openarm_open_lab_door/meta/`:

* task string `Open lab door handle`
* `bi_openarm_follower`, 25 episodes, 22,443 frames, 30 fps (seed set)
* 48-dim state and action
* cameras `left_chest` 480×848, `left_wrist_left` 480×640, `right_wrist_right` 480×640
* LeRobot v3.0

The `openarm/visuomotor_env_cfg.py` camera resolutions match this feature spec
(848×480 chest colour, 640×480 wrist UVC) so renders are directly comparable
to the real observations. Synthetic data has to match those camera
intrinsics, placements and the 48-dim convention, or real and synthetic
cannot be co-trained.

## Known gaps (carried from `cabinet_sim/HANDOFF.md`)

1. **Camera mount poses.** `scripts/openarm_cameras.py` has
   `CHEST_POS_LOCALISH`, `CHEST_PITCH_DOWN_DEG` and `WRIST_OFFSET` fitted by
   eye against real footage, not measured off the hardware. Compare
   `docs/compare_real_vs_sim.jpg`: the real chest frame puts both handles
   upper-left with floor lower-right, and the sim does not match that yet.
   The optics are solid; the mount geometry is the gap. Anyone with the
   physical robot can close it in minutes. This is **the biggest open item**
   before publishing synthetic training data.
2. **Robot base-to-cabinet geometry.** A rigid base shift computed from the
   recorded pose drives the torso ~230 mm into the cabinet, because the
   assumed arm yaw and lift don't match the real base-to-cabinet geometry.
   Needs IK, or the real base offset measured in the lab. `env_cfg.py` leans
   on a hand-tuned base pose that looks right in sim but has not been
   validated against the real rig.
3. **The OpenArm asset has no stand.** `env_cfg.py` fakes one with the
   nucleus `Stand` prop under the base, z-scaled to the ground. The real rig
   is on a tripod.
4. **Shelves are bare.** The wrist cameras will see inside once a door
   swings, so whatever is really on those shelves should be modelled before
   generating training data.
5. **`isaac_latch.py` has never been run.** It is written against Isaac 5.1
   APIs but untested. Not wired into the task — kept only in case a future
   task needs a real lever turn.
6. **Latch release direction.** Mirrored is physically sensible, but which
   way each lever actually turns needs a look in person or in the video.
7. **Which door does the robot open,** and does it approach from the front or
   at an angle. The task cfg defaults to the `hinge_left` joint for success
   (via `mdp.door_opened`'s `door_joint_name` param); swap to `hinge_right`
   for a right-handed variant.

## Isaac Sim gotchas worth keeping

* **OpenArm faces +X.** `openarm_left_link1` sits at y=+0.145,
  `openarm_right_link1` at y=-0.145, so the robot's left is +Y and forward
  is +X. Facing the cabinet across -X therefore means the robot is at
  negative x looking the right way out of the box — no yaw.
* **DOF names are `openarm_<side>_joint<N>`** with no separator before the
  digit, alongside `openarm_<side>_finger_joint<N>`. Match exactly: a
  substring test on `joint1` also hits the finger joints, and `"left" in name`
  also matches `openarm_right_left_finger`.
* **Isaac's default horizontal aperture is 20.955 mm**, not a 36 mm
  full-frame sensor, so `focal_length=24` is roughly a 47 deg lens, not 74.
  The `visuomotor_env_cfg.py` focal lengths are derived from the real
  cameras' horizontal FOV against this aperture (~15 mm for the 69.4 deg
  D435 colour imager, ~6 mm for the ~119 deg Innomaker wrist cam).
* **A bbox on a prim with no geometry** returns the empty-range sentinel
  (±3.4e38); arithmetic on it silently teleports whatever you are placing.
  `scripts/build_scene.py` fails loudly on an empty bbox instead.
* **`SingleArticulation` in Isaac 5.1** has no `get_dof_limits` (use
  `art._articulation_view.get_dof_limits()`) and no
  `set_joint_position_targets` (use
  `apply_action(ArticulationAction(joint_positions=...))`).
* **`sim_app.close()` discards buffered stdout.** Headless scripts
  (`scripts/isaac_smoke_test.py`, etc.) write results to a file with an
  fsync per line and run under `python -u`.

## References in this folder

* `docs/cabinet_preview.png` — current render.
* `docs/scene_final_00.png`, `docs/scene_final_01.png` — composed cabinet +
  OpenArm scene stills.
* `docs/compare_real_vs_sim.jpg` — side-by-side of real vs. sim chest frames.

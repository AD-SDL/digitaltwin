#!/usr/bin/env python3
"""
Latch lock for the RPL cabinet.  Isaac Sim 5.1 (falls back to 4.x names).

No physics engine gives you a latch for free.  This holds each door shut
until its lever has been turned past a threshold, then lets it swing.  The
policy therefore has to rotate the handle before the door will move, which
is the behaviour the real task has.

Implementation note.  An earlier version mutated the USD joint-limit
attributes.  That is unreliable: once the articulation is initialised, PhysX
has its own copy and later USD attribute writes may never reach it.  This
version drives the DOF gains instead, which is the supported runtime path.

  locked    stiffness high, target 0      -> the door is held shut
  released  stiffness 0, light damping    -> the door swings freely

Drop the whole thing if the real demonstrations show no distinct lever
phase: then the handle is just rigid geometry and the door opens on pull.

Usage (standalone script or script editor):

    from isaac_latch import CabinetLatch
    latch = CabinetLatch(art)                 # art = your Articulation
    world.add_physics_callback("latch", lambda dt: latch.update())
"""

import numpy as np

RELEASE_DEG = 55.0    # lever angle past which the door unlocks
RELATCH_DEG = 15.0    # lever angle below which a shut door locks again
CLOSED_DEG = 3.0      # door angle under which it counts as shut

HOLD_STIFFNESS = 1.0e5
HOLD_DAMPING = 1.0e4
FREE_DAMPING = 3.0    # stands in for hinge friction once released


class CabinetLatch:
    """Couples each handle DOF to whether its door is held shut.

    `articulation` is an initialised Isaac Sim articulation wrapping the
    cabinet prim, i.e. something exposing dof_names, get_dof_index,
    get_joint_positions and set_gains.
    """

    SIDES = ("left", "right")

    def __init__(self, articulation,
                 release_deg=RELEASE_DEG,
                 relatch_deg=RELATCH_DEG,
                 closed_deg=CLOSED_DEG,
                 verbose=False):
        self.art = articulation
        self.release_deg = release_deg
        self.relatch_deg = relatch_deg
        self.closed_deg = closed_deg
        self.verbose = verbose

        names = list(self.art.dof_names or [])
        self.idx = {}
        for s in self.SIDES:
            for kind in ("hinge", "latch"):
                n = f"{kind}_{s}"
                if n not in names:
                    raise RuntimeError(
                        f"dof '{n}' not found; articulation has {names}")
                self.idx[n] = self.art.get_dof_index(n)

        self.n_dof = len(names)
        self._locked = {s: None for s in self.SIDES}
        for s in self.SIDES:
            self._set_locked(s, True)

    # -- helpers ---------------------------------------------------------
    def _set_targets(self, target):
        """Isaac 5.1's SingleArticulation has no set_joint_position_targets;
        it takes an ArticulationAction via apply_action."""
        if hasattr(self.art, "set_joint_position_targets"):
            self.art.set_joint_position_targets(target)
            return
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        self.art.apply_action(
            ArticulationAction(joint_positions=np.asarray(target)))

    def _angle_deg(self, dof_name):
        q = self.art.get_joint_positions()
        return float(np.degrees(np.asarray(q).reshape(-1)[self.idx[dof_name]]))

    def _set_locked(self, side, locked):
        if self._locked.get(side) == locked:
            return
        i = self.idx[f"hinge_{side}"]
        kp = np.full(self.n_dof, np.nan)
        kd = np.full(self.n_dof, np.nan)
        kp[i] = HOLD_STIFFNESS if locked else 0.0
        kd[i] = HOLD_DAMPING if locked else FREE_DAMPING
        # NaN entries leave the other DOFs untouched
        self.art.set_gains(kps=kp, kds=kd)
        if locked:
            full = np.asarray(self.art.get_joint_positions()).reshape(-1).copy()
            full[i] = 0.0
            self._set_targets(full)
        self._locked[side] = locked
        if self.verbose:
            print(f"[latch] {side} -> {'locked' if locked else 'released'}")

    # -- per-step --------------------------------------------------------
    def update(self):
        for s in self.SIDES:
            lever = abs(self._angle_deg(f"latch_{s}"))
            door = abs(self._angle_deg(f"hinge_{s}"))
            if self._locked[s]:
                if lever >= self.release_deg:
                    self._set_locked(s, False)
            elif door <= self.closed_deg and lever <= self.relatch_deg:
                self._set_locked(s, True)

    def force_release(self, side=None):
        for s in ([side] if side else self.SIDES):
            self._set_locked(s, False)

    def state(self):
        return {s: ("locked" if self._locked[s] else "released")
                for s in self.SIDES}

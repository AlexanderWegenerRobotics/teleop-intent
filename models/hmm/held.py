"""Detects whether an arm is currently holding something, and which
candidates should be excluded from the target model's pool as a result.

Primary signal: SensorFrame.grasp_confirmed (ArmControl::updateGraspConfirmation,
src/arm_control.cpp) -- a real, deployment-real gripper-width-and-timing
signal, not privileged sim state, and not our own guess.

Fallback: episodes without grasp_confirmed (older logs, not yet backfilled --
see scripts/backfill_grasp_confirmed.py) use a proximity + gripper-width
heuristic. Distance alone can't tell "about to grasp" apart from "already
grasped" (an arm closing in on an object is also very close to it), so this
combines proximity with gripper state: held = nearest object-type candidate
is within HELD_DIST_THRESHOLD *and* the gripper is closed enough that
something could plausibly be between the fingers. Recomputed fresh every
frame from current state, not a latched/remembered event -- a missed grasp
never reads as holding, so a regrasp attempt isn't blocked by a false memory
of the first attempt succeeding.

Thresholds below were read off real labeled data (episode 000), used only by
the fallback path: held phases (grasp/transport/place) cluster at
EE-to-object distance ~0.09-0.12m, while even the closest 10% of approach
frames stay >=0.17m. Gripper width narrows from ~0.086m (open) to ~0.056m
(closed) over this rig's stroke; transport (definitely holding) sits at
~0.062m mean. First-pass defaults for this rig/object set, not universal
constants.

target_exclusion_mask is deliberately broader than "just the held object":
a single-gripper arm holding one object cannot simultaneously be reaching
for any OTHER object either, so every CAND_OBJECT slot is excluded while
holding, not merely the specific one in hand -- only bins (or null) remain
valid targets. Masking only the held object left a real gap: the model
would happily latch onto some other nearby object as the "target" during
transport/place, which makes no semantic sense once the gripper is full.
"""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import (SensorFrame, GLOBAL_FEATURE_NAMES,
                                            CANDIDATE_FEATURE_NAMES, Phase)
from teleop_orchestrator.contracts.features import CAND_OBJECT, CAND_GOAL

_DIST_COL = {"left": CANDIDATE_FEATURE_NAMES.index("dist_left"),
             "right": CANDIDATE_FEATURE_NAMES.index("dist_right")}

HELD_DIST_THRESHOLD = 0.15    # fallback only: 3D EE<->object distance (m) below which an object counts as held
GRIPPER_CLOSED_WIDTH = 0.075  # fallback only: gripper_width (m) below which fingers are closed enough to be holding


def is_holding(frame: SensorFrame, side: str) -> bool:
    """Whether this arm is currently holding some object-type candidate
    (real grasp_confirmed signal when available, else the proximity +
    gripper-width fallback). Doesn't say *which* one -- see
    held_object_mask/target_exclusion_mask for that.
    """
    is_object = (frame.candidate_types == CAND_OBJECT) & frame.candidate_mask
    if not is_object.any():
        return False

    if side in frame.grasp_confirmed:
        return bool(frame.grasp_confirmed[side])

    gw = float(frame.global_features[GLOBAL_FEATURE_NAMES.index(f"gripper_width_{side}")])
    if gw > GRIPPER_CLOSED_WIDTH:
        return False
    dist_col = _DIST_COL[side]
    dists = np.where(is_object, frame.candidate_features[:, dist_col], np.inf)
    return bool(dists.min() < HELD_DIST_THRESHOLD)


def held_object_mask(frame: SensorFrame, side: str) -> np.ndarray:
    """Returns a boolean mask [n_candidates], True at the index of whichever
    specific object-type candidate this arm is holding, if any (at most one
    True entry) -- the nearest object-type candidate, when is_holding is
    true. Useful for display/debugging ("what is it holding"); target
    masking itself should use target_exclusion_mask, not this.
    """
    n = len(frame.candidate_mask)
    held = np.zeros(n, dtype=bool)
    if not is_holding(frame, side):
        return held
    is_object = (frame.candidate_types == CAND_OBJECT) & frame.candidate_mask
    dists = np.where(is_object, frame.candidate_features[:, _DIST_COL[side]], np.inf)
    # World EE and object positions when both exist: the intent-log EE behind
    # the dist column is 7-14 cm off, which picked a neighbouring peg in about
    # a quarter of held GRASP frames on FMB (010); world positions got all of them.
    world, ee = frame.candidate_world_pos, frame.ee_world_pos.get(side)
    if world is not None and ee is not None:
        wd = np.linalg.norm(world - np.asarray(ee, dtype=np.float64)[None, :], axis=1)
        if np.isfinite(wd[is_object]).any():
            dists = np.where(is_object & np.isfinite(wd), wd, np.inf)
    held[int(np.argmin(dists))] = True
    return held


# Phases in which a held object is genuinely no longer the operator's target.
#
# GRASP is deliberately NOT here, and that omission is the fix for a measured
# bug. grasp_confirmed goes true the moment the gripper reports a successful
# close, which happens while the fingers are still settling and well before
# the object is being carried anywhere. Excluding objects from that instant
# meant that during GRASP -- the phase whose entire purpose is acting on that
# object -- the model was forbidden from naming it.
#
# The damage was measured, not guessed: 9952 of 92389 committed training
# frames (10.8%) had a labelled target the pool excluded, and 6434 of those
# were GRASP frames. Every model in this repo scored a guaranteed error on
# them, and the HMM had been doing so silently since v1 because _score clamped
# the impossible probability at 1e-12 instead of complaining.
#
# APPROACH is excluded for the same reason (831 frames): if the gripper
# confirms early on a mis-grasp, the operator is still reaching for that
# object and it is still the answer.
EXCLUSION_PHASES = frozenset({Phase.TRANSPORT, Phase.PLACE})


def target_exclusion_mask(frame: SensorFrame, side: str,
                          phase: int | None = None, phases=EXCLUSION_PHASES) -> np.ndarray:
    """Candidates to exclude from the target pool while this arm is carrying
    something: every CAND_OBJECT-type candidate, not just the specific one
    held -- a full gripper can't be reaching for a different object either.
    Bins are never excluded here (they're never "held").

    `phase` gates the rule to EXCLUSION_PHASES. Pass the current phase (the
    model's own estimate at inference, the label while fitting) and objects
    stay selectable during APPROACH and GRASP, where the labels say they are
    the target. Pass None for the pre-fix behaviour -- kept only so an old
    checkpoint can be re-scored under the rule it was fitted with, never as a
    default for new work.

    Note the remaining disagreement this does NOT resolve: 2433 TRANSPORT and
    254 PLACE frames still carry an object-valued target label. Those are a
    real question about what "target" means mid-carry -- the object you are
    moving, or the bin you are moving it to -- and the answer belongs in the
    labelling guide, not in this function.
    """
    n = len(frame.candidate_mask)
    if phase is not None and int(phase) not in phases:
        return np.zeros(n, dtype=bool)
    if not is_holding(frame, side):
        return np.zeros(n, dtype=bool)
    return (frame.candidate_types == CAND_OBJECT) & frame.candidate_mask


def part_type(name: str) -> str:
    """Part type an object name maps to, the key goals are typed by (spare_peg_1_b -> peg_1), as in the auto-labeler."""
    parts = name.replace("spare_", "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def held_part_index(frame: SensorFrame, side: str) -> int | None:
    """Candidate index of the object this arm is holding, or None."""
    held = held_object_mask(frame, side)
    return int(np.argmax(held)) if held.any() else None


def occupied_goal_mask(frame: SensorFrame, own_part: str | None, goal_types: dict, occupied_m: float) -> np.ndarray:
    """Goal candidates already filled by a type-matching part other than own_part (the arm's current or last held part).

    own_part keeps the slot the arm just seated a part in selectable through release and retreat;
    it drops out once the arm grasps the next part. Needs candidate_world_pos and candidate_names.
    """
    n = len(frame.candidate_mask)
    out = np.zeros(n, dtype=bool)
    world, names = frame.candidate_world_pos, frame.candidate_names
    if world is None or len(names) < n:
        return out
    finite = np.all(np.isfinite(world), axis=1)
    objs = np.flatnonzero((frame.candidate_types == CAND_OBJECT) & frame.candidate_mask & finite)
    objs = [o for o in objs if names[o] and names[o] != own_part]
    for g in np.flatnonzero((frame.candidate_types == CAND_GOAL) & frame.candidate_mask & finite):
        gt = goal_types.get(names[g])
        out[g] = any(part_type(names[o]) == gt and np.linalg.norm(world[o] - world[g]) < occupied_m for o in objs)
    return out


def pinned_exclusion_mask(frame: SensorFrame, side: str, held: int | None = None) -> np.ndarray:
    """While holding, every object except the held one, in any phase: a full gripper pins the target to its part.

    Goals are untouched, so the target can still move from the part to its slot.
    held overrides the nearest-object identification (the feature builder latches it).
    """
    n = len(frame.candidate_mask)
    if held is None:
        mask = held_object_mask(frame, side)
    else:
        mask = np.zeros(n, dtype=bool)
        mask[held] = True
    if not mask.any():
        return np.zeros(n, dtype=bool)
    return (frame.candidate_types == CAND_OBJECT) & frame.candidate_mask & ~mask

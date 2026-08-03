"""Per-arm scalar feature vector for PhaseHMM's emission model.

Derived directly from a SensorFrame via the named contract constants
(CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES) -- never a magic index -- so
this stays correct if the frame's field order ever changes. Kept local to
models/hmm/: nothing outside this package needs to know phase uses exactly
these nine signals.

EE velocity replaces an earlier joint-velocity (dq_norm) feature: joint
velocity is a poor proxy for task-relevant motion because of kinematic
redundancy (the arm can move in its nullspace -- e.g. repositioning the
elbow -- with large joint velocities but near-zero end-effector motion).
Velocity is computed as a strictly causal finite difference using each
frame's real timestamp (never assumed-uniform spacing, even though this rig
happens to sample at a fixed 33ms in practice) -- deliberately unsmoothed for
now: per-frame speed jumps measured on real episodes are small relative to
the signal, and the emission model already has a free per-class std to
absorb whatever noise magnitude a phase actually exhibits. Any future
smoothing here would have to be strictly causal (unlike common.py's
compute_ee_speed, which is a centered/acausal filter valid only for offline
plotting) to avoid a train/serve mismatch.

gaze_to_candidate_dist is gaze's counterpart to min_candidate_dist: a
*relative* geometric quantity (nearest valid candidate to the gaze point, in
the same reconciled UV space contracts.features now guarantees), not raw
absolute gaze/candidate pixel coordinates -- absolute image position has no
consistent meaning across episodes with different scene layouts, so it
wouldn't generalize the way a relative distance does. It's scene-level (not
per-arm), since gaze reflects operator attention regardless of which arm's
phase is being inferred, so both arms' phase features see the same value
each frame -- exactly like gaze_valid already is shared.

MISSING VALUES: min_candidate_dist and gaze_to_candidate_dist are NaN, not a
sentinel, on frames where they are undefined (no visible candidate / invalid
gaze), and each carries an explicit companion indicator (cand_dist_valid,
gaze_dist_valid). Earlier revisions substituted a 5.0 sentinel, which was an
order of magnitude above the real range (per-phase means 0.21-0.51 m) and so
dominated the per-class variance the emission model fits: measured on the v7
checkpoint, both features ended up with per-class std 0.65-0.78 and a
between-phase separability of at most 0.33 pooled std for ANY pair of phases
-- i.e. the two most geometrically meaningful signals in the vector had been
flattened into noise by their own missing-data encoding. phase.py's emission
model skips NaN dimensions outright (both the Mahalanobis term AND the
matching -log(sigma) normalizer, so skipping stays unbiased), which is the
correct missing-at-random treatment; whatever information "there is nothing
visible right now" carries is then carried explicitly by the indicator,
modelled as a Bernoulli rather than smuggled in as a fake distance.

Deliberately NOT fed from the deployed C++ filter's own belief
(tgt_belief_/ee_belief_): contracts.features.LEAKAGE_PREFIXES already
excludes tgt_belief_ from any model input so a learned model can't just
parrot the baseline's answer instead of learning from raw signal; using it
here would also make this model depend on that filter continuing to run at
deployment (backwards, if this model is meant to eventually stand in for
it) and isn't guaranteed present on every episode (only ones that have been
"recomputed").

is_holding (held.is_holding -- ArmControl::updateGraspConfirmation when
available, else the proximity+gripper-width fallback) targets a specific
confusion: grasp is brief and looks almost identical to idle in the other
seven features (near-zero velocity, at most a small gripper-delta blip), so
the two overlap heavily in feature space and the forward filter's emission
term can win out over a correctly-low idle<->grasp transition prior several
frames running. is_holding is a near-binary signal built specifically to
catch this, orthogonal to the velocity/force features that can't cleanly
separate it. Reused directly from held.py rather than reimplemented here, so
target-pool masking and this phase feature can never silently disagree
about whether an arm is holding something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleop_orchestrator.contracts import SensorFrame, CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES

from .held import is_holding

FEATURE_NAMES = [
    "ee_vel_x", "ee_vel_y", "ee_vel_z",
    "gripper_width", "gripper_width_delta", "contact_force_norm",
    "min_candidate_dist", "gaze_to_candidate_dist", "is_holding",
    "cand_dist_valid", "gaze_dist_valid",
]

# Features that are {0, 1} valued and must NOT be fitted as Gaussians.
# A binary dimension driven through a Gaussian emission collapses its own
# std as the class becomes pure (is_holding had std 0.095 under IDLE on v7),
# and the resulting -log(sigma) term is an unbounded, observation-independent
# bonus for whichever class happens to be purest -- the single largest
# contributor to the measured 5.4-nat unconditional preference for IDLE over
# GRASP. phase.py fits these as Bernoulli instead, where the analogous term
# is bounded by the probability floor.
BINARY_FEATURE_NAMES = frozenset({"is_holding", "cand_dist_valid", "gaze_dist_valid"})

# Boolean mask over FEATURE_NAMES, in the same order -- what PhaseHMM.fit
# wants, so the Gaussian/Bernoulli split is declared once, here, next to the
# features themselves rather than restated as indices at the fit site.
IS_BINARY_FEATURE = np.array([n in BINARY_FEATURE_NAMES for n in FEATURE_NAMES], dtype=bool)

_DIST_COL = {"left": CANDIDATE_FEATURE_NAMES.index("dist_left"),
             "right": CANDIDATE_FEATURE_NAMES.index("dist_right")}
_GAZE_IDX = [GLOBAL_FEATURE_NAMES.index("gaze_x"), GLOBAL_FEATURE_NAMES.index("gaze_y")]
_PX_IDX = [CANDIDATE_FEATURE_NAMES.index("px_u"), CANDIDATE_FEATURE_NAMES.index("px_v")]

# Undefined distances are NaN (skipped by the emission model) rather than an
# out-of-range sentinel; see the MISSING VALUES note in the module docstring.
_NO_CANDIDATE_DIST = np.nan  # 3D EE<->candidate distance is undefined when none visible this frame
_NO_GAZE_DIST = np.nan       # gaze<->candidate UV distance is undefined when gaze invalid or no candidates

# EE_POS_IDX indexes global_features' ee_{side}_x/y/z (contracts.features'
# ee_pos, sourced from the intent log's ee_left/right_x/y/z) -- the
# world-equivalent frame scene.csv/slot_dist/candidate_world_pos use (see
# world_ee_velocity below). NOT used for phase's own velocity feature -- see
# _EE_POS_SLICE just below for why.
EE_POS_IDX = {
    "left": [GLOBAL_FEATURE_NAMES.index(f"ee_left_{a}") for a in ("x", "y", "z")],
    "right": [GLOBAL_FEATURE_NAMES.index(f"ee_right_{a}") for a in ("x", "y", "z")],
}

# proprio = concat(q[7], dq[7], O_T_EE[16]); O_T_EE translation is elements
# [12:15] of its own 16-vector, offset 7+7+12 = 26 in the concatenated array.
# Phase's velocity feature uses THIS (not EE_POS_IDX/the intent log's
# ee_{side}_x/y/z) even though it's in the arm's base frame rather than
# world frame: O_T_EE is written every control-loop frame, while
# ee_{side}_x/y/z is only updated when the C++ attention filter recomputes
# its belief -- empirically, ~1 update per 10 frames on average and observed
# stretches of 1000+ consecutive stale frames (~33s) on real data. Velocity
# from a rotation-only frame difference is still directionally self-consistent
# for phase's purposes (a fixed rotation doesn't change relative motion
# magnitude or timing), but computing it from a signal that's frozen most of
# the time made phase mostly read near-zero velocity and detect phases very
# late -- a real, confirmed regression from briefly switching this to
# ee_{side}_x/y/z. Reverted. The target alignment feature (target.py) still
# needs true world-frame EE position/velocity to compare against
# candidate_world_pos -- see world_ee_velocity, which is staleness-aware
# instead of assuming a per-frame update.
_EE_POS_SLICE = slice(26, 29)


@dataclass
class PhaseFeatureState:
    """Per-arm memory phase_features needs across calls: previous gripper
    width (for gripper_width_delta) and previous EE position/timestamp (for
    the causal EE-velocity finite difference). Owned by the caller (the
    filter instance), not this stateless function -- same pattern the
    gripper-delta memory already used, just widened to cover velocity too.
    """

    gripper_width: float
    ee_pos: np.ndarray
    timestamp_ns: int


def phase_features(frame: SensorFrame, side: str,
                    prev: PhaseFeatureState | None) -> tuple[np.ndarray, PhaseFeatureState]:
    """Returns (feature_vector[9], new_state) for one arm at this frame.

    `prev` is None on the first call of an episode (reset); the caller is
    responsible for threading the returned state back in on the next call
    and resetting it to None at episode boundaries.
    """
    ee_pos = frame.proprio[f"arm_{side}"][_EE_POS_SLICE].astype(np.float64)

    if prev is None:
        ee_vel = np.zeros(3)
    else:
        dt = (frame.timestamp_ns - prev.timestamp_ns) / 1e9
        ee_vel = (ee_pos - prev.ee_pos) / dt if dt > 1e-6 else np.zeros(3)

    gw = float(frame.global_features[GLOBAL_FEATURE_NAMES.index(f"gripper_width_{side}")])
    fext_idx = [GLOBAL_FEATURE_NAMES.index(f"fext_{side}_{axis}") for axis in ("fx", "fy", "fz")]
    contact_force_norm = float(np.linalg.norm(frame.global_features[fext_idx]))

    dist_col = _DIST_COL[side]
    valid_dist = frame.candidate_features[frame.candidate_mask, dist_col]
    cand_dist_valid = bool(valid_dist.size)
    min_dist = float(valid_dist.min()) if cand_dist_valid else _NO_CANDIDATE_DIST

    gaze_dist_valid = bool(frame.gaze_valid and frame.candidate_mask.any())
    if gaze_dist_valid:
        gaze_xy = frame.global_features[_GAZE_IDX]
        cand_uv = frame.candidate_features[frame.candidate_mask][:, _PX_IDX]
        gaze_dist = float(np.linalg.norm(cand_uv - gaze_xy[None, :], axis=1).min())
    else:
        gaze_dist = _NO_GAZE_DIST

    delta = 0.0 if prev is None else gw - prev.gripper_width
    holding_feat = 1.0 if is_holding(frame, side) else 0.0

    features = np.array([
        ee_vel[0], ee_vel[1], ee_vel[2],
        gw, delta, contact_force_norm,
        min_dist, gaze_dist, holding_feat,
        float(cand_dist_valid), float(gaze_dist_valid),
    ], dtype=np.float64)
    new_state = PhaseFeatureState(gripper_width=gw, ee_pos=ee_pos, timestamp_ns=frame.timestamp_ns)
    return features, new_state


# If the intent log's ee_{side}_x/y/z hasn't changed in longer than this,
# treat velocity as unknown (zero) rather than holding a possibly ancient
# direction estimate -- observed stale runs up to ~33s on real data, and a
# 33-second-old direction is worse than no evidence at all.
_WORLD_EE_STALE_TIMEOUT_S = 0.5


@dataclass
class WorldEEState:
    """Per-arm memory world_ee_velocity needs: the last DISTINCT
    (non-stale-duplicate) world-frame EE position/timestamp seen, plus the
    velocity estimate computed at that update -- held (not zeroed) on frames
    where the underlying signal simply repeats its last value, since it
    updates far less often than the frame grid (see world_ee_velocity)."""

    pos: np.ndarray
    timestamp_ns: int
    vel: np.ndarray


def world_ee_velocity(frame: SensorFrame, side: str,
                       prev: "WorldEEState | None") -> tuple[np.ndarray, np.ndarray, "WorldEEState"]:
    """Returns (velocity[3], position[3], new_state) in world frame, for the
    target alignment feature only (target.py) -- NOT used by phase_features,
    which needs a signal that updates every frame (see _EE_POS_SLICE's
    docstring for why the two features intentionally use different sources).

    ee_{side}_x/y/z only updates when the C++ attention filter recomputes
    (empirically ~1 update per 10 frames, sometimes 1000+ frames apart), so a
    naive per-frame finite difference would read as zero almost everywhere
    with sporadic huge spikes. Instead: recompute velocity only when the
    position actually changes (using the true elapsed time since the last
    change, not one grid step), and hold that estimate on frames where it
    hasn't -- except once too much real time has passed
    (_WORLD_EE_STALE_TIMEOUT_S), at which point velocity reports zero
    (target.py's speed gate then naturally skips the alignment evidence that
    frame, same as if no data were available).
    """
    pos = frame.global_features[EE_POS_IDX[side]].astype(np.float64)

    if prev is None:
        return np.zeros(3), pos, WorldEEState(pos=pos, timestamp_ns=frame.timestamp_ns, vel=np.zeros(3))

    if np.array_equal(pos, prev.pos):
        age_s = (frame.timestamp_ns - prev.timestamp_ns) / 1e9
        vel = prev.vel if age_s <= _WORLD_EE_STALE_TIMEOUT_S else np.zeros(3)
        return vel, pos, prev

    dt = (frame.timestamp_ns - prev.timestamp_ns) / 1e9
    vel = (pos - prev.pos) / dt if dt > 1e-6 else np.zeros(3)
    return vel, pos, WorldEEState(pos=pos, timestamp_ns=frame.timestamp_ns, vel=vel)

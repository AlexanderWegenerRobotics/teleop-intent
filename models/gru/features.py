"""Feature assembly for the GRU: identical arm features to the HMM, plus the
per-candidate tensor the HMM's target head got through a different route.

WHY THE ARM FEATURES ARE IMPORTED, NOT REIMPLEMENTED
-----------------------------------------------------
phase_features and target_exclusion_mask are imported from models.hmm rather
than copied, so the two models provably consume the same numbers. A copy would
drift the first time either side was touched, and every "the GRU beats the
HMM" claim afterwards would be confounded by an input difference nobody
remembered making. The dependency runs one way only -- gru imports hmm, never
the reverse -- so the HMM package still stands alone and can be run, trained
and scored with this package absent. (Those two functions live under hmm/ for
historical reasons rather than principled ones; if that ever grates, promoting
them to models/features.py is a pure move.)

NaN handling: the arm vector carries NaN for undefined distances, which the
HMM's emission skips per-dimension. A GRU cannot take NaN, so they are
replaced with 0 here -- which is information-preserving ONLY because
features.py already emits cand_dist_valid and gaze_dist_valid alongside them.
The network can learn "distance 0 with validity 0 means absent", which is
exactly the contract those two flags were added for.

WHY THERE IS A PER-CANDIDATE TENSOR AT ALL
-------------------------------------------
The HMM answered "which phase" and "which target" with two different
mechanisms: a scalar emission over 11 arm features for phase, and a separate
Bayesian filter over candidates driven by gaze and EE-alignment for target.
Only the first of those is a feature vector. If the GRU were handed the arm
features alone it would be structurally unable to say which candidate is the
target -- the arm vector contains only aggregates (distance to the NEAREST
candidate), never anything candidate-specific.

So the target head needs per-candidate inputs, and the honest choice is to
give it exactly the quantities the sticky filter already used: gaze-to-
candidate distance, EE-to-candidate distance, the EE-velocity alignment
cosine, and candidate type. Matched information, different model class --
which is the only way the comparison measures what it claims to.

`rich` adds signals the HMM never had (gaze offset as a vector rather than a
distance, candidate pixel position, held-object flag). It is OFF by default
and exists to answer a SEPARATE question -- "does the GRU benefit from
information the HMM could not use?" -- which must not be silently mixed into
the first one.
"""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import SensorFrame, CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from teleop_orchestrator.contracts.features import CAND_OBJECT

from ..hmm.features import phase_features, world_ee_velocity, FEATURE_NAMES as ARM_FEATURE_NAMES
from ..hmm.held import target_exclusion_mask, held_object_mask

# Per-candidate features, matched to what TargetStickyFilter consumed.
CAND_FEATURE_NAMES = [
    "gaze_dist",      # gaze -> candidate distance in the shared UV space
    "gaze_valid",     # 0 when gaze is invalid; gaze_dist is then 0 and meaningless
    "ee_dist",        # 3D end-effector -> candidate distance (m), this arm
    "align_cos",      # cos angle between EE velocity and the EE -> candidate direction
    "align_valid",    # 0 when the arm is too slow or the candidate has no world position
    "is_object",      # candidate type: a pickable object
    "is_bin",         # candidate type: a destination
]
RICH_CAND_FEATURE_NAMES = ["gaze_dx", "gaze_dy", "px_u", "px_v", "is_held"]

# Matches target._MIN_EE_SPEED_FOR_ALIGNMENT: below this the direction of
# motion is noise, so the alignment cosine is reported as unavailable rather
# than as a jittery value the network would have to learn to distrust.
_MIN_EE_SPEED_FOR_ALIGNMENT = 0.01

_GAZE_IDX = [GLOBAL_FEATURE_NAMES.index("gaze_x"), GLOBAL_FEATURE_NAMES.index("gaze_y")]
_PX_IDX = [CANDIDATE_FEATURE_NAMES.index("px_u"), CANDIDATE_FEATURE_NAMES.index("px_v")]
_DIST_COL = {"left": CANDIDATE_FEATURE_NAMES.index("dist_left"),
             "right": CANDIDATE_FEATURE_NAMES.index("dist_right")}


def candidate_feature_names(rich: bool = False) -> list[str]:
    return list(CAND_FEATURE_NAMES) + (list(RICH_CAND_FEATURE_NAMES) if rich else [])


def candidate_features(frame: SensorFrame, side: str, ee_vel: np.ndarray, ee_pos: np.ndarray,
                       rich: bool = False) -> np.ndarray:
    """[n_candidates, D] per-candidate features for one arm at one frame.

    Rows for invalid candidates are still produced (so the tensor keeps a
    fixed candidate axis) but are meaningless -- the caller masks them, and
    the target head's softmax never sees them. Filling them with zeros rather
    than leaving stale values in place keeps a masking bug from silently
    training on garbage.
    """
    n = frame.candidate_mask.shape[0]
    gaze_xy = frame.global_features[_GAZE_IDX].astype(np.float64)
    cand_uv = frame.candidate_features[:, _PX_IDX].astype(np.float64)
    ee_dist = frame.candidate_features[:, _DIST_COL[side]].astype(np.float64)

    gaze_valid = bool(frame.gaze_valid)
    delta = cand_uv - gaze_xy[None, :]
    gaze_dist = np.linalg.norm(delta, axis=1) if gaze_valid else np.zeros(n)

    speed = float(np.linalg.norm(ee_vel))
    world = frame.candidate_world_pos
    align_cos = np.zeros(n)
    align_ok = np.zeros(n, dtype=bool)
    if speed > _MIN_EE_SPEED_FOR_ALIGNMENT and world is not None:
        direction = world - ee_pos[None, :]
        dnorm = np.linalg.norm(direction, axis=1)
        align_ok = (dnorm > 1e-6) & np.all(np.isfinite(world), axis=1)
        if align_ok.any():
            align_cos[align_ok] = (direction[align_ok] @ (ee_vel / speed)) / dnorm[align_ok]

    is_object = (frame.candidate_types == CAND_OBJECT)
    cols = [gaze_dist,
            np.full(n, float(gaze_valid)),
            np.nan_to_num(ee_dist, nan=0.0, posinf=0.0, neginf=0.0),
            align_cos,
            align_ok.astype(float),
            is_object.astype(float),
            (~is_object).astype(float)]

    if rich:
        cols += [delta[:, 0] if gaze_valid else np.zeros(n),
                 delta[:, 1] if gaze_valid else np.zeros(n),
                 cand_uv[:, 0], cand_uv[:, 1],
                 held_object_mask(frame, side).astype(float)]

    out = np.stack(cols, axis=1)
    out[~frame.candidate_mask] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class EpisodeFeatureBuilder:
    """Threads the per-arm recurrent feature state (gripper delta, EE velocity)
    across a single episode.

    The HMM's phase_features and world_ee_velocity are both stateful -- they
    hold the previous frame's gripper width, position and timestamp -- and the
    state must be reset at every episode boundary and never carried across
    episodes. Bundling that here means the training builder and the online
    step() path cannot disagree about when it is reset, which is the classic
    way a causal model quietly acquires a train/serve mismatch.
    """

    def __init__(self, side: str, rich: bool = False):
        self.side = side
        self.rich = rich
        self._phase_state = None
        self._world_ee_state = None

    def reset(self) -> None:
        self._phase_state = None
        self._world_ee_state = None

    def step(self, frame: SensorFrame,
             phase: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (arm_features [D_arm], candidate_features [n, D_cand], mask [n]).

        `phase` gates held.target_exclusion_mask. Pass the PREVIOUS frame's
        phase -- the label while building training tensors, the model's own
        previous prediction at inference. Using the previous frame keeps the
        two paths structurally identical: the alternative (current-frame label
        at training, previous-frame estimate at serving) would differ in two
        ways at once and make any resulting mismatch impossible to attribute.
        """
        arm, self._phase_state = phase_features(frame, self.side, self._phase_state)
        ee_vel, ee_pos, self._world_ee_state = world_ee_velocity(
            frame, self.side, self._world_ee_state)
        cand = candidate_features(frame, self.side, ee_vel, ee_pos, rich=self.rich)
        # Same target pool the HMM used: a full gripper cannot be reaching for
        # another object either, so object-type candidates drop out while
        # holding. Applied here rather than in the network so both models are
        # choosing from an identical candidate set.
        mask = frame.candidate_mask & ~target_exclusion_mask(frame, self.side, phase)
        return np.nan_to_num(arm, nan=0.0, posinf=0.0, neginf=0.0), cand, mask


__all__ = ["ARM_FEATURE_NAMES", "CAND_FEATURE_NAMES", "RICH_CAND_FEATURE_NAMES",
           "candidate_feature_names", "candidate_features", "EpisodeFeatureBuilder"]

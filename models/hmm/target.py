"""A learned, causal sticky Bayesian filter for the target head.

Structurally the same filter already running in the live intention pipeline
(intention_buffer.cpp's computeBelief / recompute_intent_belief.py's
_sticky_bayes) -- a state stays put with probability rho, else the belief
relaxes toward uniform over currently-valid candidates -- but rho, the gaze
Gaussian's sigma, and the null-state prior are fit from labeled data via grid
search instead of hand-set, and rho is phase-conditioned (tighter once the
phase HMM believes we're mid-manipulation, looser during IDLE/APPROACH).

Not a general n-state learned transition matrix: candidate identity changes
per episode/task, so there isn't enough data to learn "P(switch from object_3
to bin_1)" specifically. The sticky/uniform structure is the generalizable
part; only its few scalar parameters are learned.

Requires candidate positions and gaze to be in the same coordinate space --
true for any episode converted after the intention_buffer.cpp ray-normalization
fix, NOT true for legacy (schema <3 / pre-fix) episodes, where gaze and slot
pixels were logged in two different, mismatched camera spaces. See the
handover notes on gaze_units for which episodes qualify.

INVALID CANDIDATES MUST BE ZEROED, NOT LEFT NEUTRAL
---------------------------------------------------
step() previously gave masked-out candidates a likelihood multiplier of 1.0
while _score_grid -- the code that actually fit rho/sigma/null_prior -- gave
them 0.0. That train/serve mismatch is what made "we are placing into a bin"
read as "we are still targeting the object in the gripper". Excluding the
carried object from the pool (held.target_exclusion_mask) did not remove it:
it handed it a NEUTRAL multiplier while every legitimate candidate was
attenuated by a Gaussian < 1 and null by null_prior < 1, so exclusion made
the excluded candidate STRONGER than everything it was competing with, and
the sticky prior then kept it there. Reproduced directly: with gaze resting
on the object being carried -- which is what operators actually do during
transport -- the old code held P(carried object) = 0.99 indefinitely after
exclusion kicked in, where the fitted semantics decay it to 0 within ~20
frames. It also explains the v3 -> v4 target-accuracy regression (0.678 ->
0.626) that landed with the broadened exclusion: excluding MORE objects gave
more candidates the neutral multiplier.

Zeroing the likelihood alone is not sufficient. On frames with no evidence at
all (gaze invalid, alignment gated off by low speed) the old code returned
`predicted` untouched, so stale belief on a now-invalid candidate survived
regardless. The predict step therefore zeroes invalid states explicitly and
renormalises, which is the only place guaranteed to run every frame.

null_prior is now applied once whenever ANY evidence channel fired, rather
than only inside the gaze branch. Previously an alignment-only frame left
null with an implicit multiplier of 1.0 while every candidate got
exp(-(1-cos)^2 / 2 sigma_align^2) <= 1, i.e. the alignment channel silently
pushed mass toward null on exactly the frames it was added to help with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import numpy as np

from teleop_orchestrator.contracts import NULL_TARGET, Phase

TIGHT_PHASES = frozenset({Phase.GRASP, Phase.TRANSPORT, Phase.PLACE})

# Below this EE speed (m/s), direction-of-motion is noise -- skip the
# alignment evidence entirely rather than let a near-stationary arm's
# jittery direction vote for or against a candidate.
_MIN_EE_SPEED_FOR_ALIGNMENT = 0.01

# Weight on the committed-frame group mean in the fitting objective; the
# remainder goes to the null-frame group mean. See _combine_scores for why
# this is a weighted average of two GROUP means rather than a plain average
# over frames.
DEFAULT_COMMITTED_WEIGHT = 0.75

# Search axes for fit(). Widened after the v8 fit returned rho_loose=0.7,
# rho_tight=0.9, sigma=0.1 and null_prior=0.2 -- ALL FOUR sitting on an edge
# of the previous grid ((0.7, 0.85, 0.9), (0.9, 0.95, 0.98, 0.995),
# (..., 0.02, 0.05, 0.1), (0.05, 0.1, 0.2)). Four boundary hits is not a
# coincidence; it means the optimum was outside the box and the reported
# values were clamps, not estimates. Two of them also moved for a
# understandable reason: once step() stopped letting stale mass survive on
# invalid candidates, the filter no longer needed high stickiness to look
# stable, so the search pushed rho down against the wall.
#
# sigma keeps both scales because the two would be indistinguishable from a
# fitted value alone: ~10-50 is right if gaze/candidate pixels are raw image
# coordinates, ~0.02-0.3 if they are normalised UV. v7 and v8 both chose 0.1,
# which says normalised -- worth asserting in the converter rather than
# rediscovering here every fit.
_GRID_AXES = {
    "rho_loose": (0.2, 0.35, 0.5, 0.6, 0.7, 0.85, 0.9),
    "rho_tight": (0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.995),
    "sigma": (10.0, 20.0, 30.0, 50.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4),
    "null_prior": (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0),
}


def warn_if_on_grid_boundary(params: "TargetFilterParams") -> list[str]:
    """Reports any fitted parameter sitting on the edge of its search axis.

    A boundary hit means the grid, not the data, chose the value. Silent
    clamping is how v7 and v8 both ended up reporting four 'fitted'
    parameters that were really just the extremes of the box they were
    searched in.
    """
    hits = []
    for name in _GRID_AXES:
        axis = sorted(_GRID_AXES[name])
        value = getattr(params, name)
        if value in (axis[0], axis[-1]):
            where = "minimum" if value == axis[0] else "maximum"
            hits.append(f"{name}={value} is the {where} of its search axis")
    if hits:
        print("  WARNING: fitted target parameters on a grid boundary -- the grid may be "
              "clamping the optimum:")
        for h in hits:
            print(f"    {h}")
    return hits


@dataclass
class TargetFilterParams:
    """Fitted parameters: everything TargetStickyFilter.step needs."""

    rho_loose: float   # P(stay) while phase is IDLE/APPROACH -- attention still settling
    rho_tight: float   # P(stay) once phase is GRASP/TRANSPORT/PLACE -- target is committed
    sigma: float       # gaze Gaussian sigma, in whatever unit gaze/candidate px share
    null_prior: float  # fixed relative likelihood of the null ("no target") state
    # Gaussian width (in units of 1 - cosine_similarity) for the EE-velocity/
    # EE-to-candidate-direction alignment evidence. np.inf (the default)
    # disables the term entirely -- exp(-(1-cos)^2 / (2*inf^2)) == 1 for any
    # cos_sim, i.e. a neutral multiplier -- so any checkpoint fit before this
    # feature existed (sigma_align absent -> dataclass default) behaves
    # identically to before, no branching or version-checking needed.
    sigma_align: float = np.inf


class TargetStickyFilter:
    """Causal sticky filter over [n_candidates + 1 (null)] states for one arm."""

    def __init__(self, params: TargetFilterParams):
        self.params = params
        self.belief: np.ndarray | None = None  # sized on first step (candidate count can vary)

    def reset(self) -> None:
        """Clears belief; the next step() re-initializes it to uniform-over-valid."""
        self.belief = None

    def step(self, gaze_xy: np.ndarray, candidate_px: np.ndarray, candidate_mask: np.ndarray,
              gaze_valid: bool, phase_estimate: int,
              ee_vel: Optional[np.ndarray] = None, ee_pos: Optional[np.ndarray] = None,
              candidate_world_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns posterior[n_candidates + 1] (last entry is the null state).

        candidate_px: [n_candidates, 2] (u, v) per candidate, same convention as gaze_xy.

        Candidates excluded by candidate_mask hold exactly zero probability on
        exit -- both because the predict step zeroes them and because their
        likelihood is zero, not one. See the module docstring: getting this
        wrong turned exclusion into a promotion.

        ee_vel/ee_pos/candidate_world_pos are all optional and independent of
        gaze: when given (and sigma_align is finite), they add a second,
        gaze-independent evidence channel -- cosine similarity between EE
        velocity and the direction from the EE to each candidate, peaked at
        1 (moving straight at it). This can fire on frames where gaze_valid
        is False, which gaze evidence never could. ee_pos/candidate_world_pos
        must be in the same (world) coordinate frame -- see features.py's
        _EE_POS_IDX docstring for why that's ee_{side}_x/y/z, not O_T_EE.
        candidate_world_pos rows may be individually NaN (candidate has no
        backfilled position); those candidates just don't get this evidence
        that frame, same as an invalid/masked candidate.
        """
        n = candidate_mask.shape[0]
        valid = np.append(candidate_mask, True)  # null is always a valid state
        n_valid = int(valid.sum())
        uniform = np.where(valid, 1.0 / n_valid, 0.0)

        if self.belief is None or self.belief.shape[0] != n + 1:
            self.belief = uniform.copy()

        rho = self.params.rho_tight if phase_estimate in TIGHT_PHASES else self.params.rho_loose
        predicted = rho * self.belief + (1.0 - rho) * uniform
        # Any state that has since become invalid keeps stale mass through the
        # rho term; drop it here rather than relying on the likelihood, which
        # is skipped entirely on no-evidence frames.
        predicted = np.where(valid, predicted, 0.0)
        pred_total = predicted.sum()
        predicted = predicted / pred_total if pred_total > 1e-12 else uniform

        likelihood = np.where(valid, 1.0, 0.0)
        have_evidence = False

        if gaze_valid:
            d2 = np.sum((candidate_px - gaze_xy[None, :]) ** 2, axis=1)
            likelihood[:n] *= np.where(candidate_mask, np.exp(-d2 / (2 * self.params.sigma ** 2)), 0.0)
            have_evidence = True

        if (np.isfinite(self.params.sigma_align) and ee_vel is not None
                and ee_pos is not None and candidate_world_pos is not None):
            speed = float(np.linalg.norm(ee_vel))
            if speed > _MIN_EE_SPEED_FOR_ALIGNMENT:
                direction = candidate_world_pos - ee_pos[None, :]  # [n, 3]
                dnorm = np.linalg.norm(direction, axis=1)
                has_dir = candidate_mask & (dnorm > 1e-6) & np.all(np.isfinite(candidate_world_pos), axis=1)
                cos_sim = np.zeros(n)
                if has_dir.any():
                    cos_sim[has_dir] = (direction[has_dir] @ (ee_vel / speed)) / dnorm[has_dir]
                # 1.0 for candidates with no world position: that is MISSING
                # evidence, not exclusion -- exclusion was already applied by
                # the `valid` initialisation of `likelihood` above.
                align_lik = np.where(
                    has_dir, np.exp(-(1.0 - cos_sim) ** 2 / (2 * self.params.sigma_align ** 2)), 1.0)
                likelihood[:n] *= align_lik
                have_evidence = True

        if have_evidence:
            # Applied once per frame, for any evidence channel -- not once per
            # channel, and not only for gaze (see module docstring).
            likelihood[n] *= self.params.null_prior
            posterior = predicted * likelihood
            total = posterior.sum()
            self.belief = posterior / total if total > 1e-12 else predicted
        else:
            self.belief = predicted

        return self.belief

    @classmethod
    def fit(cls, episodes: list[dict], *,
            committed_weight: float = DEFAULT_COMMITTED_WEIGHT) -> "TargetFilterParams":
        """Grid-searches rho_loose, rho_tight, sigma, null_prior to maximize
        the weighted log-likelihood of the true labeled target across episodes.

        Each episodes[i] is a dict with per-frame arrays: gaze_xy [T,2],
        candidate_px [T,n_cand,2], candidate_mask [T,n_cand], gaze_valid [T],
        phase_label [T] (teacher-forced phase for rho selection during
        fitting -- at deployment the caller passes its own filtered estimate
        instead, but fitting against ground truth avoids compounding the
        phase HMM's errors into the target fit), target_label [T] (candidate
        index or NULL_TARGET).

        NOTE on that teacher forcing: it is a mild approximation only while
        the phase head is accurate. It was NOT mild against the v7 phase
        model, whose GRASP/PLACE recalls were 0.15 and 0.06 -- rho_tight was
        being fit for a regime the runtime almost never entered. Worth
        re-checking once the phase rework lands, and worth switching to
        filtered-phase fitting if the gap is still large.

        committed_weight controls the objective's balance (see
        _combine_scores). Averaging log-probability over raw frames, as this
        did before, is dominated by NULL frames -- roughly 75% of all frames
        in this dataset -- so the search optimised null calibration while the
        eval harness reported accuracy on committed frames only. That
        mismatch is the most likely reason the stage-2 search rejected the
        alignment channel outright in v6 and v7 (both shipped with
        sigma_align = inf, i.e. the channel silently disabled at runtime).

        Scores every grid point at once via _score_grid rather than looping
        the whole filter once per combination (was the confirmed bottleneck
        at ~20-27 min for this grid): the recursion has identical structure
        for every combo, just different scalar parameters, so all combos are
        stepped through time together with the combo axis vectorized in
        numpy, replacing an O(n_combos * n_frames) Python loop with
        O(n_frames) Python-level steps.

        If episodes also carry ee_vel/ee_pos/candidate_world_pos (see
        model.py/train.py), sigma_align is fit as a SEPARATE, second stage
        afterward: a small grid search over sigma_align alone, holding
        rho_loose/rho_tight/sigma/null_prior fixed at their stage-1 optimum,
        rather than adding a 5th axis to the already-large joint grid above.
        This is a greedy/coordinate-wise search, not a full joint optimum --
        deliberate: the gaze-only fit is already a strong starting point,
        alignment is a secondary refinement, and it keeps the search small
        enough that _score (unvectorized) is fine for stage 2.
        """
        axes = _GRID_AXES
        grid = list(product(axes["rho_loose"], axes["rho_tight"], axes["sigma"], axes["null_prior"]))
        rho_loose_arr = np.array([g[0] for g in grid])
        rho_tight_arr = np.array([g[1] for g in grid])
        sigma_arr = np.array([g[2] for g in grid])
        null_prior_arr = np.array([g[3] for g in grid])

        scores = _score_grid(rho_loose_arr, rho_tight_arr, sigma_arr, null_prior_arr, episodes,
                             committed_weight=committed_weight)
        best_k = int(np.argmax(scores))
        params = TargetFilterParams(rho_loose=float(rho_loose_arr[best_k]), rho_tight=float(rho_tight_arr[best_k]),
                                     sigma=float(sigma_arr[best_k]), null_prior=float(null_prior_arr[best_k]))
        warn_if_on_grid_boundary(params)

        if all(k in ep for ep in episodes for k in ("ee_vel", "ee_pos", "candidate_world_pos")):
            best_sigma_align = np.inf
            best_align_score = _score(params, episodes, committed_weight=committed_weight)
            for sigma_align in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                candidate_params = replace(params, sigma_align=sigma_align)
                score = _score(candidate_params, episodes, committed_weight=committed_weight)
                if score > best_align_score:
                    best_align_score, best_sigma_align = score, sigma_align
            params = replace(params, sigma_align=best_sigma_align)

        return params


def _combine_scores(committed_sum, committed_n: int, null_sum, null_n: int, weight: float):
    """Weighted average of the committed-frame and null-frame GROUP mean
    log-probabilities.

    Deliberately not a plain per-frame mean. Committed frames are the
    minority (~25% of this dataset, and 0% on episodes where one arm is
    parked), so a per-frame mean lets null calibration dominate a search
    whose whole purpose is to discriminate between candidates. Taking the
    mean within each group first makes the objective independent of the
    split's null/committed ratio; `weight` then states the trade-off
    explicitly instead of letting the dataset decide it by accident.

    Works elementwise, so it serves both the scalar _score and the vectorized
    _score_grid (where the sums are [K] arrays over grid combos).
    """
    if committed_n == 0:
        return null_sum / max(null_n, 1)
    if null_n == 0:
        return committed_sum / committed_n
    return weight * (committed_sum / committed_n) + (1.0 - weight) * (null_sum / null_n)


def _score(params: TargetFilterParams, episodes: list[dict], *,
           committed_weight: float = DEFAULT_COMMITTED_WEIGHT) -> float:
    """Weighted log-probability the filter assigns to the true labeled
    target, for one parameter set -- a simple, unvectorized reference
    implementation. fit()'s main (rho/sigma/null_prior) search uses the
    vectorized _score_grid instead and only cross-checks against this one;
    fit()'s small secondary sigma_align search uses this directly (only a
    handful of candidate values, not worth extending _score_grid's
    broadcast machinery for). If ep carries ee_vel/ee_pos/candidate_world_pos,
    they're passed through to step() so the alignment channel actually gets
    exercised; older episode dicts without those keys score gaze-only,
    same as before this channel existed."""
    committed_sum, committed_n, null_sum, null_n = 0.0, 0, 0.0, 0
    for ep in episodes:
        f = TargetStickyFilter(params)
        f.reset()
        has_align = all(k in ep for k in ("ee_vel", "ee_pos", "candidate_world_pos"))
        for t in range(len(ep["gaze_valid"])):
            align_kwargs = ({"ee_vel": ep["ee_vel"][t], "ee_pos": ep["ee_pos"][t],
                              "candidate_world_pos": ep["candidate_world_pos"][t]}
                             if has_align else {})
            post = f.step(ep["gaze_xy"][t], ep["candidate_px"][t], ep["candidate_mask"][t],
                          bool(ep["gaze_valid"][t]), int(ep["phase_label"][t]), **align_kwargs)
            true_idx = ep["target_label"][t]
            is_null = true_idx == NULL_TARGET
            state_idx = post.shape[0] - 1 if is_null else true_idx
            log_p = float(np.log(max(post[state_idx], 1e-12)))
            if is_null:
                null_sum += log_p
                null_n += 1
            else:
                committed_sum += log_p
                committed_n += 1
    return float(_combine_scores(committed_sum, committed_n, null_sum, null_n, committed_weight))


def _score_grid(rho_loose_arr: np.ndarray, rho_tight_arr: np.ndarray,
                 sigma_arr: np.ndarray, null_prior_arr: np.ndarray,
                 episodes: list[dict], *,
                 committed_weight: float = DEFAULT_COMMITTED_WEIGHT) -> np.ndarray:
    """Vectorized equivalent of calling _score once per grid combination:
    scores ALL combos (K = len(rho_loose_arr)) at once per frame. Returns
    the combined objective per combo, shape [K]. Mirrors TargetStickyFilter
    .step()'s recursion exactly -- including the zeroing of invalid states in
    the predict step -- just with a leading K (combo) axis on
    belief/predicted/likelihood instead of a single filter instance.
    """
    K = len(rho_loose_arr)
    committed_sum, null_sum = np.zeros(K), np.zeros(K)
    committed_n, null_n = 0, 0

    for ep in episodes:
        gaze_xy = ep["gaze_xy"]
        candidate_px = ep["candidate_px"]
        candidate_mask = ep["candidate_mask"]
        gaze_valid = ep["gaze_valid"]
        phase_label = ep["phase_label"]
        target_label = ep["target_label"]

        T, n = candidate_mask.shape
        belief = None

        for t in range(T):
            mask_t = candidate_mask[t]
            valid = np.append(mask_t, True)  # null always valid
            n_valid = int(valid.sum())
            uniform = np.where(valid, 1.0 / n_valid, 0.0)  # [n+1]

            if belief is None:
                belief = np.tile(uniform, (K, 1))  # [K, n+1]

            tight = int(phase_label[t]) in TIGHT_PHASES
            rho = rho_tight_arr if tight else rho_loose_arr  # [K]
            predicted = rho[:, None] * belief + (1.0 - rho)[:, None] * uniform[None, :]  # [K, n+1]
            predicted = np.where(valid[None, :], predicted, 0.0)
            pred_total = predicted.sum(axis=1, keepdims=True)
            predicted = np.where(pred_total > 1e-12,
                                  predicted / np.maximum(pred_total, 1e-12), uniform[None, :])

            if gaze_valid[t]:
                d2 = np.sum((candidate_px[t] - gaze_xy[t][None, :]) ** 2, axis=1)  # [n]
                lik_cand = np.where(mask_t[None, :],
                                     np.exp(-d2[None, :] / (2 * sigma_arr[:, None] ** 2)),
                                     0.0)  # [K, n]
                likelihood = np.concatenate([lik_cand, null_prior_arr[:, None]], axis=1)  # [K, n+1]
                posterior = predicted * likelihood
                total = posterior.sum(axis=1, keepdims=True)  # [K, 1]
                belief = np.where(total > 1e-12, posterior / np.maximum(total, 1e-12), predicted)
            else:
                belief = predicted

            true_idx = target_label[t]
            is_null = true_idx == NULL_TARGET
            state_idx = n if is_null else true_idx
            log_p = np.log(np.maximum(belief[:, state_idx], 1e-12))
            if is_null:
                null_sum += log_p
                null_n += 1
            else:
                committed_sum += log_p
                committed_n += 1

    return _combine_scores(committed_sum, committed_n, null_sum, null_n, committed_weight)

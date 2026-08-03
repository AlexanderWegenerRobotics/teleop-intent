"""HMMIntentModel: the IntentModel this package exposes to the rest of the
repo (and, once trained, to the orchestrator as a SensorModule -- step()
already takes a SensorFrame and returns an IntentOutput, so no adapter is
needed). Wraps one PhaseHMM + one TargetStickyFilter per arm.

Phase and target parameters are pooled across both arms when fit (one set of
transition/emission/rho/sigma values), since 73 episodes split by arm is a
thinner fit than 73 episodes' worth of both arms' frames pooled, and there's
no a priori reason to expect left/right to behave differently. Belief STATE
is still separate per arm (two independent PhaseHMM/TargetStickyFilter
instances) -- only the learned parameters are shared.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from teleop_orchestrator.contracts import (
    SensorFrame, ArmIntent, IntentOutput, NULL_TARGET, Phase,
    CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES,
)

from metrics import classification_metrics

from ..base import TrainableIntentModel
from .features import (phase_features, PhaseFeatureState, world_ee_velocity, WorldEEState,
                       FEATURE_NAMES)
from .held import target_exclusion_mask
from .phase import (PhaseHMM, PhaseHMMParams, fit_emission_temperature, filter_episode,
                    sub_states_from_durations, DEFAULT_STD_FLOOR_REL, DEFAULT_TEMP_GRID,
                    DEFAULT_MAX_SUB_STATES)
from .target import TargetStickyFilter, TargetFilterParams, DEFAULT_COMMITTED_WEIGHT

_GAZE_IDX = [GLOBAL_FEATURE_NAMES.index("gaze_x"), GLOBAL_FEATURE_NAMES.index("gaze_y")]
_PX_IDX = [CANDIDATE_FEATURE_NAMES.index("px_u"), CANDIDATE_FEATURE_NAMES.index("px_v")]

SIDES = ("left", "right")


class HMMIntentModel(TrainableIntentModel):
    """Causal, non-deep intent model: a learned discrete HMM for phase and a
    learned phase-conditioned sticky filter for target, run independently
    per arm from shared, pooled parameters."""

    def __init__(self, phase_params: PhaseHMMParams | None = None,
                 target_params: TargetFilterParams | None = None):
        self.phase_params = phase_params
        self.target_params = target_params
        self._phase = {s: PhaseHMM(phase_params) for s in SIDES} if phase_params else {}
        self._target = {s: TargetStickyFilter(target_params) for s in SIDES} if target_params else {}
        self._prev_phase_state: dict[str, PhaseFeatureState | None] = {s: None for s in SIDES}
        self._prev_world_ee_state: dict[str, WorldEEState | None] = {s: None for s in SIDES}

    def reset(self) -> None:
        if not self._phase or not self._target:
            raise RuntimeError("HMMIntentModel has no fitted parameters -- call fit() or load() first")
        for s in SIDES:
            self._phase[s].reset()
            self._target[s].reset()
            self._prev_phase_state[s] = None
            self._prev_world_ee_state[s] = None

    def step(self, obs: SensorFrame) -> IntentOutput:
        gaze_xy = obs.global_features[_GAZE_IDX]
        candidate_px = obs.candidate_features[:, _PX_IDX]

        arms = {}
        null_probs = {}
        pool_masks = {}
        for side in SIDES:
            feats, self._prev_phase_state[side] = phase_features(obs, side, self._prev_phase_state[side])
            phase_posterior = self._phase[side].step(feats)
            phase_estimate = int(np.argmax(phase_posterior))

            # A single-gripper arm holding one object can't be reaching for any
            # OTHER object either -- exclude every object-type candidate from
            # the target pool while holding, not just the specific one held
            # (phase's min_candidate_dist above is unaffected: proximity to
            # *something* is still phase-informative even while holding it,
            # the category error only applies to "which candidate is the goal").
            target_mask = obs.candidate_mask & ~target_exclusion_mask(obs, side, phase_estimate)

            # ee_vel/ee_pos here are DELIBERATELY NOT phase's own ee_vel
            # (feats[:3]): phase uses O_T_EE (every-frame, base frame), while
            # the alignment channel needs true world-frame EE position to
            # compare against candidate_world_pos, sourced from the intent
            # log's ee_{side}_x/y/z instead -- which updates far less often,
            # so world_ee_velocity tracks its own staleness-aware state
            # rather than a plain per-frame finite difference. See
            # features.py's _EE_POS_SLICE/world_ee_velocity docstrings for
            # why these two velocity signals are intentionally different.
            ee_vel, ee_pos, self._prev_world_ee_state[side] = world_ee_velocity(
                obs, side, self._prev_world_ee_state[side])
            belief = self._target[side].step(gaze_xy, candidate_px, target_mask,
                                              obs.gaze_valid, phase_estimate,
                                              ee_vel=ee_vel, ee_pos=ee_pos,
                                              candidate_world_pos=obs.candidate_world_pos)
            null_prob = float(belief[-1])
            cand_probs = belief[:-1]
            remaining = 1.0 - null_prob
            if remaining > 1e-9:
                target_posterior = cand_probs / remaining
            else:
                n_valid = max(int(target_mask.sum()), 1)
                target_posterior = np.where(target_mask, 1.0 / n_valid, 0.0)

            arms[side] = ArmIntent(phase_posterior=phase_posterior, target_posterior=target_posterior)
            null_probs[side] = null_prob
            pool_masks[side] = target_mask

        # target_pool_mask reports which candidates were actually selectable
        # this frame, AFTER held.target_exclusion_mask. The eval harness needs
        # it to tell a genuine mistake from a label naming a candidate the
        # model was never allowed to choose -- and it cannot derive that
        # itself without hard-coding one model's exclusion rule.
        return IntentOutput(left=arms["left"], right=arms["right"],
                            extras={"target_null_prob": null_probs,
                                    "target_pool_mask": pool_masks})

    def fit(self, train_data: list[dict], val_data: list[dict], config: dict) -> dict:
        """train_data/val_data: lists of per-episode dicts, one per (episode,
        side) pair, each with:
          phase_features [T,n_feat], phase_labels [T] (Phase ints),
          gaze_xy [T,2], candidate_px [T,n_cand,2], candidate_mask [T,n_cand],
          gaze_valid [T], target_labels [T] (candidate index or NULL_TARGET).
        Building these from raw episodes is train.py's job, not this method's
        -- keeps this class free of any hdf5/ReplaySource dependency.

        Recognised config keys (all optional, defaults in phase.py/target.py):
          std_floor_rel      per-class std floor as a fraction of the feature's
                             global std -- the fix for v7's absolute 1e-3 floor
          tie_covariance     share one pooled covariance across phases, which
                             cancels the normalizer bias exactly (diagnostic)
          gaussian_only      force every feature through a Gaussian, i.e. the
                             pre-v8 emission, for ablation
          emission_temp      "auto" (default) selects on val; a float pins it,
                             and 1.0 reproduces v7's emission confidence
          emission_temp_grid / emission_temp_metric   sweep controls
          sub_states         "auto" (default) sizes each phase's sub-state
                             chain from its observed duration CV; a 5-tuple
                             pins it; None gives one per phase, i.e. the plain
                             geometric-duration HMM v9 and earlier were
          max_sub_states     cap on the auto sizing
          target_committed_weight   objective balance for the target fit

        The emission temperature is selected on VAL, not train. Everything
        else is closed-form MLE on train, so val is otherwise unused and the
        one genuinely tuned scalar belongs there.
        """
        feature_seqs = [d["phase_features"] for d in train_data]
        label_seqs = [d["phase_labels"] for d in train_data]

        n_feat = feature_seqs[0].shape[1]
        is_binary = np.zeros(n_feat, dtype=bool) if config.get("gaussian_only") else None

        # Chain lengths are read off the TRAINING labels' segment durations --
        # never val or test, since they are structural parameters of the model
        # like the transition counts, not tuned hyperparameters.
        sub_cfg = config.get("sub_states", "auto")
        history: dict = {}
        if sub_cfg == "auto":
            sub_states, sub_info = sub_states_from_durations(
                label_seqs, Phase.N_CLASSES,
                max_sub_states=int(config.get("max_sub_states", DEFAULT_MAX_SUB_STATES)))
            history["sub_state_selection"] = sub_info
        elif sub_cfg is None:
            sub_states = np.ones(Phase.N_CLASSES, dtype=int)
        else:
            sub_states = np.asarray(sub_cfg, dtype=int)

        self.phase_params = PhaseHMM.fit(
            feature_seqs, label_seqs,
            std_floor_rel=float(config.get("std_floor_rel", DEFAULT_STD_FLOOR_REL)),
            tie_covariance=bool(config.get("tie_covariance", False)),
            is_binary=is_binary,
            sub_states=sub_states,
        )
        history["sub_states"] = self.phase_params.chain_lengths.tolist()
        history["duration_stats"] = self.phase_params.duration_stats()

        temp_cfg = config.get("emission_temp", "auto")
        if temp_cfg == "auto":
            if not val_data:
                raise ValueError(
                    "emission_temp='auto' needs a val split to select against; "
                    "pass an explicit float, or point --val-split at labeled episodes")
            grid = tuple(config.get("emission_temp_grid", DEFAULT_TEMP_GRID))
            metric = str(config.get("emission_temp_metric", "macro_f1"))
            best_temp, table = fit_emission_temperature(self.phase_params, val_data,
                                                        grid=grid, metric=metric)
            self.phase_params = replace(self.phase_params, emission_temp=best_temp)
            history["emission_temp_table"] = table
            history["emission_temp_metric"] = metric
        else:
            self.phase_params = replace(self.phase_params, emission_temp=float(temp_cfg))
        history["emission_temp"] = self.phase_params.emission_temp
        history["normalizer_bias"] = self.phase_params.normalizer_bias().tolist()
        history["normalizer_bias_spread"] = float(np.ptp(self.phase_params.normalizer_bias()))

        target_episodes = [{
            "gaze_xy": d["gaze_xy"], "candidate_px": d["candidate_px"],
            "candidate_mask": d["candidate_mask"], "gaze_valid": d["gaze_valid"],
            "phase_label": d["phase_labels"], "target_label": d["target_labels"],
            # Alignment-channel inputs (see target.py) -- only included if
            # train.py's build_episode_arrays actually gathered them (older
            # episodes / pre-backfill data won't have candidate_world_pos);
            # TargetStickyFilter.fit() detects their absence and leaves
            # sigma_align disabled (np.inf) rather than erroring.
            **({"ee_vel": d["ee_vel"], "ee_pos": d["ee_pos"], "candidate_world_pos": d["candidate_world_pos"]}
               if "ee_vel" in d else {}),
        } for d in train_data]
        self.target_params = TargetStickyFilter.fit(
            target_episodes,
            committed_weight=float(config.get("target_committed_weight", DEFAULT_COMMITTED_WEIGHT)))

        self._phase = {s: PhaseHMM(self.phase_params) for s in SIDES}
        self._target = {s: TargetStickyFilter(self.target_params) for s in SIDES}

        if val_data:
            history.update(_phase_metrics(self.phase_params, val_data))
        return history

    def save(self, path: str) -> None:
        p = self.phase_params
        np.savez(
            path,
            transition=p.transition, prior=p.prior,
            emission_mean=p.emission_mean, emission_std=p.emission_std,
            # v8 emission fields; load() tolerates their absence so v1-v7
            # checkpoints still score without a migration step.
            bernoulli_p=p.bernoulli_p if p.bernoulli_p is not None else np.zeros_like(p.emission_mean),
            is_binary=p.binary_mask,
            emission_temp=float(p.emission_temp),
            tie_covariance=bool(p.tie_covariance),
            sub_states=p.chain_lengths,
            feature_names=np.array(p.feature_names),
            rho_loose=self.target_params.rho_loose, rho_tight=self.target_params.rho_tight,
            sigma=self.target_params.sigma, null_prior=self.target_params.null_prior,
            sigma_align=self.target_params.sigma_align,
        )

    def load(self, path: str) -> None:
        z = np.load(path)
        # Every field added after v1 is optional here: an older checkpoint
        # falls back to the dataclass defaults, which are chosen to reproduce
        # that checkpoint's original behaviour exactly (all-Gaussian emission,
        # temperature 1.0, sigma_align disabled). That keeps v1-v7 scoreable
        # against the new harness, which is the only way to show the rework
        # actually moved the numbers.
        feature_names = ([str(s) for s in z["feature_names"]] if "feature_names" in z
                         else list(FEATURE_NAMES))
        self.phase_params = PhaseHMMParams(
            transition=z["transition"], prior=z["prior"],
            emission_mean=z["emission_mean"], emission_std=z["emission_std"],
            bernoulli_p=z["bernoulli_p"] if "bernoulli_p" in z else None,
            is_binary=z["is_binary"].astype(bool) if "is_binary" in z else None,
            emission_temp=float(z["emission_temp"]) if "emission_temp" in z else 1.0,
            tie_covariance=bool(z["tie_covariance"]) if "tie_covariance" in z else False,
            # Absent -> one sub-state per phase, i.e. the plain geometric-duration
            # model every checkpoint up to v9 was.
            sub_states=z["sub_states"].astype(int) if "sub_states" in z else None,
            feature_names=feature_names,
        )
        n_ckpt = self.phase_params.emission_mean.shape[1]
        if n_ckpt != len(FEATURE_NAMES):
            raise ValueError(
                f"checkpoint {path} was fit on {n_ckpt} phase features but features.py now "
                f"produces {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES)}). Re-fit it with "
                f"models.hmm.train -- scoring a checkpoint against a different feature vector "
                f"silently misaligns every emission column.")
        sigma_align = float(z["sigma_align"]) if "sigma_align" in z else np.inf
        self.target_params = TargetFilterParams(rho_loose=float(z["rho_loose"]), rho_tight=float(z["rho_tight"]),
                                                 sigma=float(z["sigma"]), null_prior=float(z["null_prior"]),
                                                 sigma_align=sigma_align)
        self._phase = {s: PhaseHMM(self.phase_params) for s in SIDES}
        self._target = {s: TargetStickyFilter(self.target_params) for s in SIDES}


def _phase_metrics(params: PhaseHMMParams, episodes: list[dict]) -> dict:
    """Causal phase metrics on held-out episodes; a sanity readout, not the
    real scoring harness (that belongs in eval/, offline-only).

    Reports macro_f1 and macro_recall alongside accuracy because accuracy on
    its own is not a usable signal on this dataset -- see metrics.py. v7
    scored 0.686 accuracy on the right arm against 0.54 macro recall, with
    PLACE at 0.06.
    """
    true, pred = [], []
    for d in episodes:
        post = filter_episode(params, d["phase_features"])
        pred.extend(np.argmax(post, axis=1).tolist())
        true.extend(np.asarray(d["phase_labels"], dtype=int).tolist())
    m = classification_metrics(true, pred, Phase.N_CLASSES)
    return {
        "val_phase_accuracy": m["accuracy"],
        "val_phase_macro_recall": m["macro_recall"],
        "val_phase_macro_f1": m["macro_f1"],
        "val_phase_recall_per_class": {Phase.NAMES[i]: float(r) for i, r in enumerate(m["per_class_recall"])},
    }

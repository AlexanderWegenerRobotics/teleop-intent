"""Shared interface for intent-recognition models.

Every model (Bayesian filter, GRU, transformer, later vision-based variants)
implements this interface so the evaluation harness and the playback tool can
treat them interchangeably. The contract is online and causal: a model sees
observations up to the current timestep and returns posteriors over the
operator's intent.

Intent is predicted as three heads:

  phase   - what the operator is trying to do (approach, grasp, transport, ...)
  target  - which candidate they are currently acting toward
  arm     - which arm they intend to use

plus an optional secondary head for the destination a picked object is headed
to, which carries real information in stacking and assembly where the
destination is a free choice.

Phase is predicted rather than supplied as an input. Deriving it from gripper
width or object attachment would work in simulation but depends on privileged
information that will not exist on real hardware, and a hard categorical input
can be confidently wrong (mis-grasps, hovering, aborted reaches). Ground-truth
poses are used to build phase LABELS in simulation; they are never model INPUTS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NULL_TARGET = -1


class Phase:
    """Intended-action classes for the phase head.

    These describe what the operator is trying to do, not the observed state of
    the system, so a mis-grasp is still GRASP while the operator retries.
    """

    UNDECIDED = 0
    APPROACH = 1
    GRASP = 2
    TRANSPORT = 3
    PLACE = 4
    N_CLASSES = 5

    NAMES = {0: "undecided", 1: "approach", 2: "grasp", 3: "transport", 4: "place"}


class Arm:
    """Arm classes for the arm head, including a none state for timesteps where
    the operator has not committed to either arm."""

    LEFT = 0
    RIGHT = 1
    NONE = 2
    N_CLASSES = 3

    NAMES = {0: "left", 1: "right", 2: "none"}


@dataclass
class CandidateFeatures:
    """Per-timestep observation passed to a model.

    Holds relative, layout-invariant features for every candidate in the scene
    plus scene-level signals that are not tied to a candidate. Everything here
    must be observable on real hardware: no ground-truth object poses, no
    attachment flags, and no precomputed belief that would let a learned model
    copy the baseline filter's answer.
    """

    candidate_features: np.ndarray   # [n_candidates, n_candidate_features]
    mask: np.ndarray                 # [n_candidates] bool, True = real candidate
    candidate_types: np.ndarray      # [n_candidates] int, semantic type per slot
    global_features: np.ndarray      # [n_global_features] gripper, wrench, ee motion
    timestamp_ns: int                # capture time of this observation
    gaze_valid: bool = True          # whether gaze was tracked this frame

    @property
    def n_candidates(self) -> int:
        """Returns the number of candidate slots in this observation, including
        padded ones; use mask to select the real candidates."""
        return int(self.candidate_features.shape[0])


@dataclass
class IntentLabel:
    """Ground-truth intent for one timestep, derived in postprocess.

    Targets are references into the candidate set so the same representation
    holds across sorting, stacking, and assembly; only the semantics of what a
    candidate is changes per task. NULL_TARGET marks an axis that is undefined
    at this timestep, such as before the operator has committed to anything.
    """

    phase: int                              # Phase class
    target: int                             # index into candidate set, or NULL_TARGET
    arm: int                                # Arm class
    target_location: int = NULL_TARGET      # destination candidate, or NULL_TARGET
    segment_id: int = -1                    # manipulation segment this frame belongs to

    @property
    def has_target(self) -> bool:
        """Whether a committed target exists at this timestep; False during the
        undecided stretches the model must learn to express as uncertainty."""
        return self.target != NULL_TARGET


@dataclass
class IntentPrediction:
    """A model's output at one timestep.

    Carries full posteriors rather than hard decisions, because downstream
    assistance is uncertainty-gated and needs calibrated probabilities and
    entropy, not just an argmax. Posteriors over candidates are defined over
    all slots; padded slots must carry zero probability.
    """

    phase_posterior: np.ndarray                        # [Phase.N_CLASSES]
    target_posterior: np.ndarray                       # [n_candidates]
    arm_posterior: np.ndarray                          # [Arm.N_CLASSES]
    location_posterior: Optional[np.ndarray] = None    # [n_candidates] or None
    extras: dict = field(default_factory=dict)         # model-specific diagnostics

    def top_phase(self) -> int:
        """Returns the most likely phase class."""
        return int(np.argmax(self.phase_posterior))

    def top_target(self) -> int:
        """Returns the index of the most likely target candidate."""
        return int(np.argmax(self.target_posterior))

    def top_arm(self) -> int:
        """Returns the most likely arm class."""
        return int(np.argmax(self.arm_posterior))

    def phase_entropy(self) -> float:
        """Returns Shannon entropy (nats) of the phase posterior."""
        return _entropy(self.phase_posterior)

    def target_entropy(self) -> float:
        """Returns Shannon entropy (nats) of the target posterior; the primary
        uncertainty signal the nudge gate reads."""
        return _entropy(self.target_posterior)

    def arm_entropy(self) -> float:
        """Returns Shannon entropy (nats) of the arm posterior."""
        return _entropy(self.arm_posterior)

    def is_confident(self, max_target_entropy: float) -> bool:
        """Whether the target posterior is peaked enough to act on, used by
        assistance to suppress nudges while the operator is still deciding."""
        return self.target_entropy() <= max_target_entropy


def _entropy(p: np.ndarray) -> float:
    """Returns the Shannon entropy in nats of a probability vector, ignoring
    zero-probability entries."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class IntentModel(ABC):
    """Abstract base for all intent-recognition models.

    Subclasses implement online prediction over a candidate set. Stateful
    models (filter, recurrent nets) keep their belief or hidden state between
    calls to step(); reset() clears it at episode boundaries.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clears internal state so the model is ready for a new episode; called
        once before the first step() of each episode."""
        ...

    @abstractmethod
    def step(self, obs: CandidateFeatures) -> IntentPrediction:
        """Consumes one timestep's observation and returns the current intent
        posteriors, using only information up to now (strictly causal)."""
        ...

    def predict_episode(self, observations: list[CandidateFeatures]) -> list[IntentPrediction]:
        """Runs the model over a full episode by resetting then stepping in
        order; the online path the harness and playback tool both use."""
        self.reset()
        return [self.step(obs) for obs in observations]

    @property
    def name(self) -> str:
        """Short identifier used to label results and checkpoints."""
        return type(self).__name__

    @property
    def is_trainable(self) -> bool:
        """Whether this model has parameters to fit; False for the Bayesian
        baseline, True for learned models. Lets the harness skip training for
        models that have nothing to train."""
        return False


class TrainableIntentModel(IntentModel):
    """Base for models with learnable parameters (GRU, transformer).

    Adds fit, save, and load on top of the online prediction interface so
    learned models share one training and checkpointing contract.
    """

    @property
    def is_trainable(self) -> bool:
        """Learned models are trainable; the harness calls fit() before
        evaluation."""
        return True

    @abstractmethod
    def fit(self, train_data, val_data, config: dict) -> dict:
        """Trains on episode-split data and returns a history dict of losses and
        metrics; must never touch the held-out test set."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes model parameters to disk for later evaluation or
        deployment."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restores parameters previously written by save()."""
        ...
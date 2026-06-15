from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CandidateFeatures:
    """Per-timestep, per-candidate observation passed to a model.

    Holds the relative, layout-invariant features for every candidate visible
    at one timestep, plus a validity mask for padded/absent slots. This is the
    model input; it deliberately excludes any precomputed belief so learned
    models cannot copy the baseline filter's answer.
    """

    features: np.ndarray          # [n_candidates, n_features]
    mask: np.ndarray              # [n_candidates] bool, True = real candidate
    candidate_types: np.ndarray   # [n_candidates] int, semantic type per slot
    timestamp_ns: int             # capture time of this observation
    gaze_valid: bool = True       # whether gaze was tracked this frame


@dataclass
class IntentLabel:
    """Ground-truth intent for a timestep, derived in postprocess.

    The target is expressed as references into the candidate set so the same
    representation holds across sorting, stacking, and assembly. A value of -1
    marks an axis that is undefined at this timestep (e.g. no target yet).
    """

    target_object: int            # index into candidate set, or -1
    arm: int                      # 0 = left, 1 = right, -1 = undefined
    target_location: int          # index into candidate set, or -1
    phase: int = -1               # approach/grasp/transport/place, or -1


@dataclass
class IntentPrediction:
    """A model's output at one timestep.

    Carries full posteriors over each intent axis rather than hard decisions,
    because downstream assistance is uncertainty-gated and needs calibrated
    probabilities and entropy, not just an argmax.
    """

    object_posterior: np.ndarray              # [n_candidates], sums to 1
    arm_posterior: np.ndarray                 # [n_arms], sums to 1
    location_posterior: Optional[np.ndarray] = None   # [n_candidates] or None
    extras: dict = field(default_factory=dict)        # model-specific diagnostics

    def top_object(self) -> int:
        """Returns the index of the most likely target object."""
        return int(np.argmax(self.object_posterior))

    def object_entropy(self) -> float:
        """Returns Shannon entropy (nats) of the object posterior; the uncertainty signal the nudge gate reads."""
        p = self.object_posterior
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))


class IntentModel(ABC):
    """Abstract base for all intent-recognition models.

    Subclasses implement online prediction over a candidate set. Stateful
    models (filter, recurrent nets) keep their belief/hidden state between
    calls to step(); reset() clears it at episode boundaries.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clears any internal state so the model is ready for a new episode; called once before the first step() of each episode."""
        ...

    @abstractmethod
    def step(self, obs: CandidateFeatures) -> IntentPrediction:
        """Consumes one timestep's observation and returns the current intent posterior, using only information up to now (strictly causal)."""
        ...

    def predict_episode(self, observations: list[CandidateFeatures]) -> list[IntentPrediction]:
        """Runs the model over a full episode by resetting then stepping in order; the default online path the harness and playback tool both use."""
        self.reset()
        return [self.step(obs) for obs in observations]

    @property
    def is_trainable(self) -> bool:
        """Whether this model has parameters to fit; False for the Bayesian
        baseline, True for learned models. Lets the harness skip training for
        models that have nothing to train."""
        return False


class TrainableIntentModel(IntentModel):
    """Base for models with learnable parameters (GRU, transformer).

    Adds fit/save/load on top of the online prediction interface so learned
    models share one training and checkpointing contract.
    """

    @property
    def is_trainable(self) -> bool:
        """Learned models are trainable; the harness will call fit() before evaluation."""
        return True

    @abstractmethod
    def fit(self, train_data, val_data, config: dict) -> dict:
        """Trains the model on episode-split data and returns a history dict (losses, metrics); must not touch the held-out test set."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializes model parameters to disk for later evaluation or deployment."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restores parameters previously written by save()."""
        ...
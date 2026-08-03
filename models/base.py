"""Shared interface for intent-recognition models.

Every model (Bayesian filter, GRU, transformer, later vision-based variants)
implements this interface so the evaluation harness and the playback tool can
treat them interchangeably. The contract is online and causal: a model sees
observations up to the current timestep and returns posteriors over the
operator's intent.

step() takes a SensorFrame and returns an IntentOutput — the same types
teleop_orchestrator's runtime uses — so a class implementing IntentModel
already satisfies SensorModule[IntentOutput] and can be registered directly
with the Orchestrator once it's trained. Intent is predicted as two heads per
arm (arm is implicit — IntentOutput.left/right, not a predicted head):

  phase   - what the operator is trying to do (approach, grasp, transport, ...)
  target  - which candidate they are currently acting toward

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
from dataclasses import dataclass

from teleop_orchestrator.contracts import NULL_TARGET, Phase, SensorFrame, ArmIntent, IntentOutput

# Phase, NULL_TARGET, SensorFrame, ArmIntent, and IntentOutput are canonical in
# teleop_orchestrator.contracts; every module imports them from there so
# labels, training, and runtime never drift apart on what these mean.


@dataclass
class IntentLabel:
    """Ground-truth intent for one timestep, for one arm, derived in postprocess.

    Targets are references into the candidate set so the same representation
    holds across sorting, stacking, and assembly; only the semantics of what a
    candidate is changes per task. NULL_TARGET marks an axis that is undefined
    at this timestep, such as before the operator has committed to anything.
    """

    phase: int                              # Phase class
    target: int                             # index into candidate set, or NULL_TARGET
    target_location: int = NULL_TARGET      # destination candidate, or NULL_TARGET
    segment_id: int = -1                    # manipulation segment this frame belongs to

    @property
    def has_target(self) -> bool:
        """Whether a committed target exists at this timestep; False during the
        undecided (IDLE) stretches the model must learn to express as uncertainty."""
        return self.target != NULL_TARGET


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
    def step(self, obs: SensorFrame) -> IntentOutput:
        """Consumes one timestep's observation and returns the current intent
        posteriors for both arms, using only information up to now (strictly
        causal)."""
        ...

    def predict_episode(self, observations: list[SensorFrame]) -> list[IntentOutput]:
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

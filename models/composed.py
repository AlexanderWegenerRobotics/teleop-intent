"""Task-corrected intent: a perceptual model's posterior times a task prior.

    p(target | sensors, task) ∝ p(target | sensors) · p(target | task state)^beta

The base model (HMM, GRU, transformer) is untouched and never sees the prior; the prior only
rescales its target posterior over the pool the base model chose from. beta = 0 is the
ablation (identity), beta = 1 the full Bayesian product; something in between is expected to
work best, since the base posterior has already absorbed whatever order it could see in
training. Phase passes through unchanged. Implements the same IntentModel contract, so
eval/score.py and the orchestrator can use it in place of the base model.
"""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import SensorFrame, ArmIntent, IntentOutput

from .base import IntentModel
from .task_prior.model import TaskPrior

SIDES = ("left", "right")


class TaskCorrectedIntentModel(IntentModel):
    def __init__(self, base: IntentModel, prior: TaskPrior, beta: float = 0.5):
        self.base = base
        self.prior = prior
        self.beta = float(beta)
        self._trackers = {s: prior.tracker() for s in SIDES}

    @property
    def config(self) -> dict:
        """The base checkpoint's config (eval/score.py reads goal_targets off it)."""
        return getattr(self.base, "config", None) or {}

    @property
    def name(self) -> str:
        return f"{self.base.name}+task_prior(beta={self.beta:g})"

    def reset(self) -> None:
        self.base.reset()
        for t in self._trackers.values():
            t.reset()

    def load(self, path: str) -> None:
        self.base.load(path)

    def step(self, obs: SensorFrame) -> IntentOutput:
        out = self.base.step(obs)
        pools = (out.extras or {}).get("target_pool_mask", {})
        arms, priors, states = {}, {}, {}
        for side in SIDES:
            ai = out.arm(side)
            occupied = self._trackers[side].step(obs, side)
            pr = self.prior.candidate_prior(obs, occupied)
            base_post = np.asarray(ai.target_posterior, dtype=np.float64)
            n = min(len(base_post), len(pr))
            pool = pools.get(side)
            pool = np.asarray(pool, dtype=bool)[:n] if pool is not None else obs.candidate_mask[:n]
            post = base_post.copy()
            if self.beta != 0.0:
                scaled = base_post[:n] * np.power(pr[:n], self.beta)
                scaled = np.where(pool, scaled, 0.0)
                total = scaled.sum()
                if total > 1e-12:
                    post[:n] = scaled / total
            arms[side] = ArmIntent(phase_posterior=ai.phase_posterior, target_posterior=post,
                                   location_posterior=ai.location_posterior)
            priors[side] = pr
            states[side] = occupied
        extras = dict(out.extras or {})
        extras["task_prior"] = priors
        extras["task_state"] = states
        extras["base_target_posterior"] = {s: out.arm(s).target_posterior for s in SIDES}
        return IntentOutput(left=arms["left"], right=arms["right"], extras=extras)

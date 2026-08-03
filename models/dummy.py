"""Throwaway model for smoke-testing playback rendering, not a real model."""
import numpy as np
from models.base import IntentModel
from teleop_orchestrator.contracts import Phase, ArmIntent, IntentOutput


class DummyPeaked(IntentModel):
    def reset(self):
        pass

    def step(self, obs):
        n = obs.candidate_features.shape[0]
        phase_p = np.zeros(Phase.N_CLASSES); phase_p[Phase.TRANSPORT] = 0.9
        phase_p += 0.1 / Phase.N_CLASSES
        phase_p /= phase_p.sum()
        target_p = np.zeros(n)
        if n:
            target_p[0] = 0.8
            target_p += 0.2 / n
            target_p /= target_p.sum()
        left = ArmIntent(phase_posterior=phase_p, target_posterior=target_p)
        right = ArmIntent(phase_posterior=phase_p, target_posterior=target_p)
        return IntentOutput(left=left, right=right)

"""Minimal stand-in for teleop_orchestrator, so the HMM unit tests can run
anywhere (CI, a laptop without the C++ side built, a fresh clone).

install() must be called BEFORE importing anything under models.hmm. It only
registers the names those modules actually import; it is not an attempt to
mirror the real contracts, and nothing outside tests/ should use it. The
tests it enables are pure numpy assertions about filter arithmetic -- the
part that was wrong -- so a faithful SensorFrame is not needed.
"""

from __future__ import annotations

import sys
import types

import numpy as np

NULL_TARGET = -1
CAND_OBJECT = 0
CAND_BIN = 1

CANDIDATE_FEATURE_NAMES = ["px_u", "px_v", "dist_left", "dist_right"]
GLOBAL_FEATURE_NAMES = (
    ["gaze_x", "gaze_y", "gripper_width_left", "gripper_width_right"]
    + [f"fext_{s}_{a}" for s in ("left", "right") for a in ("fx", "fy", "fz")]
    + [f"ee_{s}_{a}" for s in ("left", "right") for a in ("x", "y", "z")]
)


class Phase:
    IDLE, APPROACH, GRASP, TRANSPORT, PLACE = 0, 1, 2, 3, 4
    N_CLASSES = 5
    NAMES = {0: "idle", 1: "approach", 2: "grasp", 3: "transport", 4: "place"}


class SensorFrame:  # pragma: no cover - placeholder, tests build arrays directly
    pass


class ArmIntent:  # pragma: no cover
    def __init__(self, phase_posterior, target_posterior):
        self.phase_posterior = phase_posterior
        self.target_posterior = target_posterior


class IntentOutput:  # pragma: no cover
    def __init__(self, left, right, extras=None):
        self.left, self.right, self.extras = left, right, extras or {}


def install() -> None:
    """Registers the stub under teleop_orchestrator.* in sys.modules."""
    if "teleop_orchestrator" in sys.modules:
        return

    root = types.ModuleType("teleop_orchestrator")
    contracts = types.ModuleType("teleop_orchestrator.contracts")
    features = types.ModuleType("teleop_orchestrator.contracts.features")
    sources = types.ModuleType("teleop_orchestrator.sources")

    for name, value in [("NULL_TARGET", NULL_TARGET), ("Phase", Phase), ("SensorFrame", SensorFrame),
                        ("ArmIntent", ArmIntent), ("IntentOutput", IntentOutput),
                        ("CANDIDATE_FEATURE_NAMES", CANDIDATE_FEATURE_NAMES),
                        ("GLOBAL_FEATURE_NAMES", GLOBAL_FEATURE_NAMES)]:
        setattr(contracts, name, value)
    features.CAND_OBJECT = CAND_OBJECT
    features.CAND_BIN = CAND_BIN
    features.candidate_names = lambda intent: []
    contracts.features = features
    sources.ReplaySource = object
    root.contracts = contracts
    root.sources = sources

    sys.modules["teleop_orchestrator"] = root
    sys.modules["teleop_orchestrator.contracts"] = contracts
    sys.modules["teleop_orchestrator.contracts.features"] = features
    sys.modules["teleop_orchestrator.sources"] = sources


__all__ = ["install", "Phase", "NULL_TARGET", "np"]

"""Task prior, occupancy tracker and the composed model, on hand-built frames.

Needs the real teleop_orchestrator contracts on the path (the stub in _stub_contracts.py has no
SensorFrame fields); skipped otherwise.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("teleop_orchestrator.contracts.frame")

from teleop_orchestrator.contracts import SensorFrame, GLOBAL_FEATURE_NAMES, ArmIntent, IntentOutput, Phase  # noqa: E402
from teleop_orchestrator.contracts.features import CAND_OBJECT, CAND_GOAL  # noqa: E402

from models.base import IntentModel  # noqa: E402
from models.composed import TaskCorrectedIntentModel  # noqa: E402
from models.task_prior import TaskPrior, TaskStateTracker, state_key  # noqa: E402

GOALS = ["side_pos", "centre"]
TYPES = ["peg_1", "peg_4"]
GOAL_POS = np.array([[0.868, 0.0, 0.705], [0.800, 0.0, 0.705]])
OBJECTS = ["peg_1", "peg_4", "spare_peg_1_b"]


def frame(obj_pos, ee, holding=False, held_name=None):
    """Three objects then two goals; distances from the given EE world position."""
    names = OBJECTS + GOALS
    world = np.concatenate([np.asarray(obj_pos, dtype=float), GOAL_POS])
    d = np.linalg.norm(world - np.asarray(ee)[None, :], axis=1)
    feats = np.column_stack([np.zeros(5), np.zeros(5), d, d])
    g = np.zeros(len(GLOBAL_FEATURE_NAMES))
    return SensorFrame(timestamp_ns=0, frame_id=0, candidate_features=feats,
                       candidate_mask=np.ones(5, dtype=bool),
                       candidate_types=np.array([CAND_OBJECT] * 3 + [CAND_GOAL] * 2),
                       global_features=g, gaze_valid=False, grasp_confirmed={"left": bool(holding)},
                       candidate_world_pos=world, ee_world_pos={"left": np.asarray(ee, dtype=float)},
                       candidate_names=names)


FAR = [0.3, 0.3, 0.3]
SCATTER = [[0.5, 0.2, 0.7], [0.5, -0.2, 0.7], [0.6, 0.2, 0.7]]


def test_tracker_follows_seating_holding_and_retreat():
    tr = TaskStateTracker(GOALS, TYPES, occupied_m=0.01, retreat_m=0.10)
    assert not tr.step(frame(SCATTER, FAR), "left").any()

    seated = [list(GOAL_POS[0]), SCATTER[1], SCATTER[2]]
    assert tr.step(frame(seated, FAR), "left").tolist() == [True, False]

    tr.reset()
    carried = [list(GOAL_POS[0] + [0.0, 0.0, 0.003]), SCATTER[1], SCATTER[2]]
    ee_at_goal = GOAL_POS[0] + [0.0, 0.0, 0.05]
    assert tr.step(frame(carried, ee_at_goal, holding=True), "left").tolist() == [False, False]

    assert tr.step(frame(seated, ee_at_goal, holding=False), "left").tolist() == [False, False]
    assert tr.step(frame(seated, FAR, holding=False), "left").tolist() == [True, False]
    assert tr.step(frame(seated, ee_at_goal, holding=False), "left").tolist() == [True, False]


def test_prior_table_and_roundtrip():
    prior = TaskPrior(GOALS, TYPES, alpha=0.5)
    prior.fit([(0, 0)] * 9 + [(0, 1)] + [(1, 1)] * 5)
    p0 = prior.goal_distribution(0)
    assert np.isclose(p0.sum(), 1.0) and p0[0] > 0.8
    assert np.isclose(prior.goal_distribution(1)[1], 5.5 / 6.0)
    assert prior.goal_distribution(3) is None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.npz")
        prior.save(path)
        back = TaskPrior.load(path)
    assert back.goal_names == GOALS and back.goal_types == TYPES
    assert np.allclose(back.goal_distribution(1), prior.goal_distribution(1))
    assert "side_pos" in back.format_table()


def test_candidate_prior_orders_objects_by_goal_type():
    prior = TaskPrior(GOALS, TYPES, alpha=0.5).fit([(0, 0)] * 9 + [(0, 1)])
    fr = frame(SCATTER, FAR)
    pr = prior.candidate_prior(fr, np.array([False, False]))
    assert pr[3] > pr[4]
    assert pr[0] > pr[1] and pr[2] > pr[1]
    assert np.isclose(pr[0], pr[2])
    assert np.isclose(pr[:3].sum(), 1.0) and np.isclose(pr[3:].sum(), 1.0)

    seated = [list(GOAL_POS[0]), SCATTER[1], SCATTER[2]]
    prior.fit([(1, 1)] * 5)
    pr = prior.candidate_prior(frame(seated, FAR), np.array([True, False]))
    assert pr[0] < pr[1] and pr[2] < pr[1]

    assert np.all(prior.candidate_prior(fr, np.array([False, True])) == 1.0)


class _Fixed(IntentModel):
    def __init__(self, post):
        self.post = np.asarray(post, dtype=float)
        self.config = {}

    def reset(self):
        pass

    def step(self, obs):
        ai = ArmIntent(phase_posterior=np.eye(Phase.N_CLASSES)[1], target_posterior=self.post.copy())
        return IntentOutput(left=ai, right=ai, extras={"target_pool_mask": {"left": obs.candidate_mask,
                                                                            "right": obs.candidate_mask}})


def test_composed_identity_at_beta_zero_and_reweights_otherwise():
    prior = TaskPrior(GOALS, TYPES, alpha=0.5).fit([(0, 0)] * 9 + [(0, 1)])
    base_post = [0.3, 0.3, 0.3, 0.05, 0.05]
    fr = frame(SCATTER, FAR)

    m0 = TaskCorrectedIntentModel(_Fixed(base_post), prior, beta=0.0)
    m0.reset()
    assert np.allclose(m0.step(fr).left.target_posterior, base_post)

    m1 = TaskCorrectedIntentModel(_Fixed(base_post), prior, beta=1.0)
    m1.reset()
    out = m1.step(fr)
    post = out.left.target_posterior
    assert np.isclose(post.sum(), 1.0)
    assert post[0] > post[1] and post[3] > post[4]
    assert "task_prior" in out.extras and "task_state" in out.extras
    assert np.allclose(out.extras["base_target_posterior"]["left"], base_post)
    assert np.allclose(out.left.phase_posterior, np.eye(Phase.N_CLASSES)[1])


def test_tracker_holds_state_through_object_dropout():
    tr = TaskStateTracker(GOALS, TYPES, occupied_m=0.01, retreat_m=0.10)
    seated = [list(GOAL_POS[0]), SCATTER[1], SCATTER[2]]
    assert tr.step(frame(seated, FAR), "left").tolist() == [True, False]
    fr = frame(seated, FAR)
    fr.candidate_mask[:3] = False
    assert tr.step(fr, "left").tolist() == [True, False]
    assert tr.step(frame(seated, FAR), "left").tolist() == [True, False]

"""World-state tracking for the task prior: which goals are filled, as seen from one arm."""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import SensorFrame, CANDIDATE_FEATURE_NAMES
from teleop_orchestrator.contracts.features import CAND_GOAL, CAND_OBJECT

from ..hmm.held import is_holding, held_part_index, occupied_goal_mask

_DIST_COL = {"left": CANDIDATE_FEATURE_NAMES.index("dist_left"),
             "right": CANDIDATE_FEATURE_NAMES.index("dist_right")}


def state_key(occupied: np.ndarray) -> int:
    """Bitmask over goal index for an occupancy vector; the table key of TaskPrior."""
    return int(sum(1 << g for g in np.flatnonzero(np.asarray(occupied, dtype=bool))))


def key_to_occupied(key: int, n_goals: int) -> np.ndarray:
    return np.array([(key >> g) & 1 == 1 for g in range(n_goals)], dtype=bool)


class TaskStateTracker:
    """Per-arm occupancy of the task goals, in goal-config order.

    A goal is filled when a type-matching part sits within occupied_m of it, with two exceptions
    that keep the current subgoal open: the part this arm is holding does not fill anything (its
    slot stays open while it is carried there), and the slot a part was just released into stays
    open while the end-effector is still within retreat_m of it. The subgoal ends when the arm
    leaves or grasps the next part. Only candidate names, world positions, the grasp signal and
    the EE-to-goal distance are used, all of which the hardware side provides. A frame with no
    object candidates at all (a dropped scene message) keeps the previous state.
    """

    def __init__(self, goal_names: list[str], goal_types: list[str], occupied_m: float = 0.01,
                 retreat_m: float = 0.20):
        self.goal_names = list(goal_names)
        self.goal_types = dict(zip(goal_names, goal_types))
        self.occupied_m = float(occupied_m)
        self.retreat_m = float(retreat_m)
        self.reset()

    def reset(self) -> None:
        self._held_name = None
        self._recent = None
        self._prev_occ = None
        self._last_state = None

    def _held(self, frame: SensorFrame, side: str) -> int | None:
        """Held part index, latched by name from the first held frame until release (as the GRU builder does)."""
        names = list(frame.candidate_names)
        if not is_holding(frame, side):
            self._held_name = None
            return None
        if self._held_name is None or self._held_name not in names:
            h = held_part_index(frame, side)
            if h is None or h >= len(names) or not names[h]:
                return h
            self._held_name = names[h]
        return names.index(self._held_name)

    def goal_rows(self, frame: SensorFrame) -> dict[str, int]:
        """Goal name -> candidate row for this frame."""
        names = list(frame.candidate_names)
        return {names[j]: int(j) for j in np.flatnonzero(frame.candidate_types == CAND_GOAL)
                if j < len(names) and names[j] in self.goal_types}

    def step(self, frame: SensorFrame, side: str) -> np.ndarray:
        """Returns occupied[n_goals] (bool) for this frame."""
        G = len(self.goal_names)
        rows = self.goal_rows(frame)
        if not rows or frame.candidate_world_pos is None:
            return np.zeros(G, dtype=bool)
        if not ((frame.candidate_types == CAND_OBJECT) & frame.candidate_mask).any():
            return self._last_state.copy() if self._last_state is not None else np.zeros(G, dtype=bool)
        names = list(frame.candidate_names)
        held = self._held(frame, side)
        own = names[held] if held is not None and held < len(names) and names[held] else None
        occ_rows = occupied_goal_mask(frame, own, self.goal_types, self.occupied_m)
        occ = np.array([bool(occ_rows[rows[g]]) if g in rows else False for g in self.goal_names])

        if held is not None:
            self._recent = None
        elif self._prev_occ is not None:
            newly = np.flatnonzero(occ & ~self._prev_occ)
            if newly.size:
                self._recent = int(newly[0])
        self._prev_occ = occ.copy()

        state = occ.copy()
        if self._recent is not None:
            row = rows.get(self.goal_names[self._recent])
            d = float(frame.candidate_features[row, _DIST_COL[side]]) if row is not None else np.inf
            if np.isfinite(d) and d <= self.retreat_m:
                state[self._recent] = False
            else:
                self._recent = None
        self._last_state = state.copy()
        return state

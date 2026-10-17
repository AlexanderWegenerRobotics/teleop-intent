"""The task prior itself: a smoothed count table p(next goal | occupancy state).

Fitted on labeled subgoals (one count per goal-labeled segment, keyed by the occupancy state at
its first frame), so it describes the operator's assembly order and nothing else. Pick objects
inherit the mass of the goals that take their type. A state never seen in training gives no
opinion (an all-ones prior, which is a no-op after renormalisation), so a deviating operator
is left to the perceptual model rather than fought.
"""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import SensorFrame
from teleop_orchestrator.contracts.features import CAND_OBJECT, CAND_GOAL

from ..hmm.held import part_type
from .state import TaskStateTracker, state_key, key_to_occupied


class TaskPrior:
    def __init__(self, goal_names: list[str], goal_types: list[str], alpha: float = 0.5,
                 occupied_m: float = 0.01, retreat_m: float = 0.20):
        self.goal_names = list(goal_names)
        self.goal_types = list(goal_types)
        self.alpha = float(alpha)
        self.occupied_m = float(occupied_m)
        self.retreat_m = float(retreat_m)
        self.counts: dict[int, np.ndarray] = {}

    @property
    def n_goals(self) -> int:
        return len(self.goal_names)

    def tracker(self) -> TaskStateTracker:
        return TaskStateTracker(self.goal_names, self.goal_types, self.occupied_m, self.retreat_m)

    def fit(self, subgoals: list[tuple[int, int]]) -> "TaskPrior":
        """subgoals: (state_key, goal_index) per labeled subgoal."""
        self.counts = {}
        for key, g in subgoals:
            self.counts.setdefault(int(key), np.zeros(self.n_goals))[int(g)] += 1.0
        return self

    def goal_distribution(self, key: int) -> np.ndarray | None:
        """p(goal | state) with a Dirichlet(alpha) floor, or None for a state never seen in training."""
        c = self.counts.get(int(key))
        if c is None:
            return None
        p = c + self.alpha
        return p / p.sum()

    def goal_index(self, name: str) -> int | None:
        return self.goal_names.index(name) if name in self.goal_names else None

    def candidate_prior(self, frame: SensorFrame, occupied: np.ndarray) -> np.ndarray:
        """Prior over this frame's candidate rows (1.0 where the task has no opinion).

        Goal rows carry p(goal | state). Object rows carry the mass of the goals that take their
        type, split among the unseated objects of that type; objects sitting in a slot (whether or
        not the state still counts that slot as open) and objects no goal takes get the floor. Each block is normalised on its own, so the prior reorders within
        objects and within goals but leaves the object-vs-goal decision to the perceptual model.
        """
        n = len(frame.candidate_mask)
        out = np.ones(n)
        p = self.goal_distribution(state_key(occupied))
        if p is None:
            return out
        names = list(frame.candidate_names) + [""] * max(0, n - len(frame.candidate_names))
        types, mask, world = frame.candidate_types, frame.candidate_mask, frame.candidate_world_pos
        goal_rows = {names[j]: j for j in np.flatnonzero(types == CAND_GOAL) if names[j] in self.goal_names}
        for g, name in enumerate(self.goal_names):
            if name in goal_rows:
                out[goal_rows[name]] = p[g]

        floor = float(p.min())
        objs = [int(o) for o in np.flatnonzero((types == CAND_OBJECT) & mask) if names[o]]
        if not objs:
            return out
        seated = {o: False for o in objs}
        if world is not None:
            for o in objs:
                t = part_type(names[o])
                for g, name in enumerate(self.goal_names):
                    if self.goal_types[g] == t and name in goal_rows:
                        d = np.linalg.norm(world[o] - world[goal_rows[name]])
                        if np.isfinite(d) and d < self.occupied_m:
                            seated[o] = True
        per_type: dict[str, int] = {}
        for o in objs:
            if not seated[o]:
                per_type[part_type(names[o])] = per_type.get(part_type(names[o]), 0) + 1
        w = np.full(len(objs), floor)
        for i, o in enumerate(objs):
            t = part_type(names[o])
            if seated[o]:
                continue
            m = sum(p[g] for g in range(self.n_goals) if self.goal_types[g] == t)
            if m > 0.0:
                w[i] = max(m / per_type[t], floor)
        w = w / w.sum()
        for i, o in enumerate(objs):
            out[o] = w[i]
        return out

    def table(self) -> list[tuple[list[str], int, np.ndarray]]:
        """(occupied goal names, n subgoals, p(goal | state)) per state, in increasing occupancy."""
        rows = []
        for key in sorted(self.counts, key=lambda k: (bin(k).count("1"), k)):
            occ = key_to_occupied(key, self.n_goals)
            rows.append(([self.goal_names[g] for g in np.flatnonzero(occ)], int(self.counts[key].sum()),
                         self.goal_distribution(key)))
        return rows

    def format_table(self) -> str:
        head = f"{'filled':<32}{'n':>5}" + "".join(f"{g:>12}" for g in self.goal_names)
        lines = [head]
        for occ_names, n, p in self.table():
            lines.append(f"{'{' + ', '.join(occ_names) + '}':<32}{n:>5}" + "".join(f"{v:>12.3f}" for v in p))
        return "\n".join(lines)

    def save(self, path: str) -> None:
        keys = np.array(sorted(self.counts), dtype=np.int64)
        counts = np.stack([self.counts[k] for k in keys]) if len(keys) else np.zeros((0, self.n_goals))
        np.savez(path, goal_names=np.array(self.goal_names), goal_types=np.array(self.goal_types),
                 keys=keys, counts=counts, alpha=self.alpha, occupied_m=self.occupied_m,
                 retreat_m=self.retreat_m)

    @classmethod
    def load(cls, path: str) -> "TaskPrior":
        z = np.load(path)
        prior = cls([str(g) for g in z["goal_names"]], [str(t) for t in z["goal_types"]],
                    alpha=float(z["alpha"]), occupied_m=float(z["occupied_m"]), retreat_m=float(z["retreat_m"]))
        prior.counts = {int(k): np.asarray(c, dtype=np.float64) for k, c in zip(z["keys"], z["counts"])}
        return prior

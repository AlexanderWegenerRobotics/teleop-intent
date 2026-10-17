"""Loads hand labels (labels/arm_{side}_phase, arm_{side}_target_name) into
training targets aligned with contracts.features' candidate indexing.

Labels only exist offline (they're the whole point of labeling/label_tool.py);
nothing here is on the runtime path — a deployed model never sees a name, only
the candidate index contracts.features assigns it.
"""

from __future__ import annotations

import h5py
import numpy as np

from teleop_orchestrator.contracts import NULL_TARGET
from teleop_orchestrator.contracts.features import candidate_names, N_EE_SLOTS

from models.base import IntentLabel


def _decode(v) -> str:
    """Decodes an hdf5 string cell (bytes or str) to a plain str."""
    return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)


def name_to_index(names: list[str], target_name: str) -> int:
    """Maps a labeled target name to its candidate index, or NULL_TARGET for
    'null' (or any name that isn't a current candidate — e.g. a stale label
    from a scene config change)."""
    if target_name in ("null", ""):
        return NULL_TARGET
    return names.index(target_name) if target_name in names else NULL_TARGET


def is_genuinely_labeled(ep: h5py.File) -> bool:
    """Whether this episode has real (non-default) hand labels.

    label_tool.py's save() creates labels/ and sets pass2_complete=True on
    EVERY save -- including an episode that was opened and saved without any
    actual annotation -- so "labels" in ep alone means "has been opened and
    saved at least once", not "has been labeled". This checks whether any
    frame actually differs from the IDLE/null default instead.
    """
    if "labels" not in ep:
        return False
    lbl = ep["labels"]
    for side in ("left", "right"):
        pk, tk = f"arm_{side}_phase", f"arm_{side}_target_name"
        if pk in lbl and (lbl[pk][:] != 0).any():
            return True
        if tk in lbl and any(v.rstrip(b"\x00") not in (b"null", b"") for v in lbl[tk][:]):
            return True
    return False


def labeled_sides(ep: h5py.File, sides=("left", "right")) -> list[str]:
    """Arms that carry phase labels in this episode (desk episodes only have the left arm)."""
    if "labels" not in ep:
        return []
    return [s for s in sides if f"arm_{s}_phase" in ep["labels"]]


def n_pickplace(intent: h5py.Group) -> int:
    """Object/bin slot capacity of the episode, the index where goal candidates start."""
    n = sum(1 for k in intent.keys() if k.startswith("slot_type_") and k[len("slot_type_"):].isdigit())
    return max(0, n - N_EE_SLOTS)


def load_arm_labels(ep: h5py.File, side: str, goal_targets: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (phase[T] int, target[T] int) for one arm of a labeled episode.

    target is already mapped into the same candidate index space
    contracts.features uses (see candidate_names) — never the raw label
    string — so it lines up 1:1 with a model's target_posterior indices.

    With goal_targets, frames in goal_targets["phases"] take the arm's goal
    label instead (index n_pickplace + goal position in the task config, the
    slot ReplaySource appends it at); a null goal there stays NULL_TARGET.
    """
    lbl = ep["labels"]
    intent = ep["observations"]["intent"]
    names = candidate_names(intent)

    phase = lbl[f"arm_{side}_phase"][:].astype(np.int64)
    raw_targets = [_decode(v) for v in lbl[f"arm_{side}_target_name"][:]]
    if goal_targets:
        n_obj = n_pickplace(intent)
        names = (list(names) + [""] * n_obj)[:n_obj] + list(goal_targets["names"])
        gk = f"arm_{side}_goal_name"
        goals = [_decode(v) for v in lbl[gk][:]] if gk in lbl else ["null"] * len(phase)
        in_goal_phase = np.isin(phase, list(goal_targets["phases"]))
        raw_targets = [g if in_goal_phase[t] else raw_targets[t] for t, g in enumerate(goals)]
    target = np.array([name_to_index(names, t) for t in raw_targets], dtype=np.int64)
    return phase, target


def label_at(phase: np.ndarray, target: np.ndarray, t: int) -> IntentLabel:
    """Builds an IntentLabel for one timestep from the arrays load_arm_labels returns."""
    return IntentLabel(phase=int(phase[t]), target=int(target[t]))

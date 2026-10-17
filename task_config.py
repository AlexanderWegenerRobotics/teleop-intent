"""Dataset and task settings shared by the train and eval entry points."""

from __future__ import annotations

import os

import yaml

from teleop_orchestrator.contracts import Phase

SORTING_DEFAULTS = {"n_phases": 5, "held_exclusion_phases": ["TRANSPORT", "PLACE"]}


def phase_ids(names) -> list[int]:
    """Maps phase names (any case) to their Phase ids."""
    return [int(getattr(Phase, str(n).upper())) for n in names]


def goal_positions(task: dict, base_dir: str = ".") -> list[list[float]] | None:
    """Goal positions from task.goals, or from the goals block of the task yaml named in task.goals_from."""
    goals = task.get("goals")
    if goals is None and task.get("goals_from"):
        path = task["goals_from"]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(base_dir, path)
        with open(path) as f:
            goals = (yaml.safe_load(f) or {}).get("goals")
    if not goals:
        return None
    if isinstance(goals, dict):
        goals = [g["pos"] if isinstance(g, dict) else g for g in goals.values()]
    return [[float(v) for v in g] for g in goals]


def goal_targets(task: dict, base_dir: str = ".") -> dict | None:
    """Goal-candidate settings (names, types, positions, label phases) when task.goal_target_phases is set."""
    phases = task.get("goal_target_phases")
    if not phases or not task.get("goals_from"):
        return None
    path = task["goals_from"]
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(base_dir, path)
    with open(path) as f:
        goals = (yaml.safe_load(f) or {}).get("goals") or {}
    return {"names": list(goals), "types": [str(g["type"]) for g in goals.values()],
            "pos": [[float(v) for v in g["pos"]] for g in goals.values()],
            "phases": phase_ids(phases), "occupied_m": float(task.get("goal_occupied_m", 0.01))}


def load_dataset(path: str) -> dict:
    """Reads a dataset yaml into store root, split files and task settings."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    data, split, task = cfg.get("data", {}), cfg.get("split", {}), cfg.get("task", {})
    out_dir = split.get("out_dir", "data/splits")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return {
        "store_root": data.get("store_root"),
        "train_split": os.path.join(out_dir, "train.txt"),
        "val_split": os.path.join(out_dir, "val.txt"),
        "test_split": os.path.join(out_dir, "test.txt"),
        "n_phases": int(task.get("n_phases", SORTING_DEFAULTS["n_phases"])),
        "held_exclusion_phases": phase_ids(task.get("held_exclusion_phases",
                                                    SORTING_DEFAULTS["held_exclusion_phases"])),
        "task": task.get("name", "sorting"),
        "goals": goal_positions(task, base_dir),
        "goal_targets": goal_targets(task, base_dir),
        "pin_held": bool(task.get("pin_held_object", False)),
    }


def add_dataset_args(ap, splits=("train", "val")) -> None:
    """Adds --dataset plus the overridable store/split/task arguments to a parser."""
    ap.add_argument("--dataset", default=None, help="dataset yaml, e.g. configs/dataset_fmb.yaml")
    ap.add_argument("--store-root", default=None)
    for s in splits:
        ap.add_argument(f"--{s}-split", default=None)
    ap.add_argument("--n-phases", type=int, default=None,
                    help="phase classes the model predicts (5 sorting, 7 FMB); default from --dataset")
    ap.add_argument("--no-goals", action="store_true",
                    help="ignore the dataset's goal positions (goal-distance feature ablation)")
    ap.add_argument("--no-goal-targets", action="store_true",
                    help="keep goals out of the candidate set (target head ranks pick objects only)")


def resolve(args, splits=("train", "val")) -> dict:
    """Merges CLI arguments over the dataset yaml over the sorting defaults."""
    base = load_dataset(args.dataset) if args.dataset else {
        "store_root": None, "train_split": "data/splits/train.txt", "val_split": "data/splits/val.txt",
        "test_split": "data/splits/test.txt", "n_phases": SORTING_DEFAULTS["n_phases"],
        "held_exclusion_phases": phase_ids(SORTING_DEFAULTS["held_exclusion_phases"]), "task": "sorting",
        "goals": None, "goal_targets": None, "pin_held": False}
    out = dict(base)
    if args.store_root:
        out["store_root"] = args.store_root
    for s in splits:
        v = getattr(args, f"{s}_split", None)
        if v:
            out[f"{s}_split"] = v
    if args.n_phases:
        out["n_phases"] = int(args.n_phases)
    if getattr(args, "no_goals", False):
        out["goals"] = None
    if getattr(args, "no_goal_targets", False):
        out["goal_targets"] = None
    if not out["store_root"]:
        raise SystemExit("no store root: pass --store-root or --dataset")
    return out

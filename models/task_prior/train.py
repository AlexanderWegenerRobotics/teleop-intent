"""Fits the task prior from labeled episodes and reports how predictable the order is.

Usage:
    python -m models.task_prior.train --dataset configs/dataset_fmb.yaml \
        --out checkpoints/task_prior/fmb_v1.npz [--alpha 0.5] [--retreat-m 0.10]

One count per labeled subgoal: each contiguous run of a goal-valued target label, keyed by the
occupancy state at its first frame. The val split is scored with the prior alone (top-1 and
log-likelihood of the labeled goal), which says how consistent the operator's order is before
any perceptual model is involved.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from teleop_orchestrator.contracts import NULL_TARGET  # noqa: E402
from teleop_orchestrator.sources import ReplaySource  # noqa: E402

import labels as label_loader  # noqa: E402
import task_config  # noqa: E402
from metrics import segments  # noqa: E402
from models.task_prior.model import TaskPrior  # noqa: E402
from models.task_prior.state import state_key  # noqa: E402

SIDES = ("left", "right")


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def episode_subgoals(hdf5_path: str, goal_targets: dict, prior: TaskPrior) -> list[tuple[int, int]]:
    """(state_key, goal_index) for every goal-labeled segment of every labeled arm, or [] if unlabeled."""
    with h5py.File(hdf5_path, "r") as ep:
        if not label_loader.is_genuinely_labeled(ep):
            return []
        per_side = {s: label_loader.load_arm_labels(ep, s, goal_targets)
                    for s in label_loader.labeled_sides(ep, SIDES)}
    src = ReplaySource(hdf5_path, load_images=False, goals=dict(zip(goal_targets["names"], goal_targets["pos"])))
    frames = [src.frame_at(t) for t in range(len(src))]
    src.close()
    out = []
    for side, (_, target) in per_side.items():
        tracker = prior.tracker()
        tracker.reset()
        keys = [state_key(tracker.step(fr, side)) for fr in frames]
        for lbl, s, _ in segments(target):
            if lbl == NULL_TARGET:
                continue
            fr = frames[s]
            n_obj = len(fr.candidate_mask) - prior.n_goals
            if lbl < n_obj or lbl >= len(fr.candidate_mask):
                continue
            out.append((keys[s], lbl - n_obj))
    return out


def gather(store_root: str, episode_ids: list[str], goal_targets: dict, prior: TaskPrior) -> list[tuple[int, int]]:
    out, n = [], 0
    for eid in sorted(episode_ids):
        path = os.path.join(store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        sg = episode_subgoals(path, goal_targets, prior)
        if sg:
            n += 1
            out.extend(sg)
    print(f"  {n} labeled episode(s) -> {len(out)} subgoals")
    return out


def evaluate(prior: TaskPrior, subgoals: list[tuple[int, int]], label: str) -> None:
    """Prior-alone top-1, mean log-likelihood and the share of subgoals the prior would call unlikely."""
    if not subgoals:
        print(f"{label}: no subgoals")
        return
    hits, ll, low, unseen = 0, [], 0, 0
    for key, g in subgoals:
        p = prior.goal_distribution(key)
        if p is None:
            unseen += 1
            continue
        hits += int(np.argmax(p) == g)
        ll.append(np.log(p[g]))
        low += int(p[g] < 0.1)
    n = len(subgoals) - unseen
    print(f"{label}: {len(subgoals)} subgoals, {unseen} in unseen states; on the rest: "
          f"prior top-1 {hits / max(n, 1):.3f}, mean log p(label) {np.mean(ll) if ll else float('nan'):.3f}, "
          f"p(label) < 0.1 on {low} ({100 * low / max(n, 1):.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    task_config.add_dataset_args(ap, splits=("train", "val"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet smoothing per goal")
    ap.add_argument("--retreat-m", type=float, default=0.20,
                    help="a just-filled slot stays open while the EE is within this distance of it")
    args = ap.parse_args()
    ds = task_config.resolve(args, splits=("train", "val"))
    gt = ds["goal_targets"]
    if not gt:
        raise SystemExit("the task prior needs goal targets (task.goal_target_phases + goals_from in the dataset yaml)")

    prior = TaskPrior(gt["names"], gt["types"], alpha=args.alpha, occupied_m=gt["occupied_m"],
                      retreat_m=args.retreat_m)
    print("Gathering train subgoals...")
    train = gather(ds["store_root"], read_split(ds["train_split"]), gt, prior)
    prior.fit(train)
    print("\np(next goal | filled slots), train split:")
    print(prior.format_table())
    print()
    evaluate(prior, train, "train")
    if os.path.exists(ds["val_split"]):
        print("Gathering val subgoals...")
        val = gather(ds["store_root"], read_split(ds["val_split"]), gt, prior)
        evaluate(prior, val, "val")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    prior.save(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

"""Aggregate scoring for any IntentModel against held-out labeled episodes.

Offline-only, per the architecture: reads ground truth via labels.py and
steps the model directly over ReplaySource frames -- no run.hdf5 intermediate
needed for this, though the same model/frames would produce one via the
Orchestrator if you wanted a persisted run to inspect later. Nothing here is
model-specific: any class implementing IntentModel (HMM, GRU, transformer)
works unchanged, since it only touches the IntentOutput/ArmIntent contract.

Usage:
    python eval/score.py --store-root /path/to/avatar --split test \
        --model models.hmm.model.HMMIntentModel --checkpoint checkpoints/hmm/v1.npz \
        --out-dir eval/reports/hmm_v1
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys

# Same reason as scripts/diagnose_emissions.py: `python eval/score.py` puts
# eval/ on sys.path, so the repo-root modules this imports (labels, metrics)
# would otherwise only resolve when the root happens to be on PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from teleop_orchestrator.contracts import (Phase, NULL_TARGET, CANDIDATE_FEATURE_NAMES,  # noqa: E402
                                            GLOBAL_FEATURE_NAMES)
from teleop_orchestrator.contracts.features import CAND_GOAL  # noqa: E402
from teleop_orchestrator.sources import ReplaySource  # noqa: E402

import labels as label_loader  # noqa: E402
import task_config  # noqa: E402
from metrics import classification_metrics, confusion_matrix, settle_latency, latency_summary  # noqa: E402
from models.composed import TaskCorrectedIntentModel  # noqa: E402
from models.task_prior.model import TaskPrior  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SIDES = ("left", "right")


def load_model(spec: str):
    """Instantiates an IntentModel from 'module.path.ClassName'."""
    module_path, class_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


_PX = [CANDIDATE_FEATURE_NAMES.index("px_u"), CANDIDATE_FEATURE_NAMES.index("px_v")]
_GAZE = [GLOBAL_FEATURE_NAMES.index("gaze_x"), GLOBAL_FEATURE_NAMES.index("gaze_y")]


GAZE_ON_RADIUS = 0.08  # normalized ray units (~65 px at desk_cam fx); gaze farther than this from every candidate is on none
SETTLE_HOLD = 10      # frames (1/3 s at 30 Hz) the prediction must stay on the label to count as settled
EARLY_FRAMES = 30     # the first second of a subgoal, where a nudge still has lead time
PRIOR_SURPRISE = 0.1  # p(label | task state) below this means the operator deviated from the learned order

RECORD_KEYS = ("true_phase", "pred_phase", "phase_entropy", "true_target", "pred_target", "target_entropy",
               "target_reachable", "target_is_goal", "gaze_candidate", "prior_at_label")


def gaze_candidate(frame) -> int:
    """Index of the visible candidate nearest the gaze point within GAZE_ON_RADIUS, else -1 (also when gaze is invalid)."""
    if not frame.gaze_valid:
        return -1
    uv = frame.candidate_features[:, _PX]
    vis = frame.candidate_mask & np.all(np.abs(uv) < 2.0, axis=1)
    if not vis.any():
        return -1
    d = np.where(vis, np.linalg.norm(uv - frame.global_features[_GAZE][None, :], axis=1), np.inf)
    k = int(np.argmin(d))
    return k if d[k] < GAZE_ON_RADIUS else -1


def model_goal_targets(model) -> dict | None:
    """Goal-candidate settings the checkpoint was fit with; frames and labels are built to match."""
    return (getattr(model, "config", None) or {}).get("goal_targets")


def score_episode(model, hdf5_path: str) -> dict | None:
    """Returns {side: {field: [T values]}} for one labeled episode, or None
    if it has no genuine hand labels."""
    goal_targets = model_goal_targets(model)
    with h5py.File(hdf5_path, "r") as ep:
        if not label_loader.is_genuinely_labeled(ep):
            return None
        per_side_labels = {side: label_loader.load_arm_labels(ep, side, goal_targets)
                           for side in label_loader.labeled_sides(ep, SIDES)}

    goals = dict(zip(goal_targets["names"], goal_targets["pos"])) if goal_targets else None
    src = ReplaySource(hdf5_path, load_images=False, goals=goals)
    model.reset()
    # target_reachable: whether the labelled target was even reachable given the candidate
    # mask that frame -- see the note in print_target_report. prior_at_label: the task
    # prior's value on the labelled candidate (NaN without a prior or without a label).
    records = {side: {k: [] for k in RECORD_KEYS} for side in per_side_labels}
    for t in range(len(src)):
        frame = src.frame_at(t)
        out = model.step(frame)
        for side in per_side_labels:
            ai = out.arm(side)
            phase_label, target_label = per_side_labels[side]
            r = records[side]
            r["true_phase"].append(int(phase_label[t]))
            r["pred_phase"].append(ai.top_phase())
            r["phase_entropy"].append(ai.phase_entropy())
            r["true_target"].append(int(target_label[t]))
            r["pred_target"].append(ai.top_target())
            r["target_entropy"].append(ai.target_entropy())
            # Reachability must be judged against the pool the MODEL chose
            # from, not against raw visibility. Using frame.candidate_mask
            # here caught only "the candidate was not logged this frame"
            # (~2% of committed frames) and missed the dominant cause,
            # held-object exclusion (~9%), so the reported reachable-only
            # accuracy was barely different from the raw one and gave a false
            # impression that the label disagreement had been accounted for.
            pool = (out.extras or {}).get("target_pool_mask", {}).get(side)
            if pool is None:
                pool = frame.candidate_mask
            lbl = int(target_label[t])
            r["target_reachable"].append(
                bool(lbl == NULL_TARGET or (0 <= lbl < len(pool) and bool(pool[lbl]))))
            r["target_is_goal"].append(bool(0 <= lbl < len(frame.candidate_types)
                                            and frame.candidate_types[lbl] == CAND_GOAL))
            r["gaze_candidate"].append(gaze_candidate(frame))
            pr = (out.extras or {}).get("task_prior", {}).get(side)
            r["prior_at_label"].append(float(pr[lbl]) if (pr is not None and 0 <= lbl < len(pr)) else float("nan"))
    src.close()
    for side, r in records.items():
        segs = settle_latency(r["true_target"], r["pred_target"], hold=SETTLE_HOLD, null_target=NULL_TARGET)
        for sg in segs:
            sg["is_goal"] = bool(r["target_is_goal"][sg["start"]])
            sg["prior_at_start"] = float(r["prior_at_label"][sg["start"]])
            sg["episode_id"] = os.path.basename(os.path.dirname(hdf5_path))
        r["segments"] = segs
    return records


def gather(model, store_root: str, episode_ids: list[str]) -> tuple[dict, list[dict]]:
    """Runs the model over every labeled episode, returns (pooled records
    per side, per-episode summary rows for the CSV)."""
    pooled = {side: {k: [] for k in RECORD_KEYS + ("segments",)} for side in SIDES}
    per_episode = []
    n_labeled = 0
    for eid in sorted(episode_ids):
        path = os.path.join(store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        records = score_episode(model, path)
        if records is None:
            continue
        n_labeled += 1
        for side in records:
            r = records[side]
            m = classification_metrics(r["true_phase"], r["pred_phase"], Phase.N_CLASSES)
            committed = np.array(r["true_target"]) != NULL_TARGET
            hit = np.array(r["pred_target"]) == np.array(r["true_target"])
            target_acc = np.mean(hit[committed]) if committed.any() else float("nan")
            gc = np.array(r["gaze_candidate"])
            off = committed & (gc >= 0) & (gc != np.array(r["true_target"]))
            is_goal = np.array(r["target_is_goal"], dtype=bool)
            goal_segs = [sg for sg in r["segments"] if sg["is_goal"]]
            pick_segs = [sg for sg in r["segments"] if not sg["is_goal"]]
            ls_goal, ls_pick = latency_summary(goal_segs, EARLY_FRAMES), latency_summary(pick_segs, EARLY_FRAMES)
            per_episode.append({
                "episode_id": eid, "side": side, "n_frames": len(r["true_phase"]),
                "n_committed_target": int(committed.sum()),
                # n_phases_present exposes the "this arm was parked all
                # episode" case that inflates a per-episode accuracy of 0.99
                # into something that looks like a result.
                "n_phases_present": int((m["support"] > 0).sum()),
                "phase_accuracy": m["accuracy"],
                "phase_macro_recall": m["macro_recall"],
                "phase_macro_f1": m["macro_f1"],
                "target_accuracy_given_committed": target_acc,
                "n_gaze_off_label": int(off.sum()),
                "target_accuracy_gaze_off_label": np.mean(hit[off]) if off.any() else float("nan"),
                "target_accuracy_goal": np.mean(hit[committed & is_goal]) if (committed & is_goal).any() else float("nan"),
                "n_goal_subgoals": ls_goal["n"],
                "goal_settle_rate": ls_goal["settle_rate"],
                "goal_settled_median_frames": ls_goal["settled_median"],
                "goal_early_frames_correct": ls_goal["early_frames_correct"],
                "n_pick_subgoals": ls_pick["n"],
                "pick_settle_rate": ls_pick["settle_rate"],
                "pick_settled_median_frames": ls_pick["settled_median"],
                "pick_early_frames_correct": ls_pick["early_frames_correct"],
            })
            for k in pooled[side]:
                pooled[side][k].extend(r[k])
    print(f"  scored {n_labeled} labeled episode(s)")
    return pooled, per_episode


def shown_classes(true, pred) -> int:
    """Number of phase rows to report: every class seen in labels or predictions, at least five."""
    top = 4
    if len(true):
        top = max(top, int(np.max(true)))
    if len(pred):
        top = max(top, int(np.max(pred)))
    return top + 1


def print_phase_report(side: str, true, pred) -> np.ndarray:
    """Confusion matrix plus per-class and macro scores.

    Macro F1 leads, and accuracy is reported beside it rather than alone:
    accuracy on this dataset is dominated by IDLE and by episodes where one
    arm is parked for the whole run, so a model can score well while never
    predicting the phases the module exists to detect. v7 scored 0.686
    accuracy on the right arm against 0.54 macro recall, with GRASP at 0.15
    and PLACE at 0.06 -- the accuracy number hid exactly the failure that
    mattered. Watch the 'predicted' column too: a class predicted on almost
    no frames is the signature of the emission normalizer bias described in
    models/hmm/phase.py, whatever its precision looks like.
    """
    n = shown_classes(true, pred)
    m = classification_metrics(true, pred, n)
    cm = m["confusion"]
    print(f"\n=== {side} arm: phase ===")
    print(f"macro F1: {m['macro_f1']:.3f}   macro recall: {m['macro_recall']:.3f}   "
          f"accuracy: {m['accuracy']:.3f}  (n={m['n_frames']})")
    header = "true\\pred".ljust(12) + "".join(Phase.NAMES[j].rjust(11) for j in range(n))
    print(header)
    for i in range(n):
        row = Phase.NAMES[i].ljust(12) + "".join(str(cm[i, j]).rjust(11) for j in range(n))
        print(row)
    print(f"{'class':<12}{'precision':>11}{'recall':>11}{'f1':>11}{'support':>11}{'predicted':>11}")
    for i in range(n):
        print(f"{Phase.NAMES[i]:<12}{m['per_class_precision'][i]:>11.3f}{m['per_class_recall'][i]:>11.3f}"
              f"{m['per_class_f1'][i]:>11.3f}{m['support'][i]:>11d}{m['predicted'][i]:>11d}")
    return cm


def print_target_report(side: str, true_target, pred_target, reachable=None, is_goal=None, true_phase=None,
                        gaze_cand=None) -> None:
    """Two accuracies, because one of them has a floor above zero error.

    On some frames the labelled target is a candidate the model's own pool
    excludes -- most often an object the arm is holding, which
    held.target_exclusion_mask removes on the grounds that a full gripper
    cannot be reaching for something. Those frames are UNWINNABLE by
    construction: every model in this repo assigns the labelled candidate
    exactly zero probability there, so they enter the headline accuracy as
    guaranteed errors.

    Both numbers are printed because they answer different questions. The
    all-committed figure is what the module delivers today, warts included.
    The reachable-only figure is what the model is actually capable of given
    the pool it is allowed to choose from, and is the only one comparable
    against a model trained with those frames excluded (models/gru/train.py
    drops them via ignore_index). Quoting one against the other understates
    or overstates by the unreachable rate, which on this dataset is ~11%.
    """
    true_target, pred_target = np.array(true_target), np.array(pred_target)
    committed = true_target != NULL_TARGET
    print(f"\n=== {side} arm: target (frames with a committed label) ===")
    if not committed.any():
        print("no committed-target frames in this split")
        return
    acc = np.mean(pred_target[committed] == true_target[committed])
    print(f"top-1 accuracy: {acc:.3f}  (n={int(committed.sum())} committed / {len(true_target)} total)")
    if reachable is not None:
        reachable = np.asarray(reachable, dtype=bool)
        ok = committed & reachable
        n_unreach = int((committed & ~reachable).sum())
        if n_unreach:
            acc_r = np.mean(pred_target[ok] == true_target[ok])
            print(f"top-1 accuracy on REACHABLE committed frames: {acc_r:.3f}  "
                  f"(n={int(ok.sum())}; {n_unreach} frames "
                  f"[{100 * n_unreach / committed.sum():.1f}%] label a candidate the pool "
                  f"excludes and cannot be got right by any model here)")
    if is_goal is not None and np.any(is_goal):
        is_goal = np.asarray(is_goal, dtype=bool)
        for name, sel in (("pick", committed & ~is_goal), ("goal", committed & is_goal)):
            if sel.any():
                hit = pred_target[sel] == true_target[sel]
                line = f"  {name} frames: top-1 {hit.mean():.3f}  (n={int(sel.sum())})"
                if reachable is not None and (sel & reachable).any():
                    line += f"   reachable-only {np.mean(pred_target[sel & reachable] == true_target[sel & reachable]):.3f}"
                print(line)
    if gaze_cand is not None:
        gaze_cand = np.asarray(gaze_cand)
        on = committed & (gaze_cand >= 0)
        for name, sel in (("gaze on label", on & (gaze_cand == true_target)),
                          ("gaze on other candidate", on & (gaze_cand != true_target))):
            if sel.any():
                print(f"  {name}: top-1 {np.mean(pred_target[sel] == true_target[sel]):.3f}  (n={int(sel.sum())})")
        if (committed & (gaze_cand < 0)).any():
            sel = committed & (gaze_cand < 0)
            print(f"  gaze invalid or on no candidate: top-1 {np.mean(pred_target[sel] == true_target[sel]):.3f}  (n={int(sel.sum())})")
    if true_phase is not None:
        true_phase = np.asarray(true_phase)
        parts = []
        for p in np.unique(true_phase[committed]):
            sel = committed & (true_phase == p)
            parts.append(f"{Phase.NAMES.get(int(p), p)} {np.mean(pred_target[sel] == true_target[sel]):.3f} (n={int(sel.sum())})")
        print("  by true phase: " + "   ".join(parts))


def print_leadtime_report(side: str, segs: list[dict]) -> None:
    """How fast the target prediction locks onto each subgoal, which is what a nudge's lead time rests on.

    One row per subgoal kind (pick: object through APPROACH/GRASP; goal: slot through
    TRANSPORT/INSERT/PLACE/CORRECT). Settled = the first frame from which the top-1 stays on the
    label for SETTLE_HOLD frames; early = top-1 accuracy inside the first EARLY_FRAMES of a run.
    """
    print(f"\n=== {side} arm: target lead time (per subgoal, frames at 30 Hz) ===")
    if not segs:
        print("no committed-target runs in this split")
        return
    print(f"{'kind':<6}{'n':>6}{'settle rate':>13}{'median':>9}{'mean':>9}{'p90':>9}{'first ok':>10}"
          f"{'early acc':>11}{'median len':>12}")
    for name, sel in (("pick", [s for s in segs if not s["is_goal"]]), ("goal", [s for s in segs if s["is_goal"]])):
        if not sel:
            continue
        m = latency_summary(sel, EARLY_FRAMES)
        print(f"{name:<6}{m['n']:>6}{m['settle_rate']:>13.3f}{m['settled_median']:>9.1f}{m['settled_mean']:>9.1f}"
              f"{m['settled_p90']:>9.1f}{m['first_correct_median']:>10.1f}{m['early_frames_correct']:>11.3f}"
              f"{np.median([s['length'] for s in sel]):>12.0f}")
    goal_pr = np.array([s["prior_at_start"] for s in segs if s["is_goal"]], dtype=float)
    if goal_pr.size and np.isfinite(goal_pr).any():
        known = goal_pr[np.isfinite(goal_pr)]
        low = int((known < PRIOR_SURPRISE).sum())
        print(f"  task prior at goal-run start: median p(label) {np.median(known):.3f}; "
              f"{low}/{known.size} runs below {PRIOR_SURPRISE} (operator deviated from the learned order)")


def print_calibration_report(side: str, entropy, true, pred, label: str) -> None:
    entropy, correct = np.array(entropy), np.array(true) == np.array(pred)
    print(f"\n=== {side} arm: {label} entropy calibration ===")
    print(f"mean entropy when correct:   {entropy[correct].mean():.3f}" if correct.any() else "no correct predictions")
    print(f"mean entropy when incorrect: {entropy[~correct].mean():.3f}" if (~correct).any() else "no incorrect predictions")


def save_plots(out_dir: str, pooled: dict) -> None:
    if plt is None:
        print("matplotlib not installed -- skipping plots (tables above still printed)")
        return
    os.makedirs(out_dir, exist_ok=True)
    for side in SIDES:
        r = pooled[side]
        if not r["true_phase"]:
            continue
        n = shown_classes(r["true_phase"], r["pred_phase"])
        names = [Phase.NAMES[i] for i in range(n)]
        cm = confusion_matrix(r["true_phase"], r["pred_phase"], n)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{side} arm phase confusion")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phase_confusion_{side}.png"), dpi=150)
        plt.close(fig)

        entropy = np.array(r["phase_entropy"])
        correct = np.array(r["true_phase"]) == np.array(r["pred_phase"])
        bins = np.quantile(entropy, np.linspace(0, 1, 11))
        bin_idx = np.clip(np.digitize(entropy, bins[1:-1]), 0, 9)
        bin_acc = [correct[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(10)]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(range(10), bin_acc, marker="o")
        ax.set_xlabel("phase entropy decile (low -> high)"); ax.set_ylabel("accuracy")
        ax.set_title(f"{side} arm phase calibration"); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phase_calibration_{side}.png"), dpi=150)
        plt.close(fig)
    print(f"  saved plots to {out_dir}")


def save_csv(out_dir: str, per_episode: list[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_episode.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_episode[0].keys()))
        w.writeheader()
        w.writerows(per_episode)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="dataset yaml, e.g. configs/dataset_fmb.yaml")
    ap.add_argument("--store-root", default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--split-file", default=None, help="override the split file the dataset/default names")
    ap.add_argument("--model", required=True, help="e.g. models.hmm.model.HMMIntentModel")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="eval/reports/latest")
    ap.add_argument("--task-prior", default=None,
                    help="task prior .npz (models.task_prior.train); wraps the model in TaskCorrectedIntentModel")
    ap.add_argument("--prior-beta", type=float, default=0.5,
                    help="exponent on the task prior (0 = ablation, 1 = full Bayesian product)")
    args = ap.parse_args()

    ds = task_config.load_dataset(args.dataset) if args.dataset else {}
    store_root = args.store_root or ds.get("store_root")
    if not store_root:
        raise SystemExit("no store root: pass --store-root or --dataset")
    split_file = args.split_file or ds.get(f"{args.split}_split") or f"data/splits/{args.split}.txt"
    episode_ids = read_split(split_file)
    print(f"Scoring against {args.split} split ({len(episode_ids)} episode ids listed)...")

    model = load_model(args.model)
    model.load(args.checkpoint)
    if args.task_prior:
        model = TaskCorrectedIntentModel(model, TaskPrior.load(args.task_prior), beta=args.prior_beta)
        print(f"  task prior {args.task_prior}, beta {args.prior_beta:g}")

    pooled, per_episode = gather(model, store_root, episode_ids)
    if not per_episode:
        raise SystemExit("No labeled episodes found in this split -- check --store-root and split-file")

    for side in SIDES:
        r = pooled[side]
        if not r["true_phase"]:
            continue
        print_phase_report(side, r["true_phase"], r["pred_phase"])
        print_target_report(side, r["true_target"], r["pred_target"], r.get("target_reachable"),
                            r.get("target_is_goal"), r["true_phase"], r.get("gaze_candidate"))
        print_leadtime_report(side, r["segments"])
        print_calibration_report(side, r["phase_entropy"], r["true_phase"], r["pred_phase"], "phase")

    save_plots(args.out_dir, pooled)
    save_csv(args.out_dir, per_episode)


if __name__ == "__main__":
    main()

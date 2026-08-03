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

from teleop_orchestrator.contracts import Phase, NULL_TARGET  # noqa: E402
from teleop_orchestrator.sources import ReplaySource  # noqa: E402

import labels as label_loader  # noqa: E402
from metrics import classification_metrics, confusion_matrix  # noqa: E402

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


def score_episode(model, hdf5_path: str) -> dict | None:
    """Returns {side: {field: [T values]}} for one labeled episode, or None
    if it has no genuine hand labels."""
    with h5py.File(hdf5_path, "r") as ep:
        if not label_loader.is_genuinely_labeled(ep):
            return None
        per_side_labels = {side: label_loader.load_arm_labels(ep, side) for side in SIDES}

    src = ReplaySource(hdf5_path, load_images=False)
    model.reset()
    records = {side: {"true_phase": [], "pred_phase": [], "phase_entropy": [],
                       "true_target": [], "pred_target": [], "target_entropy": [],
                       # Whether the labelled target was even reachable given
                       # the candidate mask that frame -- see the note in
                       # print_target_report.
                       "target_reachable": []}
               for side in SIDES}
    for t in range(len(src)):
        frame = src.frame_at(t)
        out = model.step(frame)
        for side in SIDES:
            ai = out.arm(side)
            phase_label, target_label = per_side_labels[side]
            r = records[side]
            r["true_phase"].append(int(phase_label[t]))
            r["pred_phase"].append(ai.top_phase())
            r["phase_entropy"].append(ai.phase_entropy())
            r["true_target"].append(int(target_label[t]))
            r["pred_target"].append(ai.top_target())
            r["target_entropy"].append(ai.target_entropy())
            lbl = int(target_label[t])
            r["target_reachable"].append(
                bool(lbl == NULL_TARGET or (0 <= lbl < frame.candidate_mask.shape[0]
                                            and frame.candidate_mask[lbl])))
    src.close()
    return records


def gather(model, store_root: str, episode_ids: list[str]) -> tuple[dict, list[dict]]:
    """Runs the model over every labeled episode, returns (pooled records
    per side, per-episode summary rows for the CSV)."""
    pooled = {side: {k: [] for k in ("true_phase", "pred_phase", "phase_entropy",
                                      "true_target", "pred_target", "target_entropy",
                                      "target_reachable")}
              for side in SIDES}
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
        for side in SIDES:
            r = records[side]
            m = classification_metrics(r["true_phase"], r["pred_phase"], Phase.N_CLASSES)
            committed = np.array(r["true_target"]) != NULL_TARGET
            target_acc = (np.mean(np.array(r["pred_target"])[committed] == np.array(r["true_target"])[committed])
                          if committed.any() else float("nan"))
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
            })
            for k in pooled[side]:
                pooled[side][k].extend(r[k])
    print(f"  scored {n_labeled} labeled episode(s)")
    return pooled, per_episode


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
    n = Phase.N_CLASSES
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


def print_target_report(side: str, true_target, pred_target, reachable=None) -> None:
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
        cm = confusion_matrix(r["true_phase"], r["pred_phase"], Phase.N_CLASSES)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(Phase.N_CLASSES)); ax.set_xticklabels(Phase.NAMES.values(), rotation=45, ha="right")
        ax.set_yticks(range(Phase.N_CLASSES)); ax.set_yticklabels(Phase.NAMES.values())
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{side} arm phase confusion")
        for i in range(Phase.N_CLASSES):
            for j in range(Phase.N_CLASSES):
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
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--split-file", default=None, help="override the default data/splits/<split>.txt path")
    ap.add_argument("--model", required=True, help="e.g. models.hmm.model.HMMIntentModel")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="eval/reports/latest")
    args = ap.parse_args()

    split_file = args.split_file or f"data/splits/{args.split}.txt"
    episode_ids = read_split(split_file)
    print(f"Scoring against {args.split} split ({len(episode_ids)} episode ids listed)...")

    model = load_model(args.model)
    model.load(args.checkpoint)

    pooled, per_episode = gather(model, args.store_root, episode_ids)
    if not per_episode:
        raise SystemExit("No labeled episodes found in this split -- check --store-root and split-file")

    for side in SIDES:
        r = pooled[side]
        print_phase_report(side, r["true_phase"], r["pred_phase"])
        print_target_report(side, r["true_target"], r["pred_target"], r.get("target_reachable"))
        print_calibration_report(side, r["phase_entropy"], r["true_phase"], r["pred_phase"], "phase")

    save_plots(args.out_dir, pooled)
    save_csv(args.out_dir, per_episode)


if __name__ == "__main__":
    main()

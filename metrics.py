"""Classification metrics shared by model selection and the eval harness.

Lives at repo root next to labels.py/common.py (not inside models/ or eval/)
because both sides need it: models/hmm/phase.py selects the emission
temperature by maximising macro-F1 on val, and eval/score.py reports the same
number on test. Defining it once means a temperature chosen during fitting and
the score printed afterwards can never be measuring subtly different things.

WHY MACRO, NOT ACCURACY. Frame-level accuracy on this dataset is close to
meaningless as a headline: IDLE dominates, and several test episodes have one
arm parked for their whole duration (three of eleven left-arm episodes in the
v7 report have zero committed-target frames and score 0.84-0.99 by predicting
IDLE forever). The v7 right arm scored 0.686 accuracy against 0.54 macro
recall, with the two phases the module actually exists to detect -- GRASP and
PLACE -- at 0.15 and 0.06 recall. Any metric that lets a model score well
while never predicting PLACE is the wrong metric to tune against.

macro_f1 is the recommended selection metric over macro_recall: recall alone
is trivially inflated by over-predicting a rare class, which is exactly the
failure mode an emission-temperature sweep can wander into from the other
direction.
"""

from __future__ import annotations

import numpy as np


def segments(labels) -> list[tuple[int, int, int]]:
    """Contiguous runs of a label sequence as (label, start, end_exclusive).

    Canonical implementation: the phase model (sub-state sizing), the eval
    harness (onset latency) and the duration diagnostics all need it, and
    three private copies would eventually disagree about edge cases.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [labels.size]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def segment_durations(labels, drop_censored: bool = True) -> list[tuple[int, int]]:
    """(label, duration) per contiguous run.

    The first and last run of a sequence are dropped by default: they were cut
    short by the recording starting and stopping, not by a real transition, so
    keeping them biases the mean down and inflates the spread -- the two
    statistics duration modelling is fitted against.
    """
    runs = [(k, e - s) for k, s, e in segments(labels)]
    if drop_censored and len(runs) > 2:
        runs = runs[1:-1]
    return runs


def confusion_matrix(true, pred, n_classes: int) -> np.ndarray:
    """[n_classes, n_classes] counts, cm[true, pred]."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(true, dtype=int), np.asarray(pred, dtype=int)), 1)
    return cm


def classification_metrics(true, pred, n_classes: int) -> dict:
    """Accuracy plus per-class precision/recall/F1 and their macro averages.

    Macro averages are taken over classes with non-zero support only, so a
    split that happens to contain no frames of some phase doesn't drag the
    average toward zero for a model that was never given a chance to predict
    it. Classes with support but zero predictions score precision 0 (not NaN)
    -- never predicting a class is a real failure and must be penalised, which
    is the whole point of using this metric here.
    """
    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    cm = confusion_matrix(true, pred, n_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm)

    recall = np.divide(diag, support, out=np.full(n_classes, np.nan), where=support > 0)
    precision = np.divide(diag, predicted, out=np.zeros(n_classes), where=predicted > 0)
    precision = np.where(support > 0, precision, np.nan)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=np.nan_to_num(denom) > 0)
    f1 = np.where(support > 0, f1, np.nan)

    present = support > 0
    return {
        "accuracy": float((true == pred).mean()) if true.size else float("nan"),
        "macro_recall": float(np.nanmean(recall[present])) if present.any() else float("nan"),
        "macro_precision": float(np.nanmean(precision[present])) if present.any() else float("nan"),
        "macro_f1": float(np.nanmean(f1[present])) if present.any() else float("nan"),
        "per_class_recall": recall,
        "per_class_precision": precision,
        "per_class_f1": f1,
        "support": support,
        "predicted": predicted,
        "confusion": cm,
        "n_frames": int(true.size),
    }


def target_segments(true_target, null_target: int = -1) -> list[tuple[int, int, int]]:
    """Contiguous runs of a committed target label as (target, start, end_exclusive), nulls dropped.

    A run spans phases: an object stays the target through APPROACH and GRASP, a goal through
    TRANSPORT, INSERT and PLACE, so one run is one subgoal from the operator's point of view.
    """
    return [(k, s, e) for k, s, e in segments(true_target) if k != null_target]


def settle_latency(true_target, pred_target, hold: int = 10, null_target: int = -1) -> list[dict]:
    """Per subgoal: how long the prediction takes to lock onto the labeled target.

    first_correct is the first frame the prediction equals the label; settled is the first
    frame from which it stays equal for `hold` frames (the whole run when the run is shorter).
    Both are offsets from the run's start, or None when it never happens. This is the number
    a nudge that needs lead time depends on; frame accuracy hides it.
    """
    true_target, pred_target = np.asarray(true_target), np.asarray(pred_target)
    out = []
    for k, s, e in target_segments(true_target, null_target):
        hit = pred_target[s:e] == k
        first = int(np.argmax(hit)) if hit.any() else None
        h = min(int(hold), e - s)
        settled = None
        if h > 0:
            run = np.convolve(hit.astype(int), np.ones(h, dtype=int), mode="valid") == h
            if run.any():
                settled = int(np.argmax(run))
        out.append({"target": int(k), "start": int(s), "end": int(e), "length": int(e - s),
                    "first_correct": first, "settled": settled, "hit": hit})
    return out


def latency_summary(segs: list[dict], early: int = 30) -> dict:
    """Settle rate, latency quantiles (frames) and accuracy inside the first `early` frames of a run."""
    n = len(segs)
    if n == 0:
        return {"n": 0, "settle_rate": float("nan"), "settled_median": float("nan"), "settled_mean": float("nan"),
                "settled_p90": float("nan"), "first_correct_median": float("nan"), "early_frames_correct": float("nan")}
    settled = np.array([s["settled"] for s in segs if s["settled"] is not None], dtype=float)
    first = np.array([s["first_correct"] for s in segs if s["first_correct"] is not None], dtype=float)
    early_hits, early_total = 0.0, 0
    for s in segs:
        w = min(int(early), s["length"])
        early_total += w
        early_hits += float(np.sum(s["hit"][:w]))
    return {
        "n": n,
        "settle_rate": float(len(settled) / n),
        "settled_median": float(np.median(settled)) if settled.size else float("nan"),
        "settled_mean": float(settled.mean()) if settled.size else float("nan"),
        "settled_p90": float(np.percentile(settled, 90)) if settled.size else float("nan"),
        "first_correct_median": float(np.median(first)) if first.size else float("nan"),
        "early_frames_correct": float(early_hits / early_total) if early_total else float("nan"),
    }

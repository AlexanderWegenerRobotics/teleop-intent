"""Are the remaining phase errors about TIMING or about REPRESENTATION?

This is the experiment that should decide HSMM versus GRU, rather than
choosing on intuition. Both stories predict the same aggregate macro F1; they
predict very different distributions of WHERE the errors sit.

  TIMING / DURATION (-> HSMM). Errors cluster within a few frames of true
  phase boundaries: the model identifies the right phases but enters and
  leaves them late, and lets short phases be swallowed. The fix is explicit
  duration modelling -- a first-order HMM's state durations are geometric,
  which is badly wrong for a 15-frame grasp sitting between two long phases,
  and the only lever it has (the self-transition) also controls how hard the
  emission has to fight to move at all. Segment interiors would already be
  accurate, and no amount of extra representational power would help them.

  REPRESENTATION (-> GRU / more features). Errors are spread through segment
  interiors: mid-grasp frames genuinely look like mid-idle frames given the
  current feature vector, and no duration model can fix that because the
  evidence is not there. This is the case where memory beyond the current
  frame ("the gripper closed 300 ms ago") starts to earn its complexity.

Readouts, in increasing order of how much weight to put on them:

  [A] macro F1 by ABSOLUTE distance to the nearest boundary. Read the support
      columns, not the macro F1. This bucketing CANNOT settle the question:
      a phase whose segments average 54 frames has no frame further than 27
      from a boundary, so it is structurally absent from the far buckets and
      the "interior" score silently becomes a score on the long phases alone.
      Kept because the accuracy trend across buckets is still informative and
      because the confound is worth seeing rather than being protected from.
  [A2] macro F1 by RELATIVE position within the segment. Every phase spans
      [0, 1) whatever its length, so the class mixture stays roughly constant
      and buckets are comparable. This is the frame-level number to read.
  [B] onset latency -- for every true phase segment, how many frames until
      the model first predicts it, and how often it never does at all. This
      is also the metric the nudging application actually cares about: a
      phase detected 40 frames late is useless for shaping the operator's
      perception in time, however good it looks frame-averaged.
  [C] segment-count inflation -- predicted segments per true segment. Well
      above 1.0 means the belief is chattering inside segments, which the
      frame-level score partly hides.
  [D] THE VERDICT, at segment level: is the correct phase ever predicted
      inside each true segment? That separates "cannot recognise this phase"
      (representation) from "recognises it and mis-places its edges"
      (timing) without frame-count imbalance getting a vote. Latency is
      normalised by segment length here for the same reason [A2] exists.

Usage mirrors score.py:

    python eval/boundary_analysis.py --store-root /path/to/avatar --split test \
        --model models.hmm.model.HMMIntentModel --checkpoint checkpoints/hmm/v9.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # eval/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np  # noqa: E402

from teleop_orchestrator.contracts import Phase  # noqa: E402

from metrics import classification_metrics, segments  # noqa: E402
from score import load_model, read_split, score_episode, SIDES  # noqa: E402

# Frame-count buckets for [A]. At ~30 Hz these are roughly: within 0.17 s of a
# boundary, within 0.5 s, within 1 s, and everything further in.
_BUCKETS = [(0, 5), (6, 15), (16, 30), (31, 10 ** 9)]


def distance_to_boundary(labels: np.ndarray) -> np.ndarray:
    """[T] frames from each index to the nearest label change.

    Computed by a forward and a backward pass over the boundary positions, so
    it is exact rather than a windowed approximation, and O(T)."""
    labels = np.asarray(labels)
    T = len(labels)
    is_boundary = np.zeros(T, dtype=bool)
    changed = labels[1:] != labels[:-1]
    is_boundary[:-1] |= changed          # last frame of the outgoing segment
    is_boundary[1:] |= changed           # first frame of the incoming segment
    if not is_boundary.any():
        return np.full(T, T, dtype=np.int64)

    big = T + 1
    dist = np.where(is_boundary, 0, big).astype(np.int64)
    for i in range(1, T):
        dist[i] = min(dist[i], dist[i - 1] + 1)
    for i in range(T - 2, -1, -1):
        dist[i] = min(dist[i], dist[i + 1] + 1)
    return dist


def relative_position(labels: np.ndarray) -> np.ndarray:
    """[T] how far into its own true segment each frame sits, in [0, 1).

    The comparable counterpart to distance_to_boundary: a frame 20 into a
    54-frame place and a frame 20 into a 767-frame idle are at wildly
    different points of their phases, and absolute distance treats them as
    the same. Every phase spans the full [0, 1) range regardless of how long
    its segments are, so bucketing on this keeps the class mixture roughly
    constant across buckets."""
    out = np.zeros(len(labels), dtype=np.float64)
    for _phase, s, e in segments(labels):
        out[s:e] = np.arange(e - s) / max(e - s, 1)
    return out


def report_bucketed_accuracy(side: str, true: np.ndarray, pred: np.ndarray,
                              dist: np.ndarray, rel: np.ndarray) -> None:
    """Two bucketings of the same frames, because the obvious one lies.

    ABSOLUTE distance to a boundary cannot be compared across phases: a phase
    whose segments average 54 frames has NO frames more than 27 from a
    boundary, so it contributes nothing at all to a '31+' bucket. Bucketing
    macro F1 that way silently changes the class mixture from bucket to
    bucket, and the apparent 'interior' score becomes a score on the long
    phases only. The support columns below make that visible instead of
    letting it masquerade as a finding.

    RELATIVE position -- how far into its own segment a frame sits, as a
    fraction -- is comparable across phases by construction, and is the one to
    read for the timing-versus-representation question.
    """
    print(f"\n[A] {side} arm -- by ABSOLUTE distance to the nearest true phase boundary")
    print("    (support columns included because short phases cannot reach the far buckets)")
    header = f"    {'distance':>10}{'frames':>9}{'acc':>8}{'macroF1':>9}   "
    print(header + "".join(Phase.NAMES[k][:5].rjust(8) for k in range(Phase.N_CLASSES)))
    for lo, hi in _BUCKETS:
        sel = (dist >= lo) & (dist <= hi)
        if not sel.any():
            continue
        m = classification_metrics(true[sel], pred[sel], Phase.N_CLASSES)
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"    {label:>10}{int(sel.sum()):>9}{m['accuracy']:>8.3f}{m['macro_f1']:>9.3f}   "
              + "".join(str(int(s)).rjust(8) for s in m["support"]))

    print(f"\n[A2] {side} arm -- by RELATIVE position within the true segment "
          "(comparable across phases)")
    print(f"    {'position':>10}{'frames':>9}{'acc':>8}{'macroF1':>9}")
    edges = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
    mids = []
    for lo, hi in edges:
        sel = (rel >= lo) & (rel < hi)
        if not sel.any():
            continue
        m = classification_metrics(true[sel], pred[sel], Phase.N_CLASSES)
        print(f"    {f'{lo:.2f}-{hi:.2f}':>10}{int(sel.sum()):>9}{m['accuracy']:>8.3f}{m['macro_f1']:>9.3f}")
        if 0.25 <= lo < 0.75:
            mids.append(m["macro_f1"])
    if mids:
        print(f"    middle-of-segment macro F1: {np.mean(mids):.3f}  "
              "<- the number to compare against the segment-level verdict in [D]")


def report_onset_latency(side: str, episodes: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Per-episode segments, pooled -- never across an episode join."""
    print(f"\n[B] {side} arm -- onset latency per true segment (frames until first correct prediction)")
    print(f"    {'phase':<12}{'segments':>10}{'missed':>9}{'median':>9}{'p90':>9}{'mean len':>10}")
    for k in range(Phase.N_CLASSES):
        lat, missed, lengths = [], 0, []
        for true, pred in episodes:
            for phase, s, e in segments(true):
                if phase != k:
                    continue
                lengths.append(e - s)
                hit = np.flatnonzero(pred[s:e] == k)
                if hit.size:
                    lat.append(int(hit[0]))
                else:
                    missed += 1
        if not lengths:
            continue
        med = f"{np.median(lat):.0f}" if lat else "--"
        p90 = f"{np.percentile(lat, 90):.0f}" if lat else "--"
        print(f"    {Phase.NAMES[k]:<12}{len(lengths):>10}{missed:>9}{med:>9}{p90:>9}"
              f"{np.mean(lengths):>10.0f}")
    print("    'missed' = the model never predicted that phase anywhere inside the true segment.")
    print("    Latency is what the nudging application is limited by: a phase found 40 frames")
    print("    (~1.3 s) late cannot shape the operator's perception in time, at any accuracy.")


def report_verdict(side: str, episodes: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """The timing-versus-representation call, made at SEGMENT level.

    The question 'can the current features tell these phases apart at all?'
    is answered by whether the correct phase is EVER predicted inside a true
    segment -- not by frame accuracy, which conflates recognising a phase
    with getting its extent right. A representation failure means the model
    cannot find the phase at all (v7 predicted PLACE on 210 frames out of
    27,000, missing most segments outright). A timing failure means it finds
    essentially every segment and then mis-places the edges.

    Latency is reported as a FRACTION of segment length for the same reason
    the relative bucketing exists: 20 frames late into a 54-frame place is a
    third of the phase gone, while the same 20 frames into a 767-frame idle
    is nothing.
    """
    print(f"\n[D] {side} arm -- segment-level verdict")
    print(f"    {'phase':<12}{'segments':>10}{'found':>8}{'rate':>8}{'lat/len':>9}")
    rates, frac_lat = [], []
    for k in range(Phase.N_CLASSES):
        n, found, fracs = 0, 0, []
        for true, pred in episodes:
            for phase, s, e in segments(true):
                if phase != k:
                    continue
                n += 1
                hit = np.flatnonzero(pred[s:e] == k)
                if hit.size:
                    found += 1
                    fracs.append(int(hit[0]) / max(e - s, 1))
        if n == 0:
            continue
        rate = found / n
        rates.append(rate)
        fl = float(np.median(fracs)) if fracs else float("nan")
        if np.isfinite(fl):
            frac_lat.append(fl)
        print(f"    {Phase.NAMES[k]:<12}{n:>10}{found:>8}{rate:>8.2f}{fl:>9.2f}")

    detection = float(np.mean(rates)) if rates else float("nan")
    print(f"    mean segment detection rate: {detection:.2f};  "
          f"median onset latency as fraction of segment: {np.mean(frac_lat):.2f}")
    if detection >= 0.85:
        print("    -> TIMING. The features DO separate these phases: the model finds almost")
        print("       every segment and then gets its extent wrong. Explicit duration modelling")
        print("       (HSMM) targets exactly this; extra representational capacity would not,")
        print("       because there is no segment it is failing to recognise.")
    elif detection < 0.7:
        print("    -> REPRESENTATION. Whole segments are going undetected, so the evidence for")
        print("       those phases is not in the current features at any duration model's")
        print("       disposal. This is where memory beyond the current frame earns its keep.")
    else:
        print("    -> MIXED. Fix duration first; it is cheaper and it will re-expose whatever")
        print("       representational gap is left underneath.")


def report_segment_inflation(side: str, episodes: list[tuple[np.ndarray, np.ndarray]]) -> None:
    n_true = sum(len(segments(t)) for t, _ in episodes)
    n_pred = sum(len(segments(p)) for _, p in episodes)
    ratio = n_pred / max(n_true, 1)
    print(f"\n[C] {side} arm -- segments: {n_true} true, {n_pred} predicted  "
          f"(inflation {ratio:.2f}x)")
    if ratio > 1.5:
        print("    The belief is chattering inside segments. A duration model or a minimum")
        print("    dwell constraint removes this without costing frame accuracy -- and it is")
        print("    invisible in the frame-level score, which is why it is worth printing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    episode_ids = read_split(args.split_file or f"data/splits/{args.split}.txt")
    model = load_model(args.model)
    model.load(args.checkpoint)

    per_side = {s: {"true": [], "pred": [], "dist": [], "rel": []} for s in SIDES}
    n = 0
    for eid in sorted(episode_ids):
        path = os.path.join(args.store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        records = score_episode(model, path)
        if records is None:
            continue
        n += 1
        for side in SIDES:
            t = np.asarray(records[side]["true_phase"])
            p = np.asarray(records[side]["pred_phase"])
            per_side[side]["true"].append(t)
            per_side[side]["pred"].append(p)
            # Distance is computed PER EPISODE and then pooled, so the join
            # between two episodes is never mistaken for a phase boundary.
            per_side[side]["dist"].append(distance_to_boundary(t))
            per_side[side]["rel"].append(relative_position(t))
    print(f"  analysed {n} labeled episode(s)")

    for side in SIDES:
        report_bucketed_accuracy(side,
                                 np.concatenate(per_side[side]["true"]),
                                 np.concatenate(per_side[side]["pred"]),
                                 np.concatenate(per_side[side]["dist"]),
                                 np.concatenate(per_side[side]["rel"]))
        # [B]-[D] are segment quantities, so they stay per-episode and are
        # pooled afterwards -- concatenating first would invent a phase
        # boundary at every episode join.
        by_episode = list(zip(per_side[side]["true"], per_side[side]["pred"]))
        report_onset_latency(side, by_episode)
        report_segment_inflation(side, by_episode)
        report_verdict(side, by_episode)


if __name__ == "__main__":
    main()

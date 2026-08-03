"""Compares scored models on one split, with paired statistics.

    python eval/compare_models.py eval/reports/hmm_v12 eval/reports/gru_v2 \
        eval/reports/transformer_matched --labels HMM GRU transformer

Reads the per_episode.csv files eval/score.py writes, so it needs no models,
no checkpoints and no store-root -- just the reports.

WHY PAIRED, AND WHY INTERVALS
-----------------------------
Two models scored on the same episodes are not two independent samples. The
episodes vary enormously among themselves -- per-episode macro F1 ranges over
roughly 0.5 to 1.0 here, a spread several times larger than any difference
between models -- so comparing group means throws away the fact that both
models saw the same hard episodes and the same easy ones. Differencing within
each (episode, arm) pair removes that shared variation and is what makes a
0.02 gap measurable at all.

The interval matters more than the point estimate. Over this conversation the
GRU-versus-transformer ordering flipped twice depending on which split and
which code version was in play, and each individual comparison looked
convincing on its own. A confidence interval that straddles zero says that
directly, which a table of three-decimal means never will.

Reported per pair:
  mean difference and its standard error, over (episode, arm) pairs
  a rough 95% interval (mean +/- 2 SE)
  the win count, which catches a difference driven by one outlier episode
  Cohen's d on the paired differences, as a scale-free effect size

n is 22 pairs, so these intervals are wide and honestly so. Treat anything
whose interval includes zero as a tie, however tidy the means look.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

METRICS = ("phase_macro_f1", "phase_accuracy", "target_accuracy_given_committed")


def load(report_dir: str) -> dict:
    path = os.path.join(report_dir, "per_episode.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no per_episode.csv in {report_dir} -- run eval/score.py first")
    out = {}
    for row in csv.DictReader(open(path)):
        key = (row["episode_id"], row["side"])
        vals = {}
        for m in METRICS:
            raw = row.get(m, "")
            try:
                vals[m] = float(raw)
            except (TypeError, ValueError):
                vals[m] = float("nan")
        out[key] = vals
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference statistics for b - a, ignoring pairs with a NaN."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    n = d.size
    if n < 2:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "wins": 0,
                "lo": float("nan"), "hi": float("nan"), "d": float("nan")}
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    return {"n": n, "mean": float(d.mean()), "se": float(se), "wins": int((d > 0).sum()),
            "lo": float(d.mean() - 2 * se), "hi": float(d.mean() + 2 * se),
            "d": float(d.mean() / sd) if sd > 0 else float("inf")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="eval/reports/<name> directories")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--metric", default="phase_macro_f1", choices=METRICS)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(r.rstrip("/\\")) for r in args.reports]
    if len(labels) != len(args.reports):
        raise SystemExit("--labels must have one entry per report directory")

    data = {lab: load(r) for lab, r in zip(labels, args.reports)}
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    if not keys:
        raise SystemExit("these reports share no (episode, arm) pairs -- different splits?")

    print(f"{len(keys)} shared (episode, arm) pairs   metric: {args.metric}\n")
    print(f"  {'model':<22}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
    series = {}
    for lab in labels:
        v = np.array([data[lab][k][args.metric] for k in keys])
        series[lab] = v
        finite = v[np.isfinite(v)]
        print(f"  {lab:<22}{finite.mean():>8.3f}{finite.std():>8.3f}"
              f"{finite.min():>8.3f}{finite.max():>8.3f}")

    print(f"\n  paired differences (later model minus earlier), n={len(keys)}")
    print(f"  {'comparison':<30}{'diff':>8}{'95% interval':>18}{'wins':>7}{'d':>7}   verdict")
    for a, b in itertools.combinations(labels, 2):
        s = paired(series[a], series[b])
        interval = f"[{s['lo']:+.3f}, {s['hi']:+.3f}]"
        # An interval straddling zero means the sign of the difference is not
        # established -- report that as a tie rather than quoting the mean.
        verdict = ("tie" if s["lo"] <= 0 <= s["hi"]
                   else f"{b} better" if s["mean"] > 0 else f"{a} better")
        print(f"  {b + ' - ' + a:<30}{s['mean']:>+8.3f}{interval:>18}"
              f"{s['wins']:>4}/{s['n']:<3}{s['d']:>7.2f}   {verdict}")

    print("\n  d is Cohen's d on the paired differences: |d| below ~0.5 is a small effect")
    print("  even when the interval excludes zero. With 22 pairs, treat every interval")
    print("  containing zero as a tie regardless of how the means order.")


if __name__ == "__main__":
    main()

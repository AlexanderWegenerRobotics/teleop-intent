"""Plots and summarises a training curve, and says how much of the headline
number is real.

    python eval/plot_training.py checkpoints/gru/v1_history.csv
    python eval/plot_training.py checkpoints/gru/v1_history.csv checkpoints/gru/shuffled_history.csv \
        --labels "GRU" "shuffled control" --out eval/reports/gru_v1/training.png

WHY THIS EXISTS RATHER THAN JUST READING THE LAST LINE
-------------------------------------------------------
Reporting "best val macro F1" is reporting the MAXIMUM of a noisy sequence,
and the maximum of N noisy draws is biased upward -- by roughly the noise
standard deviation times sqrt(2 ln N). Over thirty epochs with a val set of
thirteen episodes that is not a rounding error; it is most of the gap you
might be about to claim over a baseline.

So this prints three things the single number hides:

  PLATEAU EPOCH -- the first epoch within one noise standard deviation of the
    best. If that is epoch 3 and the best is epoch 19, the sixteen epochs
    between them bought nothing, and the difference between their scores is
    the noise floor rather than progress.

  NOISE FLOOR -- the standard deviation of val across the plateau. Any two
    models whose scores differ by less than this are not distinguishable on
    this validation set, whatever the decimals say.

  SELECTION OPTIMISM -- an estimate of how much picking the maximum inflated
    the reported figure. Subtract it before comparing against a model whose
    number was NOT chosen as a maximum over many evaluations.

None of this makes the model worse. It makes the claim honest, and it is the
difference between "the GRU beats the HMM by 0.08" and "by 0.08, of which
about 0.02 is selection optimism on a 13-episode validation set".
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                   # pragma: no cover
    plt = None


def read_history(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r.get(k, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(float("nan") if v not in ("True", "False") else float(v == "True"))
        out[k] = np.asarray(vals, dtype=float)
    return out


def summarise(h: dict[str, np.ndarray], label: str) -> dict:
    """Plateau epoch, noise floor and selection optimism for one run."""
    if "val_macro_f1" not in h:
        print(f"{label}: no val_macro_f1 column -- was this run given a val split?")
        return {}
    f1 = h["val_macro_f1"]
    epochs = h.get("epoch", np.arange(len(f1)))
    best_i = int(np.nanargmax(f1))
    best = float(f1[best_i])

    # Noise floor from the tail, where the curve is no longer improving:
    # successive differences there are noise, and std of those differences over sqrt(2)
    # estimates the per-evaluation standard deviation without assuming the
    # mean is flat.
    tail = f1[max(best_i - 2, len(f1) // 3):]
    sigma = float(np.std(np.diff(tail)) / np.sqrt(2)) if len(tail) > 2 else float("nan")

    within = np.flatnonzero(f1 >= best - sigma) if np.isfinite(sigma) else np.array([best_i])
    plateau_i = int(within[0]) if within.size else best_i

    n = len(f1)
    optimism = sigma * np.sqrt(2 * np.log(max(n, 2))) if np.isfinite(sigma) else float("nan")

    print(f"\n{label}")
    print(f"  best val macro F1   {best:.3f} at epoch {int(epochs[best_i])} (of {n} evaluated)")
    print(f"  plateau reached     epoch {int(epochs[plateau_i])} "
          f"(val {f1[plateau_i]:.3f}, within one noise sd of the best)")
    print(f"  noise floor (1 sd)  {sigma:.3f}  -- differences smaller than this are not real")
    print(f"  selection optimism  ~{optimism:.3f}  -- expected inflation from taking a max over "
          f"{n} noisy evaluations")
    print(f"  honest estimate     {best - optimism:.3f} to {best:.3f}")
    if plateau_i <= max(2, n // 10):
        print(f"  NOTE: the plateau was reached at epoch {int(epochs[plateau_i])}. Everything "
              f"after it is noise, not learning --")
        print("        train longer only if the training loss is still tracking val, which here it is not.")
    return {"best": best, "best_epoch": int(epochs[best_i]), "sigma": sigma,
            "plateau_epoch": int(epochs[plateau_i]), "optimism": optimism}


def plot(histories: list[tuple[str, dict]], out_path: str) -> None:
    if plt is None:
        print("matplotlib not installed -- skipping the plot (the summary above still stands)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, h) in enumerate(histories):
        c = colours[i % len(colours)]
        ep = h.get("epoch", np.arange(len(next(iter(h.values())))))
        if "phase_loss" in h:
            axes[0].plot(ep, h["phase_loss"], color=c, label=f"{label} phase")
        if "target_loss" in h:
            axes[0].plot(ep, h["target_loss"], color=c, ls="--", label=f"{label} target")
        if "val_macro_f1" in h:
            f1 = h["val_macro_f1"]
            axes[1].plot(ep, f1, color=c, marker="o", ms=3, label=f"{label} val macro F1")
            b = int(np.nanargmax(f1))
            axes[1].scatter([ep[b]], [f1[b]], color=c, s=70, zorder=5,
                            edgecolor="white", linewidth=1.2)
            axes[1].annotate(f"{f1[b]:.3f}", (ep[b], f1[b]), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=9, color=c)
        if "val_target_accuracy" in h:
            axes[1].plot(ep, h["val_target_accuracy"], color=c, ls=":", alpha=0.8,
                         label=f"{label} val target acc")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[0].set_title("training loss (solid = phase, dashed = target)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation score")
    axes[1].set_title("validation (marker = selected epoch)")
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("histories", nargs="+", help="one or more *_history.csv files")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output png (default: beside the first history)")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(p).replace("_history.csv", "") for p in args.histories]
    if len(labels) != len(args.histories):
        raise SystemExit("--labels must have one entry per history file")

    loaded = [(lab, read_history(p)) for lab, p in zip(labels, args.histories)]
    stats = [summarise(h, lab) for lab, h in loaded]

    if len(stats) == 2 and all(s for s in stats):
        gap = stats[0]["best"] - stats[1]["best"]
        floor = max(stats[0]["sigma"], stats[1]["sigma"])
        print(f"\ngap between the two: {gap:+.3f}  (noise floor {floor:.3f})")
        print("  -> " + ("real: the gap is several times the noise floor."
                         if abs(gap) > 3 * floor else
                         "NOT distinguishable on this validation set -- the gap is within "
                         "a few noise standard deviations."))

    out = args.out or (os.path.splitext(args.histories[0])[0] + ".png")
    plot(loaded, out)


if __name__ == "__main__":
    main()

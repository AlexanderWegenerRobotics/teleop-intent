"""Regenerates every results figure and the results table from the reports.

    python eval/make_figures.py

Reads only per_episode.csv files and *_history.csv files -- no checkpoints, no
episodes, no store-root. Anyone who clones the repo can reproduce every number
and every figure in the README without the dataset.

Writes to docs/, which IS version-controlled, unlike eval/reports/. The reports
are bulky per-run intermediates that nobody cloning this repo needs; the handful
of figures the README actually cites do need to travel with it, or the README
renders as a page of broken images on GitHub.

    docs/figures/01_ablation_ladder.png    what each ingredient is worth
    docs/figures/02_model_comparison.png   the three models with paired intervals
    docs/figures/03_per_phase.png          where the gains actually land
    docs/results.md                        the same numbers as a markdown table

The figures are deliberately plain: no gridlines competing with the data, no
3D, no colour carrying information that is also in a label. They are meant to
be readable at portfolio-thumbnail size and to survive being printed in
greyscale.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHASES = ["idle", "approach", "grasp", "transport", "place"]

INK, MUTED, ACCENT, GREY = "#1f2a30", "#5f7079", "#1c7293", "#b8c4c9"

# Test-split reports, in the order they should appear.
MODELS = [("HMM", "hmm_v12"), ("GRU", "gru_v2"), ("Transformer", "transformer_matched")]

# Validation-split ablations, read from training histories. The controls live
# here rather than in the test reports because they are not deployable models --
# they exist only to bound what each ingredient contributes.
ABLATIONS = [
    ("shuffled features", "checkpoints/gru/shuffled_history.csv", "chance floor"),
    ("no temporal context", "checkpoints/gru/memoryless_v2_history.csv", "features only"),
    ("GRU", "checkpoints/gru/v2_history.csv", "features + memory"),
    ("Transformer", "checkpoints/transformer/matched_history.csv", "features + attention"),
]


def read_per_episode(report_dir: str) -> dict:
    path = os.path.join(report_dir, "per_episode.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path)):
        key = (row["episode_id"], row["side"])
        rec = {}
        for k, v in row.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = v
        out[key] = rec
    return out


def read_best_val(history_csv: str) -> float:
    if not os.path.exists(history_csv):
        return float("nan")
    vals = []
    for row in csv.DictReader(open(history_csv)):
        try:
            vals.append(float(row["val_macro_f1"]))
        except (TypeError, ValueError, KeyError):
            pass
    return max(vals) if vals else float("nan")


def confusion_from_reports(report_dir: str) -> np.ndarray | None:
    """Per-phase F1 is recomputed from the confusion matrix if one was saved;
    otherwise None. Kept optional so the script never fails on a partial run."""
    path = os.path.join(report_dir, "confusion.json")
    if not os.path.exists(path):
        return None
    return np.array(json.load(open(path)))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    sd = d.std(ddof=1)
    se = sd / np.sqrt(d.size)
    return {"mean": float(d.mean()), "se": float(se), "n": int(d.size),
            "lo": float(d.mean() - 2 * se), "hi": float(d.mean() + 2 * se),
            "wins": int((d > 0).sum())}


def fig_ablation(out_path: str, rows: list[tuple[str, str, float]]) -> None:
    """Horizontal bars: what each ingredient contributes.

    Ordered bottom-up so the reader walks from 'knows nothing' to the full
    model, which is the order the argument is made in. The chance floor is
    shown rather than implied -- on this dataset a model that has learned
    nothing still reaches 60% frame ACCURACY, so a reader who assumes zero is
    the floor will badly misread every other bar.
    """
    labels = [f"{name}\n{note}" for name, note, _ in rows]
    vals = [v for _, _, v in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))
    colours = [GREY if i < 2 else ACCENT for i in range(len(rows))]
    ax.barh(y, vals, color=colours, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("validation macro F1", fontsize=10)
    ax.set_title("What each ingredient is worth", fontsize=12, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_models(out_path: str, names: list[str], means: list[float],
               errs: list[float], pairs: list[tuple[str, dict]]) -> None:
    """Test-split means with per-episode spread, plus the paired differences.

    Two panels because the two questions are different. The left says how well
    each model does; the right says whether any pair is actually
    distinguishable, which the left panel cannot show -- episodes vary among
    themselves far more than models do.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=errs, color=ACCENT, width=0.55, capsize=5,
           error_kw={"ecolor": MUTED, "elinewidth": 1.2})
    for i, v in enumerate(means):
        ax.text(i, v + errs[i] + 0.015, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test macro F1", fontsize=10)
    ax.set_title("Held-out performance", fontsize=12, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)

    ax = axes[1]
    y = np.arange(len(pairs))
    for i, (_, s) in enumerate(pairs):
        colour = ACCENT if not (s["lo"] <= 0 <= s["hi"]) else GREY
        ax.plot([s["lo"], s["hi"]], [i, i], color=colour, lw=3, solid_capstyle="round")
        ax.plot([s["mean"]], [i], "o", color=colour, ms=7,
                markeredgecolor="white", markeredgewidth=1.2)
    ax.axvline(0, color=INK, lw=1, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs], fontsize=9.5)
    ax.set_xlabel("paired difference in macro F1 (95% interval)", fontsize=10)
    ax.set_title("Is the difference real?", fontsize=12, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.text(0.02, -0.7, "intervals crossing zero are ties", fontsize=8.5,
            color=MUTED, transform=ax.get_yaxis_transform())

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_per_phase(out_path: str, names: list[str], per_phase: dict) -> None:
    """Grouped bars per phase. The gains are not spread evenly, and the
    aggregate hides that: idle is near-solved for every model while place and
    grasp carry almost all of the difference."""
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    n = len(names)
    width = 0.8 / n
    x = np.arange(len(PHASES))
    shades = [GREY, "#7fa8b8", ACCENT][:n] if n <= 3 else None
    for i, name in enumerate(names):
        vals = [per_phase[name].get(p, np.nan) for p in PHASES]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width=width * 0.92,
               label=name, color=(shades[i] if shades else None))
    ax.set_xticks(x)
    ax.set_xticklabels(PHASES, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test F1", fontsize=10)
    ax.set_title("Per-phase F1: where the difference lands", fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9.5, ncol=n)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="eval/reports")
    # docs/, not eval/reports/figures/: eval/reports/ is gitignored, and a
    # README whose figures live in an ignored directory renders as broken
    # images for everyone but the person who generated them.
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--results-md", default="docs/results.md")
    ap.add_argument("--per-phase", default=None,
                    help="optional JSON: {model: {phase: f1}} for figure 3, since "
                         "per_episode.csv stores macro scores rather than per-phase ones")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- ablation ladder (validation) ------------------------------------
    ladder = []
    for name, hist, note in ABLATIONS:
        v = read_best_val(hist)
        if np.isfinite(v):
            ladder.append((name, note, v))
    if ladder:
        fig_ablation(os.path.join(args.out_dir, "01_ablation_ladder.png"), ladder)
        print(f"  01_ablation_ladder.png  ({len(ladder)} bars)")

    # ---- model comparison (test, paired) ---------------------------------
    data, names = {}, []
    for label, folder in MODELS:
        d = read_per_episode(os.path.join(args.reports, folder))
        if d:
            data[label] = d
            names.append(label)
    if len(names) < 2:
        raise SystemExit("need at least two scored models in eval/reports")

    keys = sorted(set.intersection(*[set(data[n]) for n in names]))
    series = {n: np.array([data[n][k]["phase_macro_f1"] for k in keys]) for n in names}
    means = [float(np.nanmean(series[n])) for n in names]
    # Standard error of the mean over episodes, not the standard deviation:
    # the bar is an estimate of the model's average, and its uncertainty is
    # what a reader needs to judge the gap.
    errs = [float(np.nanstd(series[n], ddof=1) / np.sqrt(len(keys))) for n in names]

    pairs = []
    for a, b in itertools.combinations(names, 2):
        pairs.append((f"{b} - {a}", paired(series[a], series[b])))
    fig_models(os.path.join(args.out_dir, "02_model_comparison.png"), names, means, errs, pairs)
    print(f"  02_model_comparison.png  ({len(keys)} episode-arm pairs)")

    # ---- per-phase -------------------------------------------------------
    if args.per_phase and os.path.exists(args.per_phase):
        per_phase = json.load(open(args.per_phase))
        present = [n for n in names if n in per_phase]
        if present:
            fig_per_phase(os.path.join(args.out_dir, "03_per_phase.png"), present, per_phase)
            print("  03_per_phase.png")
    else:
        print("  03_per_phase.png skipped -- pass --per-phase with a JSON of "
              "{model: {phase: f1}} taken from eval/score.py's output")

    # ---- results.md ------------------------------------------------------
    lines = ["# Results", "",
             f"Held-out test split: {len(keys)} (episode, arm) pairs.", "",
             "| model | test macro F1 | SE |", "|---|---|---|"]
    for n, m, e in zip(names, means, errs):
        lines.append(f"| {n} | {m:.3f} | {e:.3f} |")
    lines += ["", "| comparison | difference | 95% interval | verdict |", "|---|---|---|---|"]
    for name, s in pairs:
        tie = s["lo"] <= 0 <= s["hi"]
        lines.append(f"| {name} | {s['mean']:+.3f} | [{s['lo']:+.3f}, {s['hi']:+.3f}] | "
                     f"{'tie' if tie else 'distinguishable'} |")
    if ladder:
        lines += ["", "| ablation | validation macro F1 |", "|---|---|"]
        for name, note, v in ladder:
            lines.append(f"| {name} ({note}) | {v:.3f} |")
    os.makedirs(os.path.dirname(args.results_md) or ".", exist_ok=True)
    open(args.results_md, "w").write("\n".join(lines) + "\n")
    print(f"  {args.results_md}")
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()

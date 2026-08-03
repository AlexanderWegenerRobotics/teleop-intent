"""Measures per-frame inference latency for each model against the control-loop
budget.

    python eval/benchmark_latency.py --store-root /path/to/episodes --episode 008 \
        HMM=models.hmm.model.HMMIntentModel:checkpoints/hmm/v12.npz \
        GRU=models.gru.model.GRUIntentModel:checkpoints/gru/v2.pt \
        Transformer=models.transformer.model.TransformerIntentModel:checkpoints/transformer/matched.pt

WHY PERCENTILES, NOT THE MEAN
-----------------------------
The module runs inside a 30 Hz loop, so it has 33 ms per frame and a frame that
takes longer is simply late -- there is no averaging away a miss. A model whose
MEAN is 20 ms but whose p99 is 60 ms drops roughly one frame in a hundred, and
during a grasp that is exactly the frame that mattered. Mean latency is the
number that flatters; p95 and p99 are the ones that decide deployability.

WHY IT IS TIMED OVER REAL FRAMES
--------------------------------
step() is stateful. The transformer's cost grows as its context buffer fills,
so timing it on the first fifty frames of an episode understates it by a large
factor; the GRU's cost is flat by construction and would look identical either
way. Timing runs over a real episode, after a warm-up long enough to fill the
largest buffer, so every model is measured in the regime it actually runs in.

One step() call covers BOTH arms, which is the unit the control loop cares
about. No figure is reported per arm.
"""

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from teleop_orchestrator.sources import ReplaySource  # noqa: E402

BUDGET_MS = 1000.0 / 30.0        # 30 Hz control loop


def load_model(spec: str, checkpoint: str):
    module_path, class_name = spec.rsplit(".", 1)
    model = getattr(importlib.import_module(module_path), class_name)()
    model.load(checkpoint)
    return model


def benchmark(model, frames, warmup: int) -> dict:
    """Per-frame wall-clock for step(), after `warmup` untimed frames.

    perf_counter around a single step() rather than total time over N frames:
    the distribution is the point, and a total would hide the tail entirely.
    """
    model.reset()
    for f in frames[:warmup]:
        model.step(f)
    times = []
    for f in frames[warmup:]:
        t0 = time.perf_counter()
        model.step(f)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {
        "n": int(t.size),
        "mean": float(t.mean()),
        "p50": float(np.percentile(t, 50)),
        "p95": float(np.percentile(t, 95)),
        "p99": float(np.percentile(t, 99)),
        "max": float(t.max()),
        "over_budget": float((t > BUDGET_MS).mean() * 100.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="LABEL=module.Class:checkpoint")
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--episode", default=None,
                    help="episode id; defaults to the first in the test split")
    ap.add_argument("--frames", type=int, default=800, help="timed frames per model")
    ap.add_argument("--warmup", type=int, default=1100,
                    help="untimed frames first, enough to fill the largest context "
                         "buffer (the transformer's is 1023 frames)")
    ap.add_argument("--out", default="docs/figures/04_latency.png")
    args = ap.parse_args()

    if args.episode is None:
        split = "data/splits/test.txt"
        args.episode = open(split).readline().strip() if os.path.exists(split) else "008"

    path = os.path.join(args.store_root, args.episode, "episode.hdf5")
    if not os.path.exists(path):
        raise SystemExit(f"no episode at {path}")

    src = ReplaySource(path, load_images=False)
    need = args.warmup + args.frames
    if len(src) < need:
        print(f"  episode {args.episode} has {len(src)} frames, need {need}; "
              f"reducing warmup and timed frames proportionally")
        args.warmup = int(len(src) * args.warmup / need)
        args.frames = len(src) - args.warmup
    frames = [src.frame_at(t) for t in range(args.warmup + args.frames)]
    src.close()
    print(f"episode {args.episode}: {args.warmup} warm-up + {args.frames} timed frames\n")

    results = {}
    for spec in args.models:
        label, rest = spec.split("=", 1)
        cls, checkpoint = rest.rsplit(":", 1)
        model = load_model(cls, checkpoint)
        results[label] = benchmark(model, frames, args.warmup)
        print(f"  {label} done")

    print(f"\nper-frame step() latency, both arms, budget {BUDGET_MS:.1f} ms at 30 Hz\n")
    print(f"  {'model':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
          f"{'over budget':>14}{'headroom':>11}")
    for label, r in results.items():
        head = BUDGET_MS / r["p99"]
        verdict = f"{head:.1f}x" if head >= 1 else f"{1 / head:.1f}x OVER"
        print(f"  {label:<14}{r['mean']:>8.1f}m{r['p50']:>8.1f}m{r['p95']:>8.1f}m"
              f"{r['p99']:>8.1f}m{r['max']:>8.1f}m{r['over_budget']:>13.1f}%{verdict:>11}")
    print("\n  headroom is budget / p99: above 1x fits the loop, below 1x misses it")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = list(results)
    p50 = [results[k]["p50"] for k in labels]
    p99 = [results[k]["p99"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, p50, width=0.34, label="median", color="#b8c4c9")
    ax.bar(x + 0.18, p99, width=0.34, label="p99", color="#1c7293")
    ax.axhline(BUDGET_MS, color="#c0392b", lw=1.6, ls="--")
    ax.text(len(labels) - 0.45, BUDGET_MS * 1.08, "30 Hz budget", color="#c0392b", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("per-frame latency (ms)", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("Inference cost against the control-loop budget",
                 fontsize=12, color="#1f2a30", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

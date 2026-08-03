"""How wrong is the HMM's geometric duration assumption, per phase?

This is the gate on whether duration modelling is worth building at all.
Answer it from the labels before writing an HSMM, not after.

THE ASSUMPTION. In a first-order HMM, how long a phase lasts is not modelled
directly -- it falls out of the self-transition probability rho, and is
therefore geometric:

    P(duration = d) = rho^(d-1) * (1 - rho),  mean = 1/(1-rho),  CV = sqrt(rho)

Two consequences matter. The mean is free to be correct (the fit will match
it), but the SHAPE is fixed: a geometric's most likely duration is always 1
frame, and its coefficient of variation is always ~1, meaning its standard
deviation is as large as its mean. The model cannot believe "this phase
reliably lasts about 1.8 seconds" no matter how the parameters are set.

THE TEST. Compare the observed CV of each phase's segment durations against
the ~1.0 a geometric forces:

  * observed CV near 1.0  -> the geometric is a fine description of that
    phase; duration modelling will buy nothing there. Expect this for IDLE,
    whose length genuinely varies with what the operator is doing.
  * observed CV well below 1.0 -> durations are far more regular than the
    model can express, and the model is spending probability mass on
    durations that never occur. Expect this for the manipulation phases.

WHAT TO BUILD IF IT FAILS. Not necessarily an HSMM. Chaining N sub-states in
series per phase turns the duration into a negative binomial with
CV = sqrt(rho'/N) ~ 1/sqrt(N), so three sub-states reach CV 0.58 and five
reach 0.45 -- while remaining an ordinary HMM with the same forward
algorithm, just a larger transition matrix. This script prints the N implied
by each phase's observed CV. Only reach for a full HSMM if the required N is
large or the observed durations are too irregular for a negative binomial to
follow.

Usage:

    python scripts/duration_stats.py --store-root /path/to/avatar --split train
    python scripts/duration_stats.py --store-root /path/to/avatar --split train \
        --checkpoint checkpoints/hmm/v9.npz --out-dir eval/reports/durations
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from teleop_orchestrator.contracts import Phase  # noqa: E402

import labels as label_loader  # noqa: E402
from metrics import segment_durations  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SIDES = ("left", "right")

# Cap on chained sub-states per phase before the transition matrix gets
# unwieldy: N sub-states across 5 phases means a 5N x 5N matrix estimated
# from the same handful of labeled segments. Past this point an explicit
# duration distribution (HSMM) has FEWER parameters than faking it with
# structure, which is exactly when committing to the HSMM is the right call.
_MAX_SUBSTATES = 12


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def gather_durations(store_root: str, episode_ids: list[str]) -> dict[int, list[int]]:
    by_phase: dict[int, list[int]] = {k: [] for k in range(Phase.N_CLASSES)}
    n_ep = 0
    for eid in sorted(episode_ids):
        path = os.path.join(store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as ep:
            if not label_loader.is_genuinely_labeled(ep):
                continue
            n_ep += 1
            for side in SIDES:
                phase, _target = label_loader.load_arm_labels(ep, side)
                for k, d in segment_durations(phase):
                    by_phase[k].append(d)
    print(f"  {n_ep} labeled episode(s), both arms pooled, censored end segments dropped")
    return by_phase


def geometric_stats(rho: float) -> tuple[float, float, float]:
    """(mean, std, CV) of the duration a self-transition rho implies."""
    q = 1.0 - rho
    if q <= 0:
        return float("inf"), float("inf"), 1.0
    return 1.0 / q, np.sqrt(rho) / q, float(np.sqrt(rho))


def report(by_phase: dict[int, list[int]], transition: np.ndarray | None) -> dict[int, int]:
    print("\nOBSERVED segment durations (frames, ~30 Hz)")
    print(f"  {'phase':<12}{'n':>6}{'mean':>8}{'std':>8}{'CV':>7}{'p10':>7}{'median':>8}{'p90':>7}")
    obs = {}
    for k in range(Phase.N_CLASSES):
        d = np.array(by_phase[k], dtype=float)
        if d.size < 3:
            print(f"  {Phase.NAMES[k]:<12}{d.size:>6}   too few segments to characterise")
            continue
        cv = float(d.std() / d.mean()) if d.mean() > 0 else float("nan")
        obs[k] = (d.mean(), d.std(), cv)
        print(f"  {Phase.NAMES[k]:<12}{d.size:>6}{d.mean():>8.0f}{d.std():>8.0f}{cv:>7.2f}"
              f"{np.percentile(d, 10):>7.0f}{np.median(d):>8.0f}{np.percentile(d, 90):>7.0f}")

    if transition is not None:
        print("\nWHAT THE FITTED SELF-TRANSITIONS IMPLY (geometric)")
        print(f"  {'phase':<12}{'rho':>8}{'mean':>8}{'std':>8}{'CV':>7}   {'vs observed'}")
        for k in range(Phase.N_CLASSES):
            if k not in obs:
                continue
            rho = float(transition[k, k])
            gm, gs, gcv = geometric_stats(rho)
            om, os_, ocv = obs[k]
            print(f"  {Phase.NAMES[k]:<12}{rho:>8.4f}{gm:>8.0f}{gs:>8.0f}{gcv:>7.2f}   "
                  f"mean off by {gm - om:+.0f} frames; spread {gs / max(os_, 1e-9):.1f}x too wide")

    print("\nVERDICT -- is the geometric assumption costing anything?")
    print(f"  {'phase':<12}{'obs CV':>8}{'geo CV':>8}{'sub-states needed':>20}{'resulting CV':>14}")
    recommended = {}
    for k in range(Phase.N_CLASSES):
        if k not in obs:
            continue
        _m, _s, cv = obs[k]
        n_sub = max(1, int(round(1.0 / max(cv, 1e-6) ** 2)))
        n_sub = min(n_sub, _MAX_SUBSTATES)
        note = "" if n_sub > 1 else "  (geometric already adequate)"
        recommended[k] = n_sub
        print(f"  {Phase.NAMES[k]:<12}{cv:>8.2f}{1.0:>8.2f}{n_sub:>20}{1.0 / np.sqrt(n_sub):>14.2f}{note}")

    worst = [k for k, n in recommended.items() if n >= 3]
    capped = [k for k, n in recommended.items() if n >= _MAX_SUBSTATES]
    if not worst:
        print("\n  -> Durations are about as irregular as a geometric already assumes.")
        print("     Duration modelling is NOT the bottleneck; do not build an HSMM on this")
        print("     evidence -- look at features or at the labels instead.")
    elif capped:
        names = ", ".join(Phase.NAMES[k] for k in capped)
        print(f"\n  -> Duration is badly mismodelled, and severely so for: {names}.")
        print(f"     Their observed CV would need MORE than {_MAX_SUBSTATES} chained sub-states"
              " each to reproduce,")
        print("     which means a 60+ state transition matrix and a lot of parameters estimated")
        print("     from few segments. This is the case where a full HSMM -- an explicit,")
        print("     directly-fitted duration distribution per phase -- is the cheaper model,")
        print("     not the more expensive one. Committing to it is justified.")
    else:
        names = ", ".join(Phase.NAMES[k] for k in worst)
        print(f"\n  -> Duration IS mismodelled for: {names}, but within reach of sub-states.")
        print("     A chained sub-state HMM addresses this with the existing forward algorithm")
        print("     and no new inference code; build that first and only move to a full HSMM")
        print("     if it falls short.")
    return recommended


def save_plots(by_phase: dict[int, list[int]], transition: np.ndarray | None, out_dir: str) -> None:
    if plt is None:
        print("  matplotlib not installed -- skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    for k in range(Phase.N_CLASSES):
        d = np.array(by_phase[k], dtype=float)
        if d.size < 3:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(d, bins=min(30, max(5, d.size // 3)), density=True, alpha=0.65,
                label="observed", color="tab:blue")
        if transition is not None:
            rho = float(transition[k, k])
            x = np.arange(1, int(max(d.max(), 2)) + 1)
            ax.plot(x, rho ** (x - 1) * (1 - rho), color="tab:red", lw=2,
                    label=f"geometric (rho={rho:.4f})")
        ax.set_xlabel("segment duration (frames)")
        ax.set_ylabel("density")
        ax.set_title(f"{Phase.NAMES[k]} duration: observed vs HMM assumption")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"duration_{Phase.NAMES[k]}.png"), dpi=150)
        plt.close(fig)
    print(f"  saved duration plots to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train",
                    help="default train: the same episodes the transition matrix was fit on, "
                         "which is the fair comparison for 'does the fitted rho describe them'")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="optional: compare against the self-transitions actually fitted")
    ap.add_argument("--out-dir", default=None, help="optional: write per-phase histogram plots here")
    args = ap.parse_args()

    transition = None
    if args.checkpoint:
        transition = np.load(args.checkpoint)["transition"]

    episode_ids = read_split(args.split_file or f"data/splits/{args.split}.txt")
    print(f"Reading durations from the {args.split} split ({len(episode_ids)} episode ids listed)...")
    by_phase = gather_durations(args.store_root, episode_ids)

    if not any(by_phase.values()):
        raise SystemExit("No labeled episodes found -- check --store-root and the split file")

    report(by_phase, transition)
    if args.out_dir:
        save_plots(by_phase, transition, args.out_dir)


if __name__ == "__main__":
    main()

"""Is the phase emission model biased, and can it recognise its own classes?

Reads a checkpoint directly (no episodes, no store-root, no
teleop_orchestrator ReplaySource) and answers four questions that together
diagnose the failure mode described in models/hmm/phase.py. Run it after
every fit, and on any checkpoint whose confusion matrix looks strange.

    python scripts/diagnose_emissions.py checkpoints/hmm/v8.npz
    python scripts/diagnose_emissions.py checkpoints/hmm/v7.npz checkpoints/hmm/v8.npz

1. NORMALIZER BIAS. The part of each class's emission log-likelihood that
   does not depend on the observation at all. On v7 this was 24.31 nats for
   IDLE against 18.93 for GRASP -- a constant 220:1 head start for IDLE on
   every single frame. Anything above ~1-2 nats of spread here deserves
   suspicion; it is a prior masquerading as evidence.

2. SEPARABILITY. Per-feature mean gap between phase pairs in units of pooled
   std. On v7 the largest IDLE-vs-GRASP gap across all nine features was 0.75
   -- far too small for the Mahalanobis term to overcome item 1.

3. TRANSITION VS EMISSION. What the transition prior is actually worth in the
   forward recursion. transition[APPROACH, IDLE] = 0.0001 suggests a 9-nat
   wall, but IDLE's residual belief never decays (self-loop 0.9989), so the
   real margin from a confident APPROACH belief was 0.87 nats. Compare that
   number against item 1: if the bias exceeds the margin, the filter will
   fall into the biased class regardless of what the data says.

4. SELF-RECOGNITION. The decisive test. Feed the model the exact class mean
   of one phase, from a belief fully committed to the preceding phase, and
   see whether the filter tracks it. A model that cannot recognise its own
   training means on noise-free input has an emission problem, not a data
   problem. v7 fed the APPROACH mean collapsed to IDLE within ~30 frames and
   then stayed pinned at IDLE through 30 frames of the exact GRASP mean.
"""

from __future__ import annotations

import argparse
import os
import sys

# Running this as `python scripts/diagnose_emissions.py` puts scripts/ on
# sys.path, not the repo root, so `models` and `metrics` are invisible. Add
# the root explicitly rather than requiring a -m invocation or a PYTHONPATH,
# since the whole point of this script is to be quick to reach for.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from teleop_orchestrator.contracts import Phase  # noqa: E402

from models.hmm.phase import PhaseHMMParams, filter_episode, emission_loglik  # noqa: E402
from models.hmm.features import FEATURE_NAMES  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def load_params(path: str) -> PhaseHMMParams:
    """Reads phase parameters straight out of the npz.

    Deliberately does NOT go through HMMIntentModel.load, which refuses a
    checkpoint whose feature count no longer matches features.py -- correct
    for scoring, wrong for a diagnostic whose whole job is to compare old and
    new checkpoints side by side.
    """
    z = np.load(path)
    names = ([str(s) for s in z["feature_names"]] if "feature_names" in z
             else list(FEATURE_NAMES)[:z["emission_mean"].shape[1]])
    return PhaseHMMParams(
        transition=z["transition"], prior=z["prior"],
        emission_mean=z["emission_mean"], emission_std=z["emission_std"],
        bernoulli_p=z["bernoulli_p"] if "bernoulli_p" in z else None,
        is_binary=z["is_binary"].astype(bool) if "is_binary" in z else None,
        emission_temp=float(z["emission_temp"]) if "emission_temp" in z else 1.0,
        tie_covariance=bool(z["tie_covariance"]) if "tie_covariance" in z else False,
        feature_names=names,
    )


def report_normalizer_bias(p: PhaseHMMParams) -> None:
    names, K = p.feature_names, p.emission_mean.shape[0]
    binary = p.binary_mask
    print("\n[1] EMISSION NORMALIZER BIAS -- constant per-class log-likelihood, no observation involved")
    print("    (Gaussian dims contribute -log(sigma); Bernoulli dims contribute nothing constant)")
    per_feat = np.where(binary[None, :], 0.0, -np.log(p.emission_std))
    print("    " + "feature".ljust(24) + "".join(Phase.NAMES[k].rjust(11) for k in range(K)))
    for j, nm in enumerate(names):
        tag = " (bern)" if binary[j] else ""
        print("    " + (nm + tag).ljust(24) + "".join(f"{per_feat[k, j]:11.2f}" for k in range(K)))
    total = p.normalizer_bias()
    print("    " + f"TOTAL x temp({p.emission_temp:.2f})".ljust(24) + "".join(f"{t:11.2f}" for t in total))
    spread = float(np.ptp(total))
    verdict = "OK" if spread < 2.0 else ("MARGINAL" if spread < 4.0 else "BROKEN")
    print(f"    spread = {spread:.2f} nats across all phases  [{verdict}];  "
          f"largest bonus goes to {Phase.NAMES[int(np.argmax(total))]}")
    print(f"    idle - grasp = {total[Phase.IDLE] - total[Phase.GRASP]:.2f} nats "
          f"(v7: 5.39, i.e. a 220:1 head start for idle on every frame)")


def report_separability(p: PhaseHMMParams) -> None:
    m, s = p.emission_mean, p.emission_std
    names = p.feature_names
    print("\n[2] SEPARABILITY -- |mean_i - mean_j| / pooled std, per feature")
    pairs = [(Phase.IDLE, Phase.GRASP), (Phase.APPROACH, Phase.GRASP),
             (Phase.GRASP, Phase.TRANSPORT), (Phase.TRANSPORT, Phase.PLACE),
             (Phase.IDLE, Phase.APPROACH)]
    print("    " + "pair".ljust(24) + "".join(nm[:9].rjust(10) for nm in names))
    for a, b in pairs:
        d = np.abs(m[a] - m[b]) / np.sqrt((s[a] ** 2 + s[b] ** 2) / 2)
        label = f"{Phase.NAMES[a]} vs {Phase.NAMES[b]}"
        print("    " + label.ljust(24) + "".join(f"{x:10.2f}" for x in d) + f"   max={d.max():.2f}")
    print("    a max well under ~1.0 means this pair is not distinguishable frame-by-frame")


def report_transition_margin(p: PhaseHMMParams) -> None:
    print("\n[3] TRANSITION MARGIN vs NORMALIZER BIAS -- which one actually wins")
    bias = p.normalizer_bias()
    for src, good, bad in [(Phase.APPROACH, Phase.GRASP, Phase.IDLE),
                           (Phase.TRANSPORT, Phase.PLACE, Phase.IDLE)]:
        # A realistic forward-filter belief: nearly all mass on the current
        # phase, with the small residue the filter always carries. The naive
        # log(transition) ratio understates how easy the wrong move is,
        # because residual mass in a sticky state feeds itself forward.
        b = np.full(Phase.N_CLASSES, 0.005)
        b[src] = 1.0 - 0.005 * (Phase.N_CLASSES - 1)
        pred = b @ p.transition
        margin = float(np.log(pred[good] / pred[bad]))
        against = float(bias[bad] - bias[good])
        ok = "OK" if margin > against else "LOSES"
        print(f"    from {Phase.NAMES[src]:<10} the prior favours {Phase.NAMES[good]:<10} over "
              f"{Phase.NAMES[bad]:<10} by {margin:6.2f} nats; "
              f"the emission bias favours {Phase.NAMES[bad]:<10} by {against:6.2f}  [{ok}]")


def _class_mean_frame(p: PhaseHMMParams, k: int) -> np.ndarray:
    """The exact class mean, binary dims rounded so the input stays a legal
    {0, 1} value rather than a probability the model can never observe."""
    x = p.emission_mean[k].copy()
    if p.bernoulli_p is not None:
        x = np.where(p.binary_mask, np.round(p.bernoulli_p[k]), x)
    return x


def report_self_recognition(p: PhaseHMMParams, hold: int = 60) -> None:
    print(f"\n[4] SELF-RECOGNITION -- feed exact class means, {hold} frames each")
    print("    Class means are a HARSHER input than real frames: they strip the per-frame")
    print("    variation that separates the short phases, so a miss here is not by itself")
    print("    a bug. What IS a bug is absorption -- see [4b].")
    order = [Phase.IDLE, Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT, Phase.PLACE]
    frames, truth = [], []
    for k in order:
        frames.extend([_class_mean_frame(p, k)] * hold)
        truth.extend([k] * hold)
    pred = np.argmax(filter_episode(p, np.stack(frames)), axis=1)
    truth = np.asarray(truth)

    for k in order:
        got = np.bincount(pred[truth == k], minlength=Phase.N_CLASSES)
        top = int(np.argmax(got))
        mark = "ok  " if top == k else "miss"
        print(f"    [{mark}] fed {Phase.NAMES[k]:<10} -> predicted {Phase.NAMES[top]:<10} "
              f"on {got[top]}/{hold} frames")
    dominant = np.bincount(pred, minlength=Phase.N_CLASSES)
    share = dominant.max() / len(pred)
    print(f"    self-recognition accuracy: {float((pred == truth).mean()):.3f};  "
          f"most-predicted class {Phase.NAMES[int(dominant.argmax())]} takes {share:.0%} "
          f"[{'OK' if share < 0.5 else 'one class is swallowing the sequence'}]")

    ll = emission_loglik(_class_mean_frame(p, Phase.GRASP), p)
    print("    emission log-lik at the GRASP mean: " +
          "  ".join(f"{Phase.NAMES[k]}={ll[k]:.1f}" for k in range(Phase.N_CLASSES)) +
          f"  -> argmax {Phase.NAMES[int(np.argmax(ll))]}")


# Below this transition probability, a phase pair is treated as structurally
# excluded rather than absorbing. The learned matrix contains genuine
# structural zeros -- GRASP -> PLACE never occurs, because every place is
# reached through TRANSPORT -- and a filter refusing to make that jump is
# obeying the data, not failing. Only Laplace smoothing keeps these entries
# above zero at all.
_STRUCTURAL_ZERO = 1e-4


def report_absorption(p: PhaseHMMParams, settle: int = 300, probe: int = 120) -> None:
    """The decisive check. v7's real failure was not confusing two similar
    phases, it was becoming UNABLE TO LEAVE one: fed the APPROACH mean it fell
    into IDLE, reached belief 1.000, and stayed there through 30 frames of the
    exact GRASP mean. Here: settle the filter in each phase, then feed a
    different phase's mean and measure how long it takes to switch.

    Pairs the transition matrix says never happen are reported separately, not
    as failures. Distinguishing the two matters: a real absorbing state is an
    emission bug, while a structural zero that cannot be crossed is the model
    correctly encoding task order, and 'fixing' it would be wrong.

    Read this alongside per-phase recall, not instead of it. Class means are
    the LEAST discriminative point of each class's distribution, so at a low
    emission temperature a pair can look stuck here while the same phase is
    recognised perfectly well on real frames -- v8 flags IDLE -> APPROACH but
    scores 0.83 approach recall on val. Treat a flag on a pair with healthy
    recall as a latency warning (how many frames until onset is detected)
    rather than a correctness one.
    """
    print(f"\n[4b] ABSORPTION -- settle {settle} frames in one phase, then feed another for {probe}")
    worst, stuck, structural = 0, [], []
    for src in range(Phase.N_CLASSES):
        for dst in range(Phase.N_CLASSES):
            if src == dst:
                continue
            frames = [_class_mean_frame(p, src)] * settle + [_class_mean_frame(p, dst)] * probe
            pred = np.argmax(filter_episode(p, np.stack(frames)), axis=1)
            start = int(pred[settle - 1])
            if start == dst:
                continue  # already there; nothing is being asked of the filter
            moved = np.nonzero(pred[settle:] != start)[0]
            if moved.size:
                worst = max(worst, int(moved[0]))
                continue
            if p.transition[start, dst] < _STRUCTURAL_ZERO:
                structural.append((start, dst))
            else:
                stuck.append((start, dst))

    for a, b in structural:
        print(f"     [order ] {Phase.NAMES[a]:<10} -> {Phase.NAMES[b]:<10} not crossed; "
              f"transition prior is {p.transition[a, b]:.1e} (never observed) -- expected")
    for a, b in stuck:
        print(f"     [STUCK ] settled in {Phase.NAMES[a]:<10} -> fed {Phase.NAMES[b]:<10} for "
              f"{probe} frames, belief never moved (prior {p.transition[a, b]:.1e})")

    if not stuck:
        print(f"     no absorbing states; slowest response to a phase change: {worst} frames  [OK]")
    else:
        print("     Check the per-phase recall for these phases before treating this as a bug:")
        print("     healthy recall + a flag here means slow ONSET detection, not absorption.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="one or more checkpoints/hmm/*.npz")
    ap.add_argument("--hold", type=int, default=60, help="frames per phase in the self-recognition test")
    args = ap.parse_args()

    for path in args.checkpoints:
        p = load_params(path)
        print("=" * 100)
        print(f"{path}   ({p.emission_mean.shape[1]} features, emission_temp={p.emission_temp:.3f}, "
              f"tie_covariance={p.tie_covariance}, binary dims={int(p.binary_mask.sum())})")
        print("=" * 100)
        report_normalizer_bias(p)
        report_separability(p)
        report_transition_margin(p)
        report_self_recognition(p, hold=args.hold)
        report_absorption(p)
        print()


if __name__ == "__main__":
    main()

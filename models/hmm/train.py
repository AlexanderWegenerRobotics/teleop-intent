"""Fits HMMIntentModel from labeled episodes and saves a checkpoint.

Run from the teleop-intent repo root (needs labels.py on sys.path):
    python -m models.hmm.train --store-root /path/to/avatar --out checkpoints/hmm/v1.npz

Only genuinely hand-labeled episodes are used (see labels.is_genuinely_labeled
-- a labels/ group alone isn't enough, label_tool.py creates one on every save
even if nothing was actually annotated). Silently skips anything else and
reports how many it found.
"""

from __future__ import annotations

import argparse
import csv
import os

import h5py
import numpy as np

from teleop_orchestrator.sources import ReplaySource
from teleop_orchestrator.contracts import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, Phase

import labels as label_loader
from .model import HMMIntentModel, SIDES
from .features import phase_features, world_ee_velocity
from .held import target_exclusion_mask
from .target import _MIN_EE_SPEED_FOR_ALIGNMENT

_GAZE_IDX = [GLOBAL_FEATURE_NAMES.index("gaze_x"), GLOBAL_FEATURE_NAMES.index("gaze_y")]
_PX_IDX = [CANDIDATE_FEATURE_NAMES.index("px_u"), CANDIDATE_FEATURE_NAMES.index("px_v")]


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_episode_arrays(hdf5_path: str) -> dict | None:
    """Returns {side: training-array-dict} for one labeled episode, or None
    if it has no genuine (non-default) hand labels -- see is_genuinely_labeled."""
    with h5py.File(hdf5_path, "r") as ep:
        if not label_loader.is_genuinely_labeled(ep):
            return None
        per_side_labels = {side: label_loader.load_arm_labels(ep, side) for side in SIDES}

    src = ReplaySource(hdf5_path, load_images=False)
    frames = [src.frame_at(t) for t in range(len(src))]
    src.close()

    out = {}
    for side in SIDES:
        phase_labels, target_labels = per_side_labels[side]
        feats_list, state = [], None
        world_ee_state, ee_vel_list, ee_pos_list = None, [], []
        for frame in frames:
            f, state = phase_features(frame, side, state)
            feats_list.append(f)
            # Separate from phase's own ee_vel (see model.py's step() and
            # features.py's world_ee_velocity docstring for why) -- the
            # alignment feature needs true world-frame EE motion, computed
            # with its own staleness-aware state since the intent log's
            # ee_{side}_x/y/z updates far less often than the frame grid.
            v, p, world_ee_state = world_ee_velocity(frame, side, world_ee_state)
            ee_vel_list.append(v)
            ee_pos_list.append(p)

        # Same target-pool exclusion model.py applies at inference (see
        # HMMIntentModel.step) -- must match here too, or the fitted
        # rho/sigma parameters are learned against a candidate pool that
        # doesn't match what runtime actually sees.
        target_masks = np.stack([fr.candidate_mask & ~target_exclusion_mask(fr, side) for fr in frames])

        out[side] = {
            "phase_features": np.stack(feats_list),
            "phase_labels": phase_labels,
            "gaze_xy": np.stack([fr.global_features[_GAZE_IDX] for fr in frames]),
            "candidate_px": np.stack([fr.candidate_features[:, _PX_IDX] for fr in frames]),
            "candidate_mask": target_masks,
            "gaze_valid": np.array([fr.gaze_valid for fr in frames]),
            "target_labels": target_labels,
            # Target alignment-channel inputs (see target.py / world_ee_velocity).
            # candidate_world_pos may be all-NaN per frame for episodes
            # without the slot_pos_* backfill -- that's fine, TargetStickyFilter
            # treats NaN as "no evidence" per candidate.
            "ee_vel": np.stack(ee_vel_list),
            "ee_pos": np.stack(ee_pos_list),
            "candidate_world_pos": np.stack([fr.candidate_world_pos for fr in frames]),
        }
    return out


def gather(store_root: str, episode_ids: list[str]) -> list[dict]:
    """Loads every labeled episode in episode_ids, flattened to one entry per (episode, side)."""
    data = []
    n_labeled = 0
    for eid in sorted(episode_ids):
        path = os.path.join(store_root, eid, "episode.hdf5")
        if not os.path.exists(path):
            continue
        arrays = build_episode_arrays(path)
        if arrays is None:
            continue
        n_labeled += 1
        data.extend(arrays[side] for side in SIDES)
    print(f"  {n_labeled} labeled episode(s) found")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True, help="root folder containing NNN/ episode subdirs")
    ap.add_argument("--train-split", default="data/splits/train.txt")
    ap.add_argument("--val-split", default="data/splits/val.txt")
    ap.add_argument("--out", default="checkpoints/hmm/v1.npz")
    # --- emission-model knobs (see models/hmm/phase.py for what each fixes) ---
    ap.add_argument("--std-floor-rel", type=float, default=None,
                    help="per-class std floor as a fraction of each feature's global std (default 0.25)")
    ap.add_argument("--emission-temp", default="auto",
                    help="'auto' selects on val; a float pins it (1.0 = pre-v8 behaviour)")
    ap.add_argument("--emission-temp-metric", default="macro_f1", choices=["macro_f1", "macro_recall", "accuracy"],
                    help="val metric the temperature sweep maximises (default macro_f1; see metrics.py)")
    ap.add_argument("--tie-covariance", action="store_true",
                    help="ABLATION: share one pooled covariance across phases, cancelling the "
                         "per-class normalizer bias exactly")
    ap.add_argument("--gaussian-only", action="store_true",
                    help="ABLATION: model binary features as Gaussians too, i.e. the pre-v8 emission")
    ap.add_argument("--target-committed-weight", type=float, default=None,
                    help="weight on committed frames in the target objective (default 0.75)")
    ap.add_argument("--sub-states", default="auto",
                    help="phase duration chains: 'auto' sizes each from its observed duration CV, "
                         "or give 5 comma-separated integers (e.g. 1,5,1,10,4). '1,1,1,1,1' is the "
                         "plain geometric-duration HMM and reproduces v9 exactly")
    ap.add_argument("--max-sub-states", type=int, default=None,
                    help="cap for --sub-states auto (default 12; hitting it means an explicit "
                         "duration model would be cheaper than more chain)")
    args = ap.parse_args()

    if args.sub_states == "auto":
        sub_states = "auto"
    else:
        sub_states = [int(x) for x in args.sub_states.replace(" ", "").split(",")]
        if len(sub_states) != Phase.N_CLASSES:
            raise SystemExit(f"--sub-states needs {Phase.N_CLASSES} integers, got {len(sub_states)}")

    config = {
        "emission_temp": args.emission_temp if args.emission_temp == "auto" else float(args.emission_temp),
        "emission_temp_metric": args.emission_temp_metric,
        "tie_covariance": args.tie_covariance,
        "gaussian_only": args.gaussian_only,
        "sub_states": sub_states,
    }
    if args.max_sub_states is not None:
        config["max_sub_states"] = args.max_sub_states
    if args.std_floor_rel is not None:
        config["std_floor_rel"] = args.std_floor_rel
    if args.target_committed_weight is not None:
        config["target_committed_weight"] = args.target_committed_weight

    print("Gathering train split...")
    train_data = gather(args.store_root, read_split(args.train_split))
    print("Gathering val split...")
    val_data = gather(args.store_root, read_split(args.val_split))

    if not train_data:
        raise SystemExit("No labeled training episodes found -- check --store-root and data/splits/train.txt")

    _report_alignment_coverage(train_data)

    model = HMMIntentModel()
    history = model.fit(train_data, val_data, config=config)
    _report(history, model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"\nsaved checkpoint to {args.out}")


def _report_alignment_coverage(data: list[dict]) -> None:
    """How often the EE/candidate alignment channel could fire at all.

    Printed before fitting because it is the first thing to check when
    sigma_align comes back as inf (channel disabled), which it did in v6, v7
    and v8. There are two very different reasons that can happen -- the
    evidence is present and simply unhelpful, or it was almost never
    available in the first place -- and only the second is worth chasing.
    The channel needs BOTH a non-stale world-frame EE velocity above
    target._MIN_EE_SPEED_FOR_ALIGNMENT AND a candidate with a backfilled
    world position (scripts/backfill_candidate_position.py); a low number
    here points at the backfill or at world_ee_velocity's staleness timeout,
    not at the feature being a bad idea.
    """
    if not data or "candidate_world_pos" not in data[0]:
        print("alignment channel: episode arrays carry no candidate_world_pos -- channel unavailable")
        return

    frames = usable_pos = moving = both = 0
    for d in data:
        mask = d["candidate_mask"]
        has_pos = np.isfinite(d["candidate_world_pos"]).all(axis=2) & mask   # [T, n_cand]
        pos_ok = has_pos.any(axis=1)                                          # [T]
        fast = np.linalg.norm(d["ee_vel"], axis=1) > _MIN_EE_SPEED_FOR_ALIGNMENT
        frames += len(pos_ok)
        usable_pos += int(pos_ok.sum())
        moving += int(fast.sum())
        both += int((pos_ok & fast).sum())

    pct = lambda x: f"{100.0 * x / max(frames, 1):5.1f}%"  # noqa: E731
    print(f"\nalignment channel coverage over {frames} training frames:")
    print(f"  {pct(usable_pos)} have a candidate with a world position")
    print(f"  {pct(moving)} have world EE speed above the alignment gate")
    print(f"  {pct(both)} have BOTH -- this is the fraction of frames the channel can ever vote on")


def _report(history: dict, model: HMMIntentModel) -> None:
    """Prints the fit summary, leading with the two numbers that told us v7
    was broken: the spread of the per-class emission normalizer (v7: 5.4 nats
    unconditionally favouring IDLE) and per-phase recall (v7: GRASP 0.15,
    PLACE 0.06). A healthy fit keeps the spread small and no phase near zero.
    """
    sel = history.get("sub_state_selection")
    ds = history.get("duration_stats")
    if ds is not None:
        print("\nphase duration model (sub-state chains):")
        print(f"  {'phase':<12}{'segments':>10}{'obs mean':>10}{'obs CV':>9}{'N':>4}"
              f"{'model mean':>12}{'model CV':>10}   note")
        for k in range(Phase.N_CLASSES):
            info = (sel or {}).get(k, {})
            n_seg = info.get("n_segments", 0)
            obs_mean, obs_cv = info.get("mean", float("nan")), info.get("cv", float("nan"))
            print(f"  {Phase.NAMES[k]:<12}{n_seg:>10}{obs_mean:>10.0f}{obs_cv:>9.2f}"
                  f"{ds['sub_states'][k]:>4}{ds['mean_duration'][k]:>12.0f}"
                  f"{ds['cv_duration'][k]:>10.2f}   {info.get('reason', '')}")
        print(f"  {ds['n_expanded_states']} expanded states for {ds['n_phases']} phases. "
              "Mean durations are preserved exactly;")
        print("  only the SHAPE changes. Emissions are shared within each chain, so this adds")
        print("  no new continuous parameters -- see models/hmm/phase.py for the derivation.")

    table = history.get("emission_temp_table")
    if table:
        metric = history.get("emission_temp_metric", "macro_f1")
        print(f"\nemission temperature sweep on val (selecting on {metric}):")
        print(f"  {'temp':>6}{'accuracy':>11}{'macro_rec':>11}{'macro_f1':>11}")
        for row in table:
            if row.get("prior_only"):
                mark = "  <-- REFERENCE: emission off, structure only"
            elif row["emission_temp"] == history["emission_temp"]:
                mark = "  <-- selected"
            else:
                mark = ""
            print(f"  {row['emission_temp']:>6.2f}{row['accuracy']:>11.3f}"
                  f"{row['macro_recall']:>11.3f}{row['macro_f1']:>11.3f}{mark}")

        floor = next((r for r in table if r.get("prior_only")), None)
        chosen = next((r for r in table if r["emission_temp"] == history["emission_temp"]), None)
        if floor and chosen:
            gain = chosen[metric] - floor[metric]
            print(f"  sensing is worth {gain:+.3f} {metric} over the structure-only floor "
                  f"({floor[metric]:.3f} -> {chosen[metric]:.3f})")
            if gain < 0.15:
                print("  WARNING: most of the score is coming from the transition/duration prior,")
                print("  not from the features. On a stereotyped task that means the model is")
                print("  largely replaying the expected script -- and it will fail on exactly the")
                print("  deviations (fumbles, regrasps, hesitation) an intent module exists for.")

        selectable = [r["emission_temp"] for r in table if not r.get("prior_only")]
        if history["emission_temp"] in (min(selectable), max(selectable)):
            print(f"  WARNING: selected temperature {history['emission_temp']} is on the edge of "
                  "the sweep grid -- widen it, the optimum may lie outside")

    bias = history.get("normalizer_bias")
    if bias is not None:
        print("\nemission normalizer bias per phase (observation-INDEPENDENT, nats):")
        print("  " + "  ".join(f"{Phase.NAMES[i]}={b:.2f}" for i, b in enumerate(bias)))
        print(f"  spread: {history['normalizer_bias_spread']:.2f} nats "
              f"(v7 was 5.39, unconditionally favouring IDLE over GRASP)")

    if "val_phase_accuracy" in history:
        print(f"\nval phase: accuracy={history['val_phase_accuracy']:.3f}  "
              f"macro_recall={history['val_phase_macro_recall']:.3f}  "
              f"macro_f1={history['val_phase_macro_f1']:.3f}")
        print("  per-phase recall: " + "  ".join(
            f"{k}={v:.3f}" for k, v in history["val_phase_recall_per_class"].items()))

    tp = model.target_params
    print(f"\ntarget filter: rho_loose={tp.rho_loose}  rho_tight={tp.rho_tight}  "
          f"sigma={tp.sigma}  null_prior={tp.null_prior}  sigma_align={tp.sigma_align}")
    if not np.isfinite(tp.sigma_align):
        print("  note: sigma_align=inf means the EE/candidate alignment channel is DISABLED "
              "at runtime (v6 and v7 both shipped this way)")


if __name__ == "__main__":
    main()

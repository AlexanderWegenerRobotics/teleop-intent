"""
labeling/train_prefill_models.py — trains the per-arm phase and target
classifiers that labeling/prefill.py uses to draft labels for episodes you
haven't hand-labeled yet.

Only episodes with labels.attrs["label_source"] == "manual" and
pass2_complete == True are used as training data. Do NOT relax this: the
~70-odd other episodes in the store carry labels from the old heuristic
auto-labeler (segment.py, since deleted) and treating those as ground truth
would just teach the model to reproduce the heuristic's mistakes.

Re-run this after labeling more episodes — more manually-labeled episodes
should make the pre-fill more accurate over time.

Usage:
    python labeling/train_prefill_models.py             # train + save models
    python labeling/train_prefill_models.py --validate   # also report
        leave-episode-out cross-validated accuracy first (an estimate of
        how well the method generalizes to an episode it's never seen —
        NOT the accuracy of the final saved models, which are fit on all
        available data).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import compute_ee_speed, load_config  # noqa: E402

try:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
except ImportError:
    sys.exit("This script needs scikit-learn, pandas, and joblib: "
              "pip install scikit-learn pandas joblib")

SIDES = ("left", "right")
PHASE_NAMES = {0: "IDLE", 1: "APPROACH", 2: "GRASP", 3: "TRANSPORT", 4: "PLACE"}
TGT_NAMES = ["null", "bin_1", "bin_2", "object_1", "object_2", "object_3", "object_4"]
KIN_FEATS = ["speed", "force", "gwidth", "gwidth_rate", "gcmd", "ee_belief", "ee_belief_other"]
BEL_FEATS = [f"bel_{n}" for n in TGT_NAMES]
MODELS_DIR = Path(__file__).parent / "models"


def resolve_store_root(cfg: dict) -> str:
    store_root = cfg.get("data", {}).get("store_root")
    if store_root:
        return store_root
    dataset_cfg_path = ROOT / "configs" / "dataset.yaml"
    if dataset_cfg_path.exists():
        return load_config(str(dataset_cfg_path))["data"]["store_root"]
    sys.exit("data.store_root must be set in configs/playback.yaml or configs/dataset.yaml")


def find_manual_episodes(store_root: str) -> list[str]:
    eps = []
    for p in sorted(glob.glob(f"{store_root}/*/episode.hdf5")):
        with h5py.File(p, "r") as f:
            lbl = f.get("labels")
            if (lbl is not None
                    and lbl.attrs.get("label_source", None) == "manual"
                    and bool(lbl.attrs.get("pass2_complete", False))
                    and "arm_left_phase" in lbl and "arm_right_phase" in lbl):
                eps.append(p)
    return eps


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def extract_episode(path: str):
    ep_id = Path(path).parent.name
    rows = []
    with h5py.File(path, "r") as f:
        obs = f["observations"]
        act = f.get("actions")
        intent = obs.get("intent")
        lbl = f["labels"]
        T = obs["timestamp_ns"].shape[0]
        belief_cols = {name: (intent[f"tgt_belief_{name}"][:]
                               if intent is not None and f"tgt_belief_{name}" in intent
                               else np.zeros(T))
                       for name in TGT_NAMES}
        for side in SIDES:
            arm_obs = obs.get(f"arm_{side}")
            arm_act = act.get(f"arm_{side}") if act is not None else None
            other = "right" if side == "left" else "left"

            speed = (compute_ee_speed(arm_obs["O_T_EE"][:])
                     if arm_obs is not None and "O_T_EE" in arm_obs else np.zeros(T))
            force = (np.linalg.norm(arm_obs["F_ext"][:, :3], axis=1)
                     if arm_obs is not None and "F_ext" in arm_obs else np.zeros(T))
            gwidth = (arm_obs["gripper_width"][:]
                      if arm_obs is not None and "gripper_width" in arm_obs else np.zeros(T))
            gcmd = (arm_act["gripper_cmd"][:]
                    if arm_act is not None and "gripper_cmd" in arm_act else np.zeros(T))
            ee_bel = (intent[f"ee_belief_ee_{side}"][:]
                      if intent is not None and f"ee_belief_ee_{side}" in intent else np.zeros(T))
            ee_bel_other = (intent[f"ee_belief_ee_{other}"][:]
                             if intent is not None and f"ee_belief_ee_{other}" in intent else np.zeros(T))

            phase = lbl[f"arm_{side}_phase"][:]
            target = np.array([_decode(v) for v in lbl[f"arm_{side}_target_name"][:]])

            data = {
                "episode_id": ep_id, "side": side,
                "speed": speed, "force": force, "gwidth": gwidth,
                "gwidth_rate": np.gradient(gwidth), "gcmd": gcmd,
                "ee_belief": ee_bel, "ee_belief_other": ee_bel_other,
                "phase": phase, "target": target,
            }
            for name in TGT_NAMES:
                data[f"bel_{name}"] = belief_cols[name]
            rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True)


def build_dataset(store_root: str):
    eps = find_manual_episodes(store_root)
    if not eps:
        sys.exit(f"No manually-labeled episodes found under {store_root}")
    print(f"training on {len(eps)} manually-labeled episodes")
    return pd.concat([extract_episode(p) for p in eps], ignore_index=True)


def validate(df) -> None:
    print("=== leave-episode-out cross-validation (method estimate, not final-model accuracy) ===")
    for task, feats, y_col in [("phase", KIN_FEATS, "phase"), ("target", KIN_FEATS + BEL_FEATS, "target")]:
        for side in SIDES:
            sub = df[df["side"] == side].reset_index(drop=True)
            X = sub[feats].values
            y = sub[y_col].values
            groups = sub["episode_id"].values
            n_splits = min(5, sub["episode_id"].nunique())
            gkf = GroupKFold(n_splits=n_splits)
            accs = []
            for train_idx, test_idx in gkf.split(X, y, groups):
                depth = 10 if task == "phase" else 12
                m = RandomForestClassifier(n_estimators=80, max_depth=depth, n_jobs=-1,
                                            random_state=0, class_weight="balanced_subsample")
                m.fit(X[train_idx], y[train_idx])
                accs.append(accuracy_score(y[test_idx], m.predict(X[test_idx])))
            print(f"  {task}/{side}: held-out acc mean={np.mean(accs):.3f} std={np.std(accs):.3f}  (n_splits={n_splits})")
    print()


def train_final_models(df) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for side in SIDES:
        sub = df[df["side"] == side]

        phase_model = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1,
                                              random_state=0, class_weight="balanced_subsample")
        phase_model.fit(sub[KIN_FEATS].values, sub["phase"].values)
        joblib.dump(phase_model, MODELS_DIR / f"phase_{side}.joblib")

        target_model = RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1,
                                               random_state=0, class_weight="balanced_subsample")
        target_model.fit(sub[KIN_FEATS + BEL_FEATS].values, sub["target"].values)
        joblib.dump(target_model, MODELS_DIR / f"target_{side}.joblib")

        print(f"saved models for side={side} -> {MODELS_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "playback.yaml"))
    ap.add_argument("--store-root", default=None,
                     help="Override the episode store root instead of reading it from config "
                          "(useful when the config's path doesn't match where the store is mounted).")
    ap.add_argument("--validate", action="store_true",
                     help="Run leave-episode-out CV and report accuracy before training the final models.")
    args = ap.parse_args()

    if args.store_root:
        store_root = args.store_root
    else:
        cfg_path = Path(args.config)
        cfg = load_config(str(cfg_path)) if cfg_path.exists() else {"data": {}}
        store_root = resolve_store_root(cfg)

    df = build_dataset(store_root)

    if args.validate:
        validate(df)

    train_final_models(df)
    print("done.")


if __name__ == "__main__":
    main()

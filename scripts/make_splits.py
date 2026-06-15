"""make_splits.py  -  Generate reproducible train/val/test episode splits.

Reads the manifest, applies the inclusion rules from the config (which outcomes
to keep, minimum gaze validity), then splits the surviving episodes by id into
train/val/test. Splitting is always by episode (never within an episode, which
would leak future frames across splits) and is stratified so each split stays
representative on the chosen keys. Writes plain-text id lists, one id per line.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")


def load_config(path):
    """Loads the YAML config and returns its 'data' and 'split' sections; exits
    if the file is missing since the split is config-defined by design."""
    if not os.path.exists(path):
        sys.exit(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("data", {}), cfg.get("split", {})


def apply_inclusion(df, split_cfg):
    """Filters the manifest to the episodes eligible for splitting, by outcome
    and minimum gaze validity; returns the kept frame and prints what was
    dropped so exclusions are visible rather than silent."""
    keep_success = split_cfg.get("include_success", ["success"])
    min_gaze = float(split_cfg.get("min_gaze_valid_frac", 0.0))

    before = len(df)
    df = df[df["error"].fillna("") == ""]
    kept = df[df["success"].isin(keep_success)].copy()

    if min_gaze > 0 and "gaze_valid_frac" in kept.columns:
        gv = kept["gaze_valid_frac"].fillna(0.0)
        dropped_gaze = kept[gv < min_gaze]
        kept = kept[gv >= min_gaze]
        if len(dropped_gaze):
            print(f"  dropped {len(dropped_gaze)} episode(s) below gaze>={min_gaze}: "
                  f"{list(dropped_gaze['folder'])}")

    excluded = before - len(kept)
    print(f"  included {len(kept)} / {before} episodes (excluded {excluded})")
    for outcome in sorted(set(df["success"]) - set(keep_success)):
        n = int((df["success"] == outcome).sum())
        if n:
            print(f"      excluded by outcome '{outcome}': {n}")
    return kept


def stratified_split(df, ratios, stratify_keys, seed):
    """Assigns each episode to train/val/test within strata defined by the
    stratify keys, so every split stays representative; returns three lists of
    episode ids. Falls back to a plain random split if no keys are given."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []

    if stratify_keys:
        df = df.copy()
        df["_stratum"] = df[stratify_keys].astype(str).agg("|".join, axis=1)
        groups = [g for _, g in df.groupby("_stratum")]
    else:
        groups = [df]

    for g in groups:
        ids = g["episode_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        train += ids[:n_train]
        val += ids[n_train:n_train + n_val]
        test += ids[n_train + n_val:]

    return sorted(train), sorted(val), sorted(test)


def write_split(out_dir, name, ids):
    """Writes one split as a plain-text file of episode ids, one per line, for
    easy diffing and version control."""
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print(f"  wrote {path}  ({len(ids)} episodes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()

    data_cfg, split_cfg = load_config(args.config)
    manifest_path = data_cfg.get("manifest", "data/manifest.csv")
    if not os.path.exists(manifest_path):
        sys.exit(f"Manifest not found: {manifest_path}. Run build_manifest.py first.")

    df = pd.read_csv(manifest_path)
    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        sys.exit(f"Split ratios must sum to 1.0, got {sum(ratios.values())}")

    print("Applying inclusion rules ...")
    kept = apply_inclusion(df, split_cfg)
    if kept.empty:
        sys.exit("No episodes left after inclusion filtering.")

    print("Generating stratified split ...")
    train, val, test = stratified_split(
        kept, ratios, split_cfg.get("stratify_by", []), split_cfg.get("seed", 0))

    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        sys.exit(f"Split overlap detected (bug): {overlap}")

    out_dir = split_cfg.get("out_dir", "data/splits")
    os.makedirs(out_dir, exist_ok=True)
    write_split(out_dir, "train", train)
    write_split(out_dir, "val", val)
    write_split(out_dir, "test", test)
    print(f"  total split: {len(train)+len(val)+len(test)} episodes (seed {split_cfg.get('seed', 0)})")


if __name__ == "__main__":
    main()

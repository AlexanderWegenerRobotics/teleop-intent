"""
Scan all episodes into a single manifest.

Walks the global data store, opens every episode.hdf5, and records one row per
episode (id, path, length, outcome, integrity flags) into data/manifest.csv.
The manifest is a faithful inventory: it records EVERY episode regardless of
outcome and makes no inclusion decisions — filtering happens later at split
time, driven by config. Folder names need not be consecutive; each episode is
identified by the id stored inside its hdf5, not by folder position.

Engagement: the operator passes through non-ENGAGED states (e.g. IDLE before
the HMD is on, per-device resets). Those frames are not valid training samples,
so the manifest reports the engaged fraction and computes gaze validity over
engaged frames only — the statistics that actually matter. It still records
every episode; the per-frame masking happens later in the loader.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --config configs/dataset.yaml
    python scripts/build_manifest.py --store-root /data/teleop --out data/manifest.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    """Reads a YAML config if present and returns its 'data' and 'state'
    sections, or empty dicts when no config/file is available."""
    if not path or yaml is None or not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("data", {}), cfg.get("state", {})


def resolve_train_states(state_cfg):
    """Maps the configured train-state names to their integer codes via the
    enum in config; defaults to ENGAGED=4 when no config is present."""
    enum = state_cfg.get("enum", {"ENGAGED": 4})
    names = state_cfg.get("train_states", ["ENGAGED"])
    return [int(enum[n]) for n in names if n in enum]


def decode_attr(val):
    """Normalizes an HDF5 attribute to a plain Python value, decoding bytes to
    str so JSON-string attrs like color_bin_mapping are readable."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):
        return val.item()
    return val


def find_episode_files(store_root, episode_glob):
    """Returns the sorted list of episode.hdf5 paths under store_root matching
    the folder glob; tolerates gaps and non-consecutive folder names."""
    pattern = os.path.join(store_root, episode_glob, "episode.hdf5")
    return sorted(glob.glob(pattern))


def engaged_mask(obs, train_states):
    """Returns a boolean per-frame mask of timesteps where the system is in a
    training state, requiring BOTH arms to be engaged; returns None if state
    was not carried into the hdf5."""
    states = []
    for arm in ("arm_left", "arm_right"):
        g = obs.get(arm)
        if g is not None and "state" in g:
            states.append(g["state"][:].astype(np.int64))
    if not states:
        return None
    both = np.ones(len(states[0]), dtype=bool)
    for s in states:
        both &= np.isin(s, train_states)
    return both


def scan_episode(path, train_states):
    """Opens one episode.hdf5 and extracts metadata and integrity flags into a
    dict; on any read error returns a row flagged with the error instead of
    raising, so one bad file never aborts the whole scan."""
    row = {"path": path, "folder": os.path.basename(os.path.dirname(path)), "error": ""}
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            row["episode_id"] = decode_attr(a.get("episode_id", row["folder"]))
            row["success"] = decode_attr(a.get("success", ""))
            row["seed"] = decode_attr(a.get("seed", ""))
            row["mode"] = decode_attr(a.get("mode", ""))
            row["color_bin_mapping"] = decode_attr(a.get("color_bin_mapping", ""))
            row["rate_hz"] = decode_attr(a.get("rate_hz", ""))
            row["image_scale"] = decode_attr(a.get("image_scale", ""))
            row["schema_version"] = decode_attr(a.get("schema_version", ""))

            obs = f.get("observations")
            if obs is not None and "timestamp_ns" in obs:
                ts = obs["timestamp_ns"][:]
                row["n_frames"] = int(len(ts))
                row["duration_s"] = round(float((ts[-1] - ts[0]) / 1e9), 3) if len(ts) > 1 else 0.0
            else:
                row["n_frames"] = 0
                row["duration_s"] = 0.0

            mask = engaged_mask(obs, train_states) if obs is not None else None
            if mask is not None:
                row["n_engaged"] = int(mask.sum())
                row["engaged_frac"] = round(float(mask.mean()), 3) if len(mask) else 0.0
                row["has_state"] = True
            else:
                row["n_engaged"] = row.get("n_frames", 0)
                row["engaged_frac"] = float("nan")
                row["has_state"] = False

            intent = obs.get("intent") if obs is not None else None
            row["has_intent"] = bool(intent is not None and len(intent.keys()) > 0)
            if row["has_intent"] and "gaze_valid" in intent:
                gv = intent["gaze_valid"][:] > 0.5
                row["gaze_valid_frac"] = round(float(gv.mean()), 3) if len(gv) else 0.0
                if mask is not None and mask.sum() > 0 and len(gv) == len(mask):
                    row["gaze_valid_engaged"] = round(float(gv[mask].mean()), 3)
                else:
                    row["gaze_valid_engaged"] = float("nan")
            else:
                row["gaze_valid_frac"] = float("nan")
                row["gaze_valid_engaged"] = float("nan")

            imgs = obs.get("images") if obs is not None else None
            cams = sorted(imgs.keys()) if imgs is not None else []
            row["cameras"] = ",".join(cams)
            row["has_images"] = bool(cams)

            scene = f.get("scene")
            if scene is not None:
                row["n_objects"] = sum(1 for i in range(8) if f"obj{i}_pose" in scene)
                row["n_bins"] = sum(1 for i in range(8) if f"bin{i}_pose" in scene)
            else:
                row["n_objects"] = 0
                row["n_bins"] = 0
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def print_summary(df):
    """Prints a human-readable summary of the scan: totals, outcome breakdown,
    engagement, and any episodes missing intent/images/state or flagged with
    errors — the integrity readout over the whole dataset."""
    print(f"\n{'='*52}\nMANIFEST SUMMARY\n{'='*52}")
    print(f"  Episodes found      : {len(df)}")
    errs = df[df["error"] != ""]
    print(f"  Read errors         : {len(errs)}")
    for _, r in errs.iterrows():
        print(f"      {r['folder']}: {r['error']}")

    ok = df[df["error"] == ""]
    if len(ok):
        print(f"\n  Outcome breakdown   :")
        for outcome, n in ok["success"].value_counts().items():
            print(f"      {str(outcome) or '<none>':12s} {n}")

        print(f"\n  Missing intent group: {(~ok['has_intent']).sum()}")
        print(f"  Missing images      : {(~ok['has_images']).sum()}")
        if "has_state" in ok.columns:
            print(f"  Missing state       : {(~ok['has_state']).sum()}")

        eng = ok["engaged_frac"].dropna()
        if len(eng):
            print(f"  Engaged fraction    : min={eng.min():.2f}  mean={eng.mean():.2f}  max={eng.max():.2f}")

        gv = ok["gaze_valid_frac"].dropna()
        gve = ok["gaze_valid_engaged"].dropna()
        if len(gv):
            print(f"  Gaze-valid (all)    : min={gv.min():.2f}  mean={gv.mean():.2f}  max={gv.max():.2f}")
        if len(gve):
            print(f"  Gaze-valid (engaged): min={gve.min():.2f}  mean={gve.mean():.2f}  max={gve.max():.2f}")
            low = ok[ok["gaze_valid_engaged"] < 0.5]
            if len(low):
                print(f"      low engaged-gaze (<0.5): {list(low['folder'])}")

        print(f"  Total frames        : {int(ok['n_frames'].sum())}")
        if "n_engaged" in ok.columns:
            print(f"  Engaged frames      : {int(ok['n_engaged'].sum())}")
    print(f"{'='*52}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml", help="dataset config (YAML)")
    ap.add_argument("--store-root", default=None, help="override data store root")
    ap.add_argument("--episode-glob", default=None, help="override episode folder glob")
    ap.add_argument("--out", default=None, help="override manifest output path")
    args = ap.parse_args()

    data_cfg, state_cfg = load_config(args.config)
    store_root = args.store_root or data_cfg.get("store_root") or "data/raw"
    episode_glob = args.episode_glob or data_cfg.get("episode_glob") or "[0-9][0-9][0-9]"
    out_path = args.out or data_cfg.get("manifest") or "data/manifest.csv"
    train_states = resolve_train_states(state_cfg)

    print(f"Scanning {store_root!r} (glob {episode_glob!r}) ...")
    print(f"  train states (engaged): {train_states}")
    files = find_episode_files(store_root, episode_glob)
    if not files:
        sys.exit(f"No episode.hdf5 files found under {store_root!r}.")

    rows = [scan_episode(p, train_states) for p in files]
    df = pd.DataFrame(rows)

    cols = ["episode_id", "folder", "path", "n_frames", "n_engaged", "engaged_frac",
            "duration_s", "success", "seed", "mode", "color_bin_mapping",
            "has_intent", "gaze_valid_frac", "gaze_valid_engaged", "has_state",
            "cameras", "has_images", "n_objects", "n_bins", "rate_hz",
            "image_scale", "schema_version", "error"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} episodes)")
    print_summary(df)


if __name__ == "__main__":
    main()
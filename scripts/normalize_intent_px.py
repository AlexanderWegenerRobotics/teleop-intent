"""
scripts/normalize_intent_px.py  -  Normalize pixel features in existing episode HDF5 files.

Rewrites gaze_px_x/y and slot_px_* columns inside observations/intent from raw
pixel coordinates to [0, 1] UV space.  Other columns are copied unchanged.

Coordinate conventions:
  gaze_px_x   : logged in full-stereo space [0, full_stereo_w) -> divide by full_stereo_w
  gaze_px_y   : [0, full_stereo_h)                              -> divide by full_stereo_h
  slot_px_*_x : per-eye space [0, eye_w) where eye_w = full_stereo_w / 2 -> divide by eye_w
  slot_px_*_y : [0, full_stereo_h)                              -> divide by full_stereo_h

The true full-stereo native width is derived from the stored per-eye image width
and the image_scale attribute, NOT from gaze_px_ref_width (which may have been
stored as half the correct value by an earlier converter version).

Use --force to re-normalize files that were previously normalized with the wrong
reference (detected when stored ref_w != derived true_ref_w).

Uses in-place dataset replacement (del + create) so no file copy or replace is
needed.  The HDF5 del does not reclaim bytes, but the two float columns are
negligible in size compared to images.

Usage:
    python scripts/normalize_intent_px.py --store C:/path/to/avatar
    python scripts/normalize_intent_px.py --store C:/path/to/avatar --dry-run
    python scripts/normalize_intent_px.py --store C:/path/to/avatar --force
    python scripts/normalize_intent_px.py --file path/to/episode.hdf5 --force
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import h5py

def _true_full_stereo_w(f: h5py.File) -> float | None:
    """Derive full-stereo native width from per-eye image shape and image_scale attr."""
    scale = float(f.attrs.get("image_scale", 0))
    if scale <= 0:
        return None
    imgs = f["observations"].get("images")
    if imgs is None:
        return None
    for eye in ("head_cam_right", "head_cam_left"):
        if eye in imgs:
            per_eye_stored_w = imgs[eye].shape[2]
            return per_eye_stored_w * 2.0 / scale
    return None


def normalize_file(path: str, dry_run: bool, force: bool) -> str:
    with h5py.File(path, "r") as src:
        intent = src["observations"].get("intent")
        if intent is None:
            return "skip (no intent group)"

        already_normalized = bool(intent.attrs.get("px_normalized", False))
        stored_ref_w = float(intent.attrs.get("gaze_px_ref_width", 0))
        ref_h = float(intent.attrs.get("gaze_px_ref_height", 0))

        true_ref_w = _true_full_stereo_w(src)
        if true_ref_w is None:
            return "skip (cannot derive true ref width — missing image_scale or head_cam images)"
        if ref_h <= 0:
            return "skip (missing gaze_px_ref_height attr)"

        eye_w = true_ref_w / 2.0

        if already_normalized:
            if not force:
                return "skip (already normalized; use --force to re-normalize)"
            if abs(stored_ref_w - true_ref_w) < 1:
                return "skip (already normalized with correct ref)"
            rescale = stored_ref_w / true_ref_w
            action = f"rescale x{rescale:.3f} (was ref {int(stored_ref_w)}, correct {int(true_ref_w)})"
        else:
            rescale = None
            action = f"normalize (ref {int(true_ref_w)}x{int(ref_h)})"

        cols_to_norm = [k for k in intent.keys()
                        if k in ("gaze_px_x", "gaze_px_y")
                        or (k.startswith("slot_px_") and k.endswith(("_x", "_y")))]

        if dry_run:
            return f"would {action} — {len(cols_to_norm)} columns"

    os.chmod(path, 0o666)
    with h5py.File(path, "r+") as f:
        intent = f["observations"]["intent"]
        for col in cols_to_norm:
            arr = intent[col][:].astype(np.float32)
            if rescale is not None:
                if col == "gaze_px_x":
                    arr *= rescale
                elif col.endswith("_x"):
                    arr *= rescale
            else:
                if col == "gaze_px_x":
                    arr /= true_ref_w
                elif col == "gaze_px_y":
                    arr /= ref_h
                elif col.endswith("_x"):
                    arr /= eye_w
                elif col.endswith("_y"):
                    arr /= ref_h
            del intent[col]
            intent.create_dataset(col, data=arr)
        intent.attrs["px_normalized"] = True
        intent.attrs["gaze_px_ref_width"] = int(true_ref_w)

    return f"ok — {action}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--store", help="Root folder containing NNN/ episode subdirs")
    group.add_argument("--file", help="Single episode.hdf5 to normalize")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything")
    ap.add_argument("--force", action="store_true", help="Re-normalize even if px_normalized=True (fixes wrong-ref runs)")
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.store, "*", "episode.hdf5")))
        if not files:
            sys.exit(f"No episode.hdf5 files found under {args.store}")

    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} file(s)")
    ok = skip = fail = 0
    for path in files:
        ep_id = os.path.basename(os.path.dirname(path))
        try:
            result = normalize_file(path, args.dry_run, args.force)
            print(f"  {ep_id}: {result}")
            if result.startswith("ok"):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  {ep_id}: FAILED — {e}")
            fail += 1

    print(f"\nDone: {ok} normalized, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()

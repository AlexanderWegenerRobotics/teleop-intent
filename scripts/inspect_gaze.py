"""
Quick diagnostic: print gaze UV values and where they'd appear on each camera.

Usage:
    python scripts/inspect_gaze.py --episode 076
    python scripts/inspect_gaze.py --episode 076 --frames 10
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import h5py
    import yaml
except ImportError:
    sys.exit("pip install h5py pyyaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", type=int, default=20, help="number of frames to sample")
    args = ap.parse_args()

    with open(ROOT / "configs" / "dataset.yaml") as f:
        store_root = yaml.safe_load(f)["data"]["store_root"]
    path = Path(store_root) / args.episode / "episode.hdf5"

    with h5py.File(path, "r") as ep:
        intent = ep["observations"]["intent"]
        imgs   = ep["observations"]["images"]

        px_norm  = bool(intent.attrs.get("px_normalized", False))
        ref_w    = float(intent.attrs.get("gaze_px_ref_width", 0))
        ref_h    = float(intent.attrs.get("gaze_px_ref_height", 0))

        left_shape  = imgs["head_cam_left"].shape[1:]  if "head_cam_left"  in imgs else None
        right_shape = imgs["head_cam_right"].shape[1:] if "head_cam_right" in imgs else None

        gaze_x = intent["gaze_px_x"][:]
        gaze_y = intent["gaze_px_y"][:]
        valid  = intent["gaze_valid"][:].astype(bool) if "gaze_valid" in intent else np.ones(len(gaze_x), bool)

        print(f"Episode : {args.episode}")
        print(f"px_normalized   : {px_norm}")
        print(f"gaze_px_ref     : {ref_w} x {ref_h}")
        print(f"head_cam_left   : {left_shape}")
        print(f"head_cam_right  : {right_shape}")
        print(f"gaze_px_x range : [{gaze_x.min():.4f}, {gaze_x.max():.4f}]")
        print(f"gaze_px_y range : [{gaze_y.min():.4f}, {gaze_y.max():.4f}]")
        print(f"gaze_valid frac : {valid.mean():.2%}")
        print()

        if px_norm:
            uv_x = gaze_x
            uv_y = gaze_y
            eye_w_uv = 0.5
        else:
            full_w = ref_w if ref_w > 0 else (right_shape[1] * 2 if right_shape else 1280)
            full_h = ref_h if ref_h > 0 else (right_shape[0] if right_shape else 960)
            uv_x = gaze_x / full_w
            uv_y = gaze_y / full_h
            eye_w_uv = 0.5

        sw = right_shape[1] if right_shape else 160
        sh = right_shape[0] if right_shape else 240

        indices = np.linspace(0, len(gaze_x) - 1, args.frames, dtype=int)
        print(f"{'t':>5}  {'uv_x':>6}  {'uv_y':>6}  {'valid':>5}  {'direct_gx':>9}  {'gy':>6}  (direct = uv_x * stored_w)")
        print("-" * 65)
        for t in indices:
            ux, uy = float(uv_x[t]), float(uv_y[t])
            v = bool(valid[t])
            gy = uy * sh
            direct_gx = ux * sw

            print(f"{t:>5}  {ux:>6.3f}  {uy:>6.3f}  {str(v):>5}  {direct_gx:>9.1f}  {gy:>6.1f}")


if __name__ == "__main__":
    main()

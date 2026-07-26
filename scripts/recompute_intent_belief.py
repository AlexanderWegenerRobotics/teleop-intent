"""
scripts/recompute_intent_belief.py  -  Recompute slot_px and slot_belief in HDF5 files.

The original data was computed with head_position=[0,0,1.2] (hardcoded default).
The correct value is [0,0,1.704] (from robot_config base_pose).
This script recomputes all slot projections and beliefs using the correct head position.

Sources per episode (all in the raw episode data folder):
  head.csv        — pan/tilt angles (q_0=pan, q_1=tilt) with wall_clock_ns
  scene.csv       — object/bin world poses with wall_clock_ns
  episode.hdf5    — camera intrinsics (attrs on head_cam_left dataset),
                    intention log resampled onto the 30Hz HDF5 grid,
                    timestamp_arrival_ns used for time alignment

The projection math mirrors intention_buffer.cpp exactly:
  R_CH = R_tilt(q1) * R_pan(q0)          (head orientation)
  p_H  = R_CH * (p_world - head_pos)     (into head frame)
  p_C  = p_H - cam_offset                (subtract camera-in-head offset)
  p_CV = R_body2cv * p_C                 (body -> OpenCV axes)
  u    = fx * p_CV.x / p_CV.z + cx
  v    = fy * p_CV.y / p_CV.z + cy

Camera extrinsics from robot_config_avatar.yaml (head camera):
  position:   [0.05, 0.0, 0.035]
  euler_xyz:  [1.5708, -1.5708, 0.0]   (ZYX applied as Rz*Ry*Rx)

Usage:
    python scripts/recompute_intent_belief.py --store C:/path/to/avatar
    python scripts/recompute_intent_belief.py --store C:/path/to/avatar --dry-run
    python scripts/recompute_intent_belief.py --file path/to/episode/folder
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("pip install h5py")

# ---------------------------------------------------------------------------
# Constants — mirror robot_config_avatar.yaml / intention_buffer.hpp
# ---------------------------------------------------------------------------

HEAD_POSITION = np.array([0.0, 0.0, 1.844])   # tilt joint in world: base_pose(1.704) + link_1(0.08) + link_2(0.06)
GAZE_SIGMA_PX = 30.0                           # from IntentionBufferConfig default
NULL_PRIOR    = 0.1                            # belief[N] prior for null slot
BELIEF_TEMP   = 3.0                            # from robot_config camera.belief_temperature
EMA_ALPHA     = 0.3                            # kept for reference; superseded by sticky Bayes
STICKY_RHO_EE  = 0.85                          # P(stay) for end-effector attention (looser — EE attention shifts often)
STICKY_RHO_TGT = 0.95                          # P(stay) for object/bin attention (tighter — target intent is sticky)

# Camera offset in head frame (from robot_config camera.position / euler_xyz)
CAM_OFFSET = np.array([0.05, 0.0, 0.035])

def _euler_xyz_to_rotation(ex, ey, ez):
    """Build rotation matrix from euler_xyz = Rz * Ry * Rx (same as avatar.cpp)."""
    cx, sx = np.cos(ex), np.sin(ex)
    cy, sy = np.cos(ey), np.sin(ey)
    cz, sz = np.cos(ez), np.sin(ez)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

# Not used for projection itself (avatar.cpp doesn't apply cam orientation to p_C_body,
# only uses position offset). Kept here for reference.
_CAM_ORIENTATION = _euler_xyz_to_rotation(1.5708, -1.5708, 0.0)

# Body -> OpenCV axis remap (from intention_buffer.cpp):
#   body X -> CV Z, body Y -> CV -X, body Z -> CV -Y
R_BODY2CV = np.array([
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1,  0,  0],
], dtype=float)

# Slot types (from intention_sample.hpp)
SLOT_EE_LEFT  = 0
SLOT_EE_RIGHT = 1
SLOT_PICK_OBJ = 2
SLOT_PLACE    = 3  # bins


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _R_pan(pan: float) -> np.ndarray:
    c, s = np.cos(pan), np.sin(pan)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def _R_tilt(tilt: float) -> np.ndarray:
    c, s = np.cos(tilt), np.sin(tilt)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def project_to_image(p_world: np.ndarray, pan: float, tilt: float, intrinsics: dict) -> tuple[float, float] | None:
    """Project a world point to (u, v) in native camera pixels. Returns None if behind camera."""
    R_CH = _R_tilt(tilt) @ _R_pan(-pan)  # tilt: axis=+Y matches R_Y(q_tilt); pan: passive → negate
    p_H  = R_CH @ (p_world - HEAD_POSITION)
    p_C  = p_H - CAM_OFFSET
    p_CV = R_BODY2CV @ p_C
    if p_CV[2] <= 0:
        return None
    u = intrinsics["fx"] * p_CV[0] / p_CV[2] + intrinsics["cx"]
    v = intrinsics["fy"] * p_CV[1] / p_CV[2] + intrinsics["cy"]
    return float(u), float(v)


def slot_likelihood(gaze_u: float, gaze_v: float,
                    center: np.ndarray,
                    half_extents: np.ndarray,
                    pan: float, tilt: float,
                    intrinsics: dict) -> float:
    """Gaussian likelihood at gaze vs projected slot center + corners."""
    sigma2 = GAZE_SIGMA_PX ** 2

    def gaussian(p):
        uv = project_to_image(p, pan, tilt, intrinsics)
        if uv is None:
            return 0.0
        du, dv = gaze_u - uv[0], gaze_v - uv[1]
        return float(np.exp(-(du*du + dv*dv) / (2 * sigma2)))

    best = gaussian(center)
    if np.sum(half_extents**2) > 1e-8:
        dx, dy, dz = half_extents
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    best = max(best, gaussian(center + np.array([sx*dx, sy*dy, sz*dz])))
    return best


def compute_ee_belief(gaze_u: float, gaze_v: float,
                      kernels: list[tuple[np.ndarray, np.ndarray]],
                      pan: float, tilt: float,
                      intrinsics: dict,
                      temperature: float = BELIEF_TEMP) -> np.ndarray:
    """EE-only belief: no null slot. Normalises ee_left vs ee_right only.
    Without null, the distribution always sums to 1 between the two EEs —
    entropy on this signal later tells us whether attention is clearly on one arm.
    """
    raw = np.array([
        slot_likelihood(gaze_u, gaze_v, c, h, pan, tilt, intrinsics)
        for c, h in kernels
    ], dtype=np.float32)
    if temperature != 1.0:
        raw = np.power(raw, 1.0 / temperature)
    total = raw.sum()
    return raw / total if total > 1e-9 else np.ones(len(kernels), dtype=np.float32) / len(kernels)


def compute_belief(gaze_u: float, gaze_v: float,
                   kernels: list[tuple[np.ndarray, np.ndarray]],
                   pan: float, tilt: float,
                   intrinsics: dict,
                   temperature: float = BELIEF_TEMP) -> np.ndarray:
    """Compute normalised belief over N slots + 1 null slot."""
    N = len(kernels)
    belief = np.array([
        slot_likelihood(gaze_u, gaze_v, c, h, pan, tilt, intrinsics)
        for c, h in kernels
    ] + [NULL_PRIOR], dtype=np.float32)

    if temperature != 1.0:
        belief = np.power(belief, 1.0 / temperature)

    total = belief.sum()
    if total > 1e-6:
        belief /= total
    return belief


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        header = f.readline().strip().split(";")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def load_head_angles(episode_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (wall_clock_ns, pan_rad, tilt_rad) arrays from head.csv."""
    path = os.path.join(episode_dir, "head.csv")
    if not os.path.exists(path):
        return None
    hdr, data = _load_csv(path)
    if "wall_clock_ns" not in hdr or "q_0" not in hdr or "q_1" not in hdr:
        return None
    ts  = data[:, hdr.index("wall_clock_ns")].astype(np.int64)
    pan = data[:, hdr.index("q_0")]
    tilt= data[:, hdr.index("q_1")]
    return ts, pan, tilt


def load_scene(episode_dir: str) -> dict | None:
    """Returns dict with wall_clock_ns and per-object/bin position arrays and names."""
    import csv as _csv

    path = os.path.join(episode_dir, "scene.csv")
    if not os.path.exists(path):
        return None

    with open(path, newline="") as f:
        reader = _csv.reader(f, delimiter=";")
        hdr = next(reader)
        rows = list(reader)

    if not rows or "wall_clock_ns" not in hdr:
        return None

    # Determine which column indices are numeric (can be parsed as float).
    # Skips name/label string columns so they don't break the numeric array.
    def _is_numeric(col_idx: int) -> bool:
        for row in rows:
            if col_idx < len(row) and row[col_idx].strip():
                try:
                    float(row[col_idx])
                    return True
                except ValueError:
                    return False
        return True  # empty column — treat as numeric

    num_col_indices = [i for i in range(len(hdr)) if _is_numeric(i)]
    num_col_map = {orig_i: new_i for new_i, orig_i in enumerate(num_col_indices)}

    # Build numeric array (positional data only)
    data = np.array([[float(row[i]) if i < len(row) and row[i].strip() else float("nan")
                      for i in num_col_indices] for row in rows], dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Remap hdr lookups to use reduced numeric column indices
    def _num_idx(col: str) -> int:
        orig = hdr.index(col)
        return num_col_map[orig]

    ts = data[:, _num_idx("wall_clock_ns")].astype(np.int64)
    n_objects = int(data[0, _num_idx("n_objects")]) if "n_objects" in hdr else 0
    n_bins    = int(data[0, _num_idx("n_bins")])    if "n_bins"    in hdr else 0

    # Helper: read a string column from the raw rows (constant across time; take row 0)
    def _str_col(prefix: str) -> str:
        col = prefix + "name"
        if col in hdr:
            return rows[0][hdr.index(col)].strip()
        return ""

    out = {"ts": ts, "objects": [], "bins": []}
    for i in range(n_objects):
        p = f"obj{i}_"
        if p+"x" not in hdr:
            break
        pos = np.stack([data[:, _num_idx(p+"x")],
                        data[:, _num_idx(p+"y")],
                        data[:, _num_idx(p+"z")]], axis=1)
        out["objects"].append({"pos": pos, "name": _str_col(p)})
    for i in range(n_bins):
        p = f"bin{i}_"
        if p+"x" not in hdr:
            break
        pos = np.stack([data[:, _num_idx(p+"x")],
                        data[:, _num_idx(p+"y")],
                        data[:, _num_idx(p+"z")]], axis=1)
        out["bins"].append({"pos": pos, "name": _str_col(p)})
    return out


def nearest_idx(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp find the nearest index in src_ts."""
    idx = np.searchsorted(src_ts, query_ts)
    idx = np.clip(idx, 0, len(src_ts) - 1)
    left = np.clip(idx - 1, 0, len(src_ts) - 1)
    use_left = np.abs(src_ts[left] - query_ts) < np.abs(src_ts[idx] - query_ts)
    idx[use_left] = left[use_left]
    return idx


def get_intrinsics(ep: h5py.File, ep_dir: str) -> dict | None:
    """Read camera intrinsics, preferring camera_params.json in the episode folder."""
    import json

    scale = float(ep.attrs.get("image_scale", 0.25))

    # 1. camera_params.json written by Avatar::writeCameraParams() — most reliable
    params_path = os.path.join(ep_dir, "camera_params.json")
    if not os.path.exists(params_path):
        # also check one level up (shared across episodes)
        params_path = os.path.join(ep_dir, "..", "camera_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        # prefer head_head_cam; fall back to any entry with fx
        for name in ("head_head_cam", *params.keys()):
            if name in params and "fx" in params[name]:
                p = params[name]
                return {"fx": p["fx"], "fy": p["fy"],
                        "cx": p["cx"], "cy": p["cy"],
                        "image_scale": scale}

    # 2. Fallback: attrs on HDF5 eye datasets
    imgs = ep["observations"].get("images")
    if imgs is not None:
        for cam in ("head_cam_left", "head_cam_right"):
            if cam in imgs:
                ds = imgs[cam]
                if all(k in ds.attrs for k in ("fx", "fy", "cx", "cy")):
                    return {"fx": float(ds.attrs["fx"]), "fy": float(ds.attrs["fy"]),
                            "cx": float(ds.attrs["cx"]), "cy": float(ds.attrs["cy"]),
                            "image_scale": scale}

    return None


# ---------------------------------------------------------------------------
# Main recompute
# ---------------------------------------------------------------------------

def _sticky_bayes(raw: np.ndarray, rho: float) -> np.ndarray:
    """Apply sticky Bayesian filter along axis 0 (time).

    raw: (T, N) array of per-frame normalized likelihoods (already temperature-scaled).
    Returns smoothed (T, N) posterior array.
    Each row of raw must sum to 1; output rows also sum to 1.
    """
    T, N = raw.shape
    out = np.zeros_like(raw)
    out[0] = raw[0]
    uniform = np.ones(N) / N
    for t in range(1, T):
        p_pred  = rho * out[t - 1] + (1.0 - rho) * uniform
        post    = raw[t] * p_pred
        total   = post.sum()
        out[t]  = post / total if total > 1e-9 else p_pred
    return out


def recompute_episode(hdf5_path: str, dry_run: bool,
                      ema_alpha: float = EMA_ALPHA,
                      rho_ee: float  = STICKY_RHO_EE,
                      rho_tgt: float = STICKY_RHO_TGT) -> str:
    ep_dir = str(Path(hdf5_path).parent)

    with h5py.File(hdf5_path, "r") as ep:
        intent = ep["observations"].get("intent")
        if intent is None:
            return "skip (no intent group)"

        intrinsics = get_intrinsics(ep, ep_dir)
        if intrinsics is None:
            return "skip (no camera intrinsics in HDF5)"

        if "timestamp_arrival_ns" not in intent:
            return "skip (no timestamp_arrival_ns in intent)"

        arrival_ns = intent["timestamp_arrival_ns"][:].astype(np.int64)
        T = len(arrival_ns)

        # Slot structure from intent group
        n_slots_arr = intent["n_slots"][:].astype(int) if "n_slots" in intent else np.zeros(T, int)
        slot_types_keys = sorted(k for k in intent.keys() if k.startswith("slot_type_"))
        n_slot_cols = len(slot_types_keys)

        # EE positions (world frame) — stored in intent group
        ee_left  = np.stack([intent["ee_left_x"][:],  intent["ee_left_y"][:],  intent["ee_left_z"][:]], axis=1)
        ee_right = np.stack([intent["ee_right_x"][:], intent["ee_right_y"][:], intent["ee_right_z"][:]], axis=1)

        # Gaze — already normalized; convert back to native pixels for belief computation
        px_norm   = bool(intent.attrs.get("px_normalized", False))
        gaze_px_x = intent["gaze_px_x"][:]
        gaze_px_y = intent["gaze_px_y"][:]
        ref_w = float(intent.attrs.get("gaze_px_ref_width",  0))
        ref_h = float(intent.attrs.get("gaze_px_ref_height", 0))

        # Old slot pixel data (to verify we're changing something)
        old_slot_px_u = {k: intent[k][:] for k in intent.keys() if k.startswith("slot_px_u_")}

        # gaze_valid may be stored per-frame as uint8
        gaze_valid_arr = intent["gaze_valid"][:] if "gaze_valid" in intent else np.ones(T, dtype=np.uint8)

    # Load head angles
    head = load_head_angles(ep_dir)
    if head is None:
        return "skip (no head.csv)"
    head_ts, head_pan, head_tilt = head

    # Load scene
    scene = load_scene(ep_dir)
    if scene is None:
        return "skip (no scene.csv)"

    # Align head and scene to intent arrival timestamps
    head_idx  = nearest_idx(head_ts, arrival_ns)
    scene_idx = nearest_idx(scene["ts"], arrival_ns)

    image_scale = intrinsics["image_scale"]

    # Build slot names (constant for the episode, scene.csv order):
    #   slot 0: ee_left, slot 1: ee_right, slots 2+: objects, then bins
    slot_names_ep = ["ee_left", "ee_right"]
    for obj in scene["objects"]:
        slot_names_ep.append(obj["name"] or f"obj{len(slot_names_ep)-2}")
    for bn in scene["bins"]:
        slot_names_ep.append(bn["name"] or f"bin{len(slot_names_ep)-2-len(scene['objects'])}")

    # Slot name lists (used for name-keyed HDF5 storage — avoids any lex-sort issues)
    ee_names  = ["ee_left", "ee_right"]           # always exactly 2 EE slots
    tgt_names = ([obj["name"] for obj in scene["objects"]] +
                 [bn["name"]  for bn  in scene["bins"]])

    N_ee  = len(ee_names)   # 2
    N_tgt = len(tgt_names)  # up to 8

    # Output arrays: separate EE and target belief streams + projection arrays
    # EE:  [ee_left, ee_right]               → N_ee columns (no null — entropy used later)
    # Tgt: [obj0..objN, bin0..binM, null]    → N_tgt+1 columns
    MAX_SLOTS = 10
    new_slot_px_u  = np.full((T, MAX_SLOTS), -1.0, dtype=np.float32)
    new_slot_px_v  = np.full((T, MAX_SLOTS), -1.0, dtype=np.float32)
    new_ee_belief  = np.zeros((T, N_ee),     dtype=np.float32)
    new_tgt_belief = np.zeros((T, N_tgt + 1), dtype=np.float32)

    cam_w = intrinsics["cx"] * 2  # 1280 native px
    cam_h = intrinsics["cy"] * 2  # 960  native px

    for t in range(T):
        pan  = float(head_pan[head_idx[t]])
        tilt = float(head_tilt[head_idx[t]])
        si   = scene_idx[t]
        gaze_valid = bool(gaze_valid_arr[t])

        if px_norm:
            gaze_u = float(gaze_px_x[t]) * cam_w
            gaze_v = float(gaze_px_y[t]) * cam_h
        else:
            gaze_u = float(gaze_px_x[t]) * 0.5
            gaze_v = float(gaze_px_y[t])

        # Build kernels — keep full list for projection; split for belief
        ee_kernels  = [(ee_left[t],  np.zeros(3)),
                       (ee_right[t], np.zeros(3))]
        tgt_kernels = [(obj["pos"][si], np.zeros(3)) for obj in scene["objects"]] + \
                      [(bn["pos"][si],  np.zeros(3)) for bn  in scene["bins"]]
        all_kernels = ee_kernels + tgt_kernels

        # Project all slots (joint list, stored in slot_px_u_i / slot_px_v_i)
        for i, (center, _) in enumerate(all_kernels):
            if i >= MAX_SLOTS:
                break
            uv = project_to_image(center, pan, tilt, intrinsics)
            if uv is not None:
                new_slot_px_u[t, i] = uv[0]
                new_slot_px_v[t, i] = uv[1]

        # Separate belief computation per group
        if gaze_valid:
            new_ee_belief[t]  = compute_ee_belief(gaze_u, gaze_v, ee_kernels,  pan, tilt, intrinsics)
            new_tgt_belief[t] = compute_belief(gaze_u, gaze_v, tgt_kernels, pan, tilt, intrinsics)
        else:
            # EE: uniform (no strong evidence either way)
            new_ee_belief[t]  = 1.0 / N_ee
            # Target: all mass on null
            new_tgt_belief[t, N_tgt] = 1.0

    # Separate sticky Bayesian smoothing for EE and target streams
    new_ee_belief  = _sticky_bayes(new_ee_belief,  rho_ee)
    new_tgt_belief = _sticky_bayes(new_tgt_belief, rho_tgt)

    if dry_run:
        # Report max change in slot_px_u_0 as sanity check
        if "slot_px_u_0" in old_slot_px_u:
            old_u0 = old_slot_px_u["slot_px_u_0"]
            new_u0 = new_slot_px_u[:, 0]
            valid  = (old_u0 > 0) & (new_u0 > 0)
            delta  = np.abs(new_u0[valid] - old_u0[valid]).mean() if valid.any() else float("nan")
            return f"would recompute — mean |Δslot_px_u_0| = {delta:.1f} native px"
        return "would recompute"

    # Write back
    os.chmod(hdf5_path, 0o666)
    with h5py.File(hdf5_path, "r+") as ep:
        intent = ep["observations"]["intent"]

        # Slot projections (joint order: ee_left, ee_right, obj0..N, bin0..M)
        for i in range(MAX_SLOTS):
            for prefix, arr in (("slot_px_u_", new_slot_px_u), ("slot_px_v_", new_slot_px_v)):
                key = f"{prefix}{i}"
                if key in intent:
                    del intent[key]
                intent.create_dataset(key, data=arr[:, i])

        # EE belief — keyed by name: ee_belief_ee_left, ee_belief_ee_right (no null)
        for i, name in enumerate(ee_names):
            key = f"ee_belief_{name}"
            if key in intent: del intent[key]
            intent.create_dataset(key, data=new_ee_belief[:, i])
        if "ee_belief_null" in intent: del intent["ee_belief_null"]  # remove stale null if present

        # Target belief — keyed by name: tgt_belief_object_1, ..., tgt_belief_null
        for i, name in enumerate(tgt_names):
            key = f"tgt_belief_{name}"
            if key in intent: del intent[key]
            intent.create_dataset(key, data=new_tgt_belief[:, i])
        if "tgt_belief_null" in intent: del intent["tgt_belief_null"]
        intent.create_dataset("tgt_belief_null", data=new_tgt_belief[:, N_tgt])

        # Remove old monolithic slot_belief_* datasets (replaced by ee_belief_* / tgt_belief_*)
        for key in list(intent.keys()):
            if key.startswith("slot_belief_"):
                del intent[key]
        # Store slot names (scene.csv order) as variable-length string datasets
        dt_str = h5py.string_dtype()
        for i, name in enumerate(slot_names_ep[:MAX_SLOTS]):
            key = f"slot_name_{i}"
            if key in intent:
                del intent[key]
            intent.create_dataset(key, data=np.array([name] * T, dtype=object), dtype=dt_str)
        intent.attrs["belief_head_position"] = list(HEAD_POSITION)
        intent.attrs["belief_recomputed"]    = True
        intent.attrs["belief_rho_ee"]         = rho_ee
        intent.attrs["belief_rho_tgt"]        = rho_tgt
        intent.attrs["slot_order"]           = "scene_csv"  # slot indices follow scene.csv object order

    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--store", help="Root folder containing NNN/ episode subdirs")
    grp.add_argument("--file",  help="Single episode folder (containing episode.hdf5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rho-ee",  type=float, default=STICKY_RHO_EE,
                    help=f"Sticky Bayes ρ for EE attention (default={STICKY_RHO_EE})")
    ap.add_argument("--rho-tgt", type=float, default=STICKY_RHO_TGT,
                    help=f"Sticky Bayes ρ for object/bin attention (default={STICKY_RHO_TGT})")
    args = ap.parse_args()

    if args.file:
        hdf5 = args.file if args.file.endswith(".hdf5") else os.path.join(args.file, "episode.hdf5")
        files = [hdf5]
    else:
        files = sorted(glob.glob(os.path.join(args.store, "*", "episode.hdf5")))
        if not files:
            sys.exit(f"No episode.hdf5 found under {args.store}")

    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} episode(s)")
    ok = skip = fail = 0
    for path in files:
        ep_id = Path(path).parent.name
        try:
            result = recompute_episode(path, args.dry_run, rho_ee=args.rho_ee, rho_tgt=args.rho_tgt)
            print(f"  {ep_id}: {result}")
            if result.startswith("ok"):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  {ep_id}: FAILED — {e}")
            import traceback; traceback.print_exc()
            fail += 1

    print(f"\nDone: {ok} recomputed, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()

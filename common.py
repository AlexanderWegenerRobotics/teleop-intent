"""
common.py — shared, dependency-light helpers used by both viz/playback.py
(Rerun-based review tool) and labeling/label_tool.py (standalone Tkinter
labeler).

Deliberately has NO dependency on rerun-sdk or Tkinter: it's imported by two
tools that otherwise share nothing (one needs a spawned Rerun viewer, the
other needs Tk/PIL/matplotlib), and neither should have to install the
other's dependencies just to reuse this math. Everything here is pure
numpy/h5py/yaml logic: config loading, candidate naming, EE speed, and the
gaze/slot-box pixel projection so the two tools can never silently drift
apart on how a gaze point or a slot box maps onto the image.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import yaml

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")


# ---------------------------------------------------------------------------
# Phase enum (shared by playback, labeling, and the old segment.py schema)
# ---------------------------------------------------------------------------

PHASE_NAMES = {0: "IDLE", 1: "APPROACH", 2: "GRASP", 3: "TRANSPORT", 4: "PLACE"}
PHASE_IDS = {name: pid for pid, name in PHASE_NAMES.items()}  # "IDLE" -> 0


# ---------------------------------------------------------------------------
# Config / candidate naming
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Returns merged config: file defaults overridden by CLI flags later."""
    with open(path) as f:
        return yaml.safe_load(f)


def candidate_names_from_cfg(cfg: dict, n_slots: int) -> list[str]:
    """Returns per-candidate display names from config, padding with slot indices."""
    names = cfg["display"].get("candidate_names") or []
    result = list(names[:n_slots])
    for i in range(len(result), n_slots):
        result.append(f"slot_{i}")
    return result


def candidate_names_from_hdf5(intent, n_slots: int, cfg: dict) -> list[str]:
    """Returns slot display names from HDF5 slot_name_* datasets (written by recompute).

    If slot_name_{i} datasets are present (written by recompute_intent_belief in
    scene.csv order), use those — they match the belief and projection indices exactly.
    Falls back to config-based names if HDF5 names are absent.
    """
    hdf5_names = []
    for i in range(n_slots):
        key = f"slot_name_{i}"
        if key in intent:
            val = intent[key][0]
            if isinstance(val, (bytes, np.bytes_)):
                val = val.decode("utf-8")
            hdf5_names.append(str(val))
        else:
            break
    if len(hdf5_names) == n_slots:
        return hdf5_names
    return candidate_names_from_cfg(cfg, n_slots)


def gaze_color(cfg: dict) -> list[int]:
    """Returns the [R, G, B] gaze dot color from config."""
    return cfg["gaze"].get("color", [245, 158, 11])


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

def compute_ee_speed(o_t_ee: np.ndarray, smooth_win: int = 9) -> np.ndarray:
    """Returns smoothed EE speed (m/frame) from an O_T_EE [T,16] array's
    translation column (indices 12:15). Used as a phase-judgment aid: high
    speed usually means TRANSPORT/APPROACH, near-zero means GRASP/PLACE/IDLE."""
    pos = o_t_ee[:, 12:15]
    vel = np.gradient(pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    win = max(1, min(smooth_win, len(speed)))
    kernel = np.ones(win) / win
    return np.convolve(speed, kernel, mode="same")


# ---------------------------------------------------------------------------
# Gaze / slot-box pixel projection (shared math — do not duplicate!)
# ---------------------------------------------------------------------------

def compute_gaze_px(intent, t: int, cfg: dict, img_shape: tuple) -> Optional[tuple[float, float]]:
    """Returns the (x, y) pixel position of the gaze point on the primary
    camera image at frame t, in stored-image pixel space, or None if gaze is
    disabled/invalid/out of range for this frame.

    Gaze UV is in full-stereo space [0, 1] where 0.5 = boundary between eyes.
    Both left and right camera images span the same scene (stereo pair with
    small baseline), so gaze UV maps approximately directly to each camera's
    pixel space.
    """
    if not cfg["gaze"].get("show", True):
        return None
    if "gaze_px_x" not in intent or "gaze_px_y" not in intent:
        return None
    if "gaze_valid" in intent and not bool(intent["gaze_valid"][t]):
        return None

    stored_h, stored_w = img_shape[:2]
    px_normalized = bool(intent.attrs.get("px_normalized", False))

    raw_x = float(intent["gaze_px_x"][t])
    raw_y = float(intent["gaze_px_y"][t])

    if px_normalized:
        gaze_uv_x, gaze_uv_y = raw_x, raw_y
    else:
        gaze_ref_w = float(intent.attrs.get("gaze_px_ref_width", stored_w * 2))
        gaze_ref_h = float(intent.attrs.get("gaze_px_ref_height", stored_h))
        full_w = gaze_ref_w if gaze_ref_w > stored_w else stored_w * 2
        gaze_uv_x = raw_x / full_w
        gaze_uv_y = raw_y / gaze_ref_h

    if gaze_uv_x < 0.0 or gaze_uv_x > 1.0:
        return None

    return gaze_uv_x * stored_w, gaze_uv_y * stored_h


def compute_slot_boxes(intent, t: int, img_shape: tuple, ep, names: list[str]) -> tuple[list[dict], list[dict]]:
    """Returns (ee_boxes, tgt_boxes): lists of {"u", "v", "belief", "label"}
    dicts for the projected end-effector and object/bin slots at frame t, in
    stored-image pixel space. Belief is 0..1 (0 if no belief data present) —
    callers map it to a color however suits their renderer.
    """
    n_slots_val = int(intent["n_slots"][t]) if "n_slots" in intent else 0
    if n_slots_val == 0:
        return [], []

    image_scale = float(ep.attrs.get("image_scale", 0.25))
    stored_h, stored_w = img_shape[:2]

    has_split = any(k.startswith("ee_belief_") for k in intent.keys())
    if has_split:
        belief_by_name = {}
        for k in intent.keys():
            if k.startswith("ee_belief_") and k != "ee_belief_null":
                belief_by_name[k[len("ee_belief_"):]] = float(intent[k][t])
            elif k.startswith("tgt_belief_") and k != "tgt_belief_null":
                belief_by_name[k[len("tgt_belief_"):]] = float(intent[k][t])
    else:
        legacy_keys = sorted((k for k in intent.keys() if k.startswith("slot_belief_")),
                             key=lambda k: int(k.rsplit("_", 1)[-1]))
        belief_by_name = {names[i]: float(intent[k][t])
                          for i, k in enumerate(legacy_keys) if i < len(names)}

    EE_SLOT_NAMES = {"ee_left", "ee_right"}
    ee_boxes, tgt_boxes = [], []

    for i in range(n_slots_val):
        u_key, v_key = f"slot_px_u_{i}", f"slot_px_v_{i}"
        if u_key not in intent or v_key not in intent:
            break
        u_native, v_native = float(intent[u_key][t]), float(intent[v_key][t])
        if u_native < 0 or v_native < 0:
            continue
        u_stored, v_stored = u_native * image_scale, v_native * image_scale
        if not (0 <= u_stored <= stored_w and 0 <= v_stored <= stored_h):
            continue
        name = names[i] if i < len(names) else f"slot_{i}"
        entry = {"u": u_stored, "v": v_stored, "belief": belief_by_name.get(name, 0.0), "label": name}
        (ee_boxes if name in EE_SLOT_NAMES else tgt_boxes).append(entry)

    return ee_boxes, tgt_boxes

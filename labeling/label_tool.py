"""
labeling/label_tool.py — intent labeling and review tool.

Phases come from labeling/schema.yaml, the per-task phase subset and goals
from labeling/tasks/<task>.yaml. Auto labels (labeling/auto_label.py) are
imported from auto_labels.json next to the episode and reviewed boundary by
boundary; every frame you touch is marked manual, and re-importing auto labels
never overwrites manual frames.

Writes labels/arm_{side}_phase, arm_{side}_target_name (object, same as
before), arm_{side}_goal_name, arm_{side}_source (0 default, 1 auto, 2 manual)
and labels.attrs events/reviewed/schema_version/task.

Usage:
    python labeling/label_tool.py --task labeling/tasks/fmb_board_1.yaml --store-root D:/data/fmb-1-desk
    python labeling/label_tool.py --episode 076

Keys:
    ←/→ step 1, shift+←/→ step 10, space play, 0-9 pick phase, enter label to here,
    n/p next/prev boundary, f next flag, [ ] move boundary 1 frame, { } 10 frames,
    tab switch arm, ctrl+s save, ctrl+z undo
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")
try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    sys.exit("Pillow is required: pip install pillow")
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import compute_ee_speed, compute_gaze_px, compute_slot_boxes, candidate_names_from_hdf5, load_config  # noqa: E402

SCHEMA_PATH = HERE / "schema.yaml"
DEFAULT_TASK = HERE / "tasks" / "sorting.yaml"
AUTO_NAME = "auto_labels.json"
SRC_DEFAULT, SRC_AUTO, SRC_MANUAL = 0, 1, 2
DISPLAY_SCALE = 2.6
OBJ_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
              "#9c755f", "#bab0ac"]


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


class Schema:
    def __init__(self, schema_path: Path, task_path: Path):
        s = load_yaml(schema_path)
        t = load_yaml(task_path)
        self.version = s["schema_version"]
        self.task = t.get("task", Path(task_path).stem)
        self.all = {p["name"]: p for p in s["phases"]}
        self.ids = {p["name"]: p["id"] for p in s["phases"]}
        self.names = {p["id"]: p["name"] for p in s["phases"]}
        self.colors = {p["id"]: p["color"] for p in s["phases"]}
        self.task_phases = [n for n in t.get("phases", []) if n in self.all] or \
                           [p["name"] for p in s["phases"] if not p.get("reserved")]
        self.goals = list((t.get("goals") or {}).keys())
        self.goal_pos = {k: np.array(v["pos"], float) for k, v in (t.get("goals") or {}).items()}
        self.goal_type = {k: v.get("type", "") for k, v in (t.get("goals") or {}).items()}
        self.use_bins_as_goals = bool(t.get("bins_are_goals", not self.goals))
        self.null = s.get("null_target", "null")

    def legend(self) -> str:
        return "  ".join(f"{i}={n}" for i, n in sorted(self.names.items()))


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def part_type(name: str) -> str:
    parts = name.replace("spare_", "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def segment_bounds(keys: list) -> list[int]:
    return [t for t in range(1, len(keys)) if keys[t] != keys[t - 1]]


# ---------------------------------------------------------------------------

def discover_episodes(store_root: Path) -> list[str]:
    manifest = ROOT / "data" / "manifest.csv"
    found = sorted(p.parent.name for p in store_root.glob("*/episode.hdf5"))
    if found:
        return found
    if manifest.exists():
        with open(manifest, newline="") as f:
            return [r["episode_id"] for r in csv.DictReader(f) if r.get("episode_id")]
    return []


def episode_status(store_root: Path, eid: str) -> str:
    folder = store_root / eid
    tags = []
    try:
        with h5py.File(folder / "episode.hdf5", "r") as ep:
            if "labels" in ep:
                tags.append("reviewed" if ep["labels"].attrs.get("reviewed", False) else "labeled")
    except Exception:
        pass
    if (folder / AUTO_NAME).exists():
        tags.append("auto")
    return f"{eid}   {' '.join(tags)}"


class EpisodeSession:
    def __init__(self, path: Path, cfg: dict, schema: Schema):
        self.path = path
        self.folder = path.parent
        self.cfg = cfg
        self.schema = schema
        self.ep = h5py.File(path, "r")
        self._bind()
        obs = self.obs
        self.T = len(self.timestamps)
        self.sides = [s for s in ("left", "right") if f"arm_{s}" in obs] or ["left"]

        cams = list(self.imgs.keys())
        primary = cfg.get("camera", {}).get("primary", "head_cam_left")
        self.primary_cam = primary if primary in cams else cams[0]
        self.extra_cams = [c for c in cams if c != self.primary_cam and "wrist" in c]

        self.names = []
        if self.intent is not None and "n_slots" in self.intent:
            try:
                self.names = candidate_names_from_hdf5(self.intent, int(self.intent["n_slots"][0]), cfg)
            except Exception:
                self.names = []

        self.obj_names, self.obj_pos, bins = self._scene_objects()
        self.goal_names = list(schema.goals) + (bins if schema.use_bins_as_goals else [])

        self.speed, self.force, self.gwidth, self.gcmd = {}, {}, {}, {}
        for side in self.sides:
            a = obs.get(f"arm_{side}")
            act = self.act.get(f"arm_{side}") if self.act is not None else None
            self.speed[side] = compute_ee_speed(a["O_T_EE"][:]) if a is not None and "O_T_EE" in a else np.zeros(self.T)
            self.force[side] = np.linalg.norm(a["F_ext"][:, :3], axis=1) if a is not None and "F_ext" in a else np.zeros(self.T)
            self.gwidth[side] = a["gripper_width"][:] if a is not None and "gripper_width" in a else np.zeros(self.T)
            self.gcmd[side] = act["gripper_cmd"][:] if act is not None and "gripper_cmd" in act else np.zeros(self.T)

        self.goal_dist = self._goal_distances()
        self.cam = self._camera_model(self.primary_cam)

        self.phase, self.obj, self.goal, self.src = {}, {}, {}, {}
        self.events: list[dict] = []
        self.flags: list[dict] = []
        self.reviewed = False
        self.loaded_from = "blank"
        self._load_labels()
        self._history: list[tuple] = []

    def _bind(self):
        self.obs = self.ep["observations"]
        self.act = self.ep.get("actions")
        self.imgs = self.obs["images"]
        self.intent = self.obs.get("intent")
        self.timestamps = self.obs["timestamp_ns"][:].astype(np.int64)

    def _scene_objects(self):
        names, pos, bins = [], [], []
        sc = self.ep.get("scene")
        if sc is None:
            return names, pos, bins
        for i in range(20):
            if f"obj{i}_pose" not in sc:
                continue
            p = sc[f"obj{i}_pose"][:, :3]
            if np.nanmean(p[:, 2]) < 0.3:
                continue
            n = _decode(sc[f"obj{i}_name"][()]).strip("\x00 ") if f"obj{i}_name" in sc else f"object_{i + 1}"
            names.append(n)
            pos.append(p)
        for i in range(20):
            if f"bin{i}_pose" in sc:
                n = _decode(sc[f"bin{i}_name"][()]).strip("\x00 ") if f"bin{i}_name" in sc else f"bin_{i + 1}"
                bins.append(n)
        return names, pos, bins

    def _camera_model(self, cam):
        a = self.imgs[cam].attrs
        if not all(k in a for k in ("fx", "fy", "cx", "cy", "T_world_cam")):
            return None
        T = np.asarray(a["T_world_cam"], float)
        return {"R": T[:3, :3], "t": T[:3, 3], "fx": float(a["fx"]), "fy": float(a["fy"]),
                "cx": float(a["cx"]), "cy": float(a["cy"]), "scale": float(self.ep.attrs.get("image_scale", 0.25))}

    def project(self, pts: np.ndarray, display_scale: float):
        c = self.cam
        if c is None:
            return None
        pc = (np.atleast_2d(pts) - c["t"]) @ c["R"]
        z = -pc[:, 2]
        k = c["scale"] * display_scale
        u = (c["cx"] + c["fx"] * pc[:, 0] / np.maximum(z, 1e-6)) * k
        v = (c["cy"] - c["fy"] * pc[:, 1] / np.maximum(z, 1e-6)) * k
        u[z <= 0.05] = np.nan
        return np.stack([u, v], 1)

    def _goal_distances(self):
        out = {}
        for n, p in zip(self.obj_names, self.obj_pos):
            gps = [gp for g, gp in self.schema.goal_pos.items() if self.schema.goal_type.get(g) == part_type(n)]
            if gps:
                out[n] = np.min([np.linalg.norm(p - gp, axis=1) for gp in gps], axis=0)
        return out

    def _load_labels(self):
        lbl = self.ep.get("labels")
        null = self.schema.null
        for side in self.sides:
            self.phase[side] = np.zeros(self.T, np.int8)
            self.obj[side] = [null] * self.T
            self.goal[side] = [null] * self.T
            self.src[side] = np.zeros(self.T, np.int8)
        if lbl is not None and any(f"arm_{s}_phase" in lbl for s in self.sides):
            for side in self.sides:
                pk, tk, gk, sk = (f"arm_{side}_phase", f"arm_{side}_target_name",
                                  f"arm_{side}_goal_name", f"arm_{side}_source")
                if pk in lbl and len(lbl[pk]) == self.T:
                    self.phase[side] = lbl[pk][:].astype(np.int8)
                if tk in lbl and len(lbl[tk]) == self.T:
                    self.obj[side] = [_decode(v).rstrip("\x00") for v in lbl[tk][:]]
                if gk in lbl and len(lbl[gk]) == self.T:
                    self.goal[side] = [_decode(v).rstrip("\x00") for v in lbl[gk][:]]
                if sk in lbl and len(lbl[sk]) == self.T:
                    self.src[side] = lbl[sk][:].astype(np.int8)
                else:
                    self.src[side][:] = SRC_MANUAL
            self.reviewed = bool(lbl.attrs.get("reviewed", False))
            if "events" in lbl.attrs:
                self.events = json.loads(lbl.attrs["events"])
            self.loaded_from = "hdf5"
            self._load_flags_only()
        elif (self.folder / AUTO_NAME).exists():
            self.import_auto(keep_manual=True, snapshot=False)

    def _load_flags_only(self):
        p = self.folder / AUTO_NAME
        if not p.exists():
            return
        data = json.loads(p.read_text())
        self.flags = []
        for side, arm in data.get("arms", {}).items():
            for f in arm.get("flags", []):
                self.flags.append(dict(f, side=side, t=self.ns_to_frame(f["t_ns"])))

    def ns_to_frame(self, ns: int) -> int:
        return int(np.clip(np.searchsorted(self.timestamps, ns), 0, self.T - 1))

    def import_auto(self, keep_manual=True, snapshot=True) -> bool:
        p = self.folder / AUTO_NAME
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        self.flags, events = [], []
        for side, arm in data.get("arms", {}).items():
            if side not in self.sides:
                continue
            if snapshot:
                self.snapshot(side)
            ph = np.zeros(self.T, np.int8)
            ob = [self.schema.null] * self.T
            go = [self.schema.null] * self.T
            covered = np.zeros(self.T, bool)
            for s in arm.get("segments", []):
                a = self.ns_to_frame(s["start_ns"])
                b = int(np.searchsorted(self.timestamps, s["end_ns"], side="right"))
                b = min(max(b, a + 1), self.T)
                ph[a:b] = self.schema.ids.get(s["phase"], 0)
                for t in range(a, b):
                    ob[t], go[t] = s["object"], s["goal"]
                covered[a:b] = True
            keep = (self.src[side] == SRC_MANUAL) if keep_manual else np.zeros(self.T, bool)
            for t in np.flatnonzero(~keep):
                self.phase[side][t] = ph[t]
                self.obj[side][t] = ob[t]
                self.goal[side][t] = go[t]
                self.src[side][t] = SRC_AUTO if covered[t] else SRC_DEFAULT
            for e in arm.get("events", []):
                events.append(dict(e, side=side))
            for f in arm.get("flags", []):
                self.flags.append(dict(f, side=side, t=self.ns_to_frame(f["t_ns"])))
        self.events = events
        self.loaded_from = "auto"
        return True

    def close(self):
        self.ep.close()

    def frame_image(self, t: int, cam: Optional[str] = None) -> "Image.Image":
        cam = cam or self.primary_cam
        t = max(0, min(self.T - 1, t))
        img = Image.fromarray(self.imgs[cam][t]).convert("RGB")
        if cam == self.primary_cam and self.intent is not None:
            draw = ImageDraw.Draw(img)
            raw = self.imgs[cam][t]
            try:
                gp = compute_gaze_px(self.intent, t, self.cfg, raw.shape, ep=self.ep)
                if gp is not None:
                    r = self.cfg.get("gaze", {}).get("dot_radius_px", 4)
                    draw.ellipse([gp[0] - r, gp[1] - r, gp[0] + r, gp[1] + r], fill=(220, 30, 30))
                _, boxes = compute_slot_boxes(self.intent, t, raw.shape, self.ep, self.names)
                for b in boxes:
                    draw.rectangle([b["u"] - 8, b["v"] - 8, b["u"] + 8, b["v"] + 8], outline=(255, 220, 0), width=1)
            except Exception:
                pass
        return img

    def key(self, side, t):
        return int(self.phase[side][t]), self.obj[side][t], self.goal[side][t]

    def boundaries(self, side) -> list[int]:
        keys = [self.key(side, t) for t in range(self.T)]
        return segment_bounds(keys)

    def snapshot(self, side):
        self._history.append((side, self.phase[side].copy(), list(self.obj[side]), list(self.goal[side]),
                              self.src[side].copy()))
        if len(self._history) > 50:
            self._history.pop(0)

    def undo(self) -> Optional[str]:
        if not self._history:
            return None
        side, ph, ob, go, src = self._history.pop()
        self.phase[side], self.obj[side], self.goal[side], self.src[side] = ph, ob, go, src
        return side

    def apply_range(self, side, lo, hi, phase_name, obj, goal):
        lo, hi = sorted((max(0, min(self.T - 1, lo)), max(0, min(self.T - 1, hi))))
        self.snapshot(side)
        self.phase[side][lo:hi + 1] = self.schema.ids[phase_name]
        for t in range(lo, hi + 1):
            self.obj[side][t] = obj
            self.goal[side][t] = goal
        self.src[side][lo:hi + 1] = SRC_MANUAL

    def shift_boundary(self, side, b, delta) -> int:
        nb = int(np.clip(b + delta, 1, self.T - 1))
        if nb == b:
            return b
        self.snapshot(side)
        if nb > b:
            ref = b - 1
            rng = range(b, nb)
        else:
            ref = b
            rng = range(nb, b)
        for t in rng:
            self.phase[side][t] = self.phase[side][ref]
            self.obj[side][t] = self.obj[side][ref]
            self.goal[side][t] = self.goal[side][ref]
            self.src[side][t] = SRC_MANUAL
        return nb

    def clear(self, side):
        self.snapshot(side)
        self.phase[side][:] = 0
        self.obj[side] = [self.schema.null] * self.T
        self.goal[side] = [self.schema.null] * self.T
        self.src[side][:] = SRC_MANUAL

    def save(self, reviewed: Optional[bool] = None):
        if reviewed is not None:
            self.reviewed = reviewed
        self.ep.close()
        with h5py.File(self.path, "r+") as ep:
            lbl = ep["labels"] if "labels" in ep else ep.create_group("labels")
            if "segments" in lbl:
                del lbl["segments"]
            for side in self.sides:
                for k in ("phase", "target_name", "goal_name", "source"):
                    n = f"arm_{side}_{k}"
                    if n in lbl:
                        del lbl[n]
                ds = lbl.create_dataset(f"arm_{side}_phase", data=self.phase[side])
                ds.attrs["legend"] = self.schema.legend()
                lbl.create_dataset(f"arm_{side}_target_name", data=np.array([n.encode() for n in self.obj[side]], dtype="S32"))
                lbl.create_dataset(f"arm_{side}_goal_name", data=np.array([n.encode() for n in self.goal[side]], dtype="S32"))
                ds = lbl.create_dataset(f"arm_{side}_source", data=self.src[side])
                ds.attrs["legend"] = "0=default 1=auto 2=manual"
            lbl.attrs["pass2_complete"] = True
            lbl.attrs["schema_version"] = self.schema.version
            lbl.attrs["task"] = self.schema.task
            lbl.attrs["reviewed"] = bool(self.reviewed)
            lbl.attrs["events"] = json.dumps(self.events)
            any_auto = any((self.src[s] == SRC_AUTO).any() for s in self.sides)
            any_manual = any((self.src[s] == SRC_MANUAL).any() for s in self.sides)
            lbl.attrs["label_source"] = "auto+manual" if any_auto and any_manual else ("auto" if any_auto else "manual")
        self.ep = h5py.File(self.path, "r")
        self._bind()


# ---------------------------------------------------------------------------

class ContextPlots:
    def __init__(self, parent: tk.Widget, schema: Schema):
        self.schema = schema
        self.fig = Figure(figsize=(7.6, 6.4), dpi=92)
        gs = self.fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1.1, 0.45, 0.45, 0.25])
        self.ax_speed = self.fig.add_subplot(gs[0])
        self.ax_force = self.fig.add_subplot(gs[1], sharex=self.ax_speed)
        self.ax_grip = self.fig.add_subplot(gs[2], sharex=self.ax_speed)
        self.ax_goal = self.fig.add_subplot(gs[3], sharex=self.ax_speed)
        self.ax_phase = self.fig.add_subplot(gs[4], sharex=self.ax_speed)
        self.ax_obj = self.fig.add_subplot(gs[5], sharex=self.ax_speed)
        self.ax_src = self.fig.add_subplot(gs[6], sharex=self.ax_speed)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.cursors = []
        self.bmarks = []
        self._t = 0
        self.side = "left"
        self.obj_color = {}
        self.session = None
        self.follow = True
        self.on_seek = None
        self._dragging = False
        self._drag_x = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._texts = []

    @staticmethod
    def _style(ax, title):
        ax.cla()
        ax.set_title(title, fontsize=9, loc="left", pad=2)
        ax.tick_params(labelsize=7)

    def set_session(self, s: EpisodeSession, side: str):
        self.side = side
        self.session = s
        x = np.arange(s.T)
        self.cursors = []
        self._style(self.ax_speed, "EE speed")
        self.ax_speed.callbacks.connect("xlim_changed", self._on_xlim)
        for sd in s.sides:
            self.ax_speed.plot(x, s.speed[sd], lw=0.7, label=sd)
        self._style(self.ax_force, "contact force |F_ext|")
        for sd in s.sides:
            self.ax_force.plot(x, s.force[sd], lw=0.7)
        self._style(self.ax_grip, "gripper (solid width, dashed cmd)")
        for i, sd in enumerate(s.sides):
            self.ax_grip.plot(x, s.gwidth[sd], lw=0.9, color=f"C{i}")
            self.ax_grip.plot(x, s.gcmd[sd], lw=0.9, color=f"C{i}", ls="--")
        self._style(self.ax_goal, "part → nearest goal [m]")
        names = list(s.obj_names)
        self.obj_color = {n: OBJ_COLORS[i % len(OBJ_COLORS)] for i, n in enumerate(names)}
        for n, d in s.goal_dist.items():
            self.ax_goal.plot(x, np.minimum(d, 0.3), lw=0.8, color=self.obj_color.get(n), label=n)
        if s.goal_dist:
            self.ax_goal.legend(fontsize=6, ncol=4, loc="upper right")
        self.ax_goal.set_ylim(-0.01, 0.31)
        self.ax_goal.set_title("part → nearest goal [m]   |  green seated, red miss, magenta push, "
                               "blue dashed insert attempt #n, orange band retract", fontsize=9, loc="left", pad=2)
        for e in s.events:
            typ = e.get("type")
            if typ in ("seated", "grasp_miss", "push"):
                c = {"seated": "g", "grasp_miss": "r", "push": "m"}[typ]
                self.ax_goal.axvline(s.ns_to_frame(e["t_ns"]), color=c, lw=0.6, alpha=0.6)
            elif typ == "retract" and "t_end_ns" in e:
                self.ax_goal.axvspan(s.ns_to_frame(e["t_ns"]), s.ns_to_frame(e["t_end_ns"]), color="orange",
                                     alpha=0.35, lw=0)
            elif typ == "insert_attempt":
                fr = s.ns_to_frame(e["t_ns"])
                self.ax_goal.axvline(fr, color="b", lw=0.8, ls="--", alpha=0.7)
                self.ax_goal.text(fr, 0.29, f" #{e.get('attempt', '')}", fontsize=7, color="b", va="top", clip_on=True)
        self.update_labels(s, redraw=False)
        self.fig.tight_layout(h_pad=0.3)
        self.canvas.draw()

    def _ribbon(self, ax, rgb, title):
        self._style(ax, title)
        ax.imshow(rgb[None], aspect="auto", extent=[0, rgb.shape[0], 0, 1], interpolation="nearest")
        ax.set_yticks([])

    @staticmethod
    def short(name: str) -> str:
        return name.replace("spare_", "s_")

    def _legend(self, ax, items):
        handles = [Patch(facecolor=c, edgecolor="#666", lw=0.4, label=n) for n, c in items]
        ax.legend(handles=handles, ncol=len(handles), fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  frameon=False, borderaxespad=0.1, handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    def _segment_texts(self):
        for txt in self._texts:
            try:
                txt.remove()
            except (ValueError, NotImplementedError):
                pass
        self._texts = []
        s = self.session
        if s is None:
            return
        lo, hi = self.ax_speed.get_xlim()
        span = max(1.0, hi - lo)
        side = self.side
        rows = ((self.ax_phase, [s.schema.names[int(p)] for p in s.phase[side]]),
                (self.ax_obj, list(s.obj[side])))
        for ax, keys in rows:
            a = 0
            for t in range(1, s.T + 1):
                if t == s.T or keys[t] != keys[a]:
                    l, r = max(a, lo), min(t, hi)
                    if r - l > 0.045 * span and keys[a] not in (s.schema.null, "IDLE"):
                        self._texts.append(ax.text((l + r) / 2, 0.5, self.short(keys[a]), ha="center", va="center",
                                                   fontsize=7, clip_on=True))
                    a = t

    def _on_xlim(self, _ax):
        if self.session is not None and not getattr(self, "_in_update", False):
            self._segment_texts()

    def update_labels(self, s: EpisodeSession, redraw=True):
        self._in_update = True
        xlim = self.ax_speed.get_xlim()
        side = self.side
        ph_rgb = np.array([to_rgb(self.schema.colors.get(int(p), "#000000")) for p in s.phase[side]])
        self._ribbon(self.ax_phase, ph_rgb, f"phase ({side})")
        self._legend(self.ax_phase, [(n, self.schema.colors[self.schema.ids[n]]) for n in self.schema.task_phases])
        ob_rgb = np.array([to_rgb(self.obj_color.get(o, "#ffffff")) if o != self.schema.null else (1, 1, 1)
                           for o in s.obj[side]])
        self._ribbon(self.ax_obj, ob_rgb, "object")
        self._legend(self.ax_obj, [(self.short(n), c) for n, c in self.obj_color.items()])
        src_rgb = np.array([(0.85, 0.85, 0.85), (0.55, 0.75, 0.95), (0.95, 0.45, 0.2)])[s.src[side]]
        self._ribbon(self.ax_src, src_rgb, "source: grey default, blue auto, orange manual")
        for ax in (self.ax_phase, self.ax_obj):
            for b in s.boundaries(side):
                ax.axvline(b, color="k", lw=0.4)
        for c in self.cursors:
            try:
                c.remove()
            except (ValueError, NotImplementedError):
                pass
        self.cursors = [ax.axvline(self._t, color="black", lw=1, ls="--")
                        for ax in (self.ax_speed, self.ax_force, self.ax_grip, self.ax_goal,
                                   self.ax_phase, self.ax_obj, self.ax_src)]
        if xlim != (0.0, 1.0):
            self.ax_speed.set_xlim(xlim)
        else:
            self.ax_speed.set_xlim(0, s.T)
        self._in_update = False
        self._texts = []
        self._segment_texts()
        if redraw:
            self.canvas.draw_idle()

    def _axes(self):
        return (self.ax_speed, self.ax_force, self.ax_grip, self.ax_goal, self.ax_phase, self.ax_obj, self.ax_src)

    def _x_from_event(self, ev):
        if ev.x is None:
            return None
        return float(self.ax_speed.transData.inverted().transform((ev.x, 0))[0])

    def _seek_to(self, x):
        if self.on_seek is None:
            return
        t = int(round(min(max(x, 0), self._total() - 1)))
        if t != self._t:
            self.on_seek(t)

    def _edge_shift(self, x):
        lo, hi = self.ax_speed.get_xlim()
        span = hi - lo
        margin = 0.06 * span
        if x > hi - margin and hi < self._total():
            depth = min(3.0, (x - (hi - margin)) / margin)
            return max(1.0, 0.012 * span * depth)
        if x < lo + margin and lo > 0:
            depth = min(3.0, ((lo + margin) - x) / margin)
            return -max(1.0, 0.012 * span * depth)
        return 0.0

    def _on_press(self, ev):
        if ev.button != 1 or self.toolbar.mode or ev.inaxes not in self._axes():
            return
        self._dragging = True
        self._follow_saved = self.follow
        self.follow = False
        self._drag_x = self._x_from_event(ev)
        self._seek_to(self._drag_x)
        if not getattr(self, "_ticking", False):
            self._edge_tick()

    def _on_motion(self, ev):
        if not self._dragging:
            return
        x = self._x_from_event(ev)
        if x is None:
            return
        self._drag_x = x
        lo, hi = self.ax_speed.get_xlim()
        self._seek_to(min(max(x, lo), hi))

    def _edge_tick(self):
        if not self._dragging or self._drag_x is None:
            self._ticking = False
            return
        self._ticking = True
        shift = self._edge_shift(self._drag_x)
        if shift:
            lo, hi = self.ax_speed.get_xlim()
            self._set_window(lo + shift, hi - lo)
            self._drag_x += shift
            nlo, nhi = self.ax_speed.get_xlim()
            self._seek_to(min(max(self._drag_x, nlo), nhi))
            self.canvas.draw_idle()
        self.canvas.get_tk_widget().after(40, self._edge_tick)

    def _on_release(self, ev):
        if self._dragging:
            self._dragging = False
            self.follow = self._follow_saved

    def _total(self):
        return self.session.T if self.session is not None else 1

    def _set_window(self, lo, span):
        T = self._total()
        span = min(max(span, 30), T)
        lo = min(max(lo, 0), T - span)
        self.ax_speed.set_xlim(lo, lo + span)

    def move_cursor(self, t):
        old = self._t
        self._t = t
        if self.follow and self.session is not None:
            lo, hi = self.ax_speed.get_xlim()
            span = hi - lo
            if span < 0.98 * self._total():
                rel = (old - lo) / span if lo <= old <= hi else 0.5
                nlo = t - rel * span
                if abs(nlo - lo) > 1e-6:
                    self._set_window(nlo, span)
        for c in self.cursors:
            c.set_xdata([t, t])
        self.canvas.draw_idle()

    def zoom(self, factor):
        if self.session is None:
            return
        lo, hi = self.ax_speed.get_xlim()
        span = hi - lo
        rel = (self._t - lo) / span if lo <= self._t <= hi else 0.5
        nspan = span * factor
        self._set_window(self._t - rel * nspan, nspan)
        self.canvas.draw_idle()

    def full_view(self):
        if self.session is None:
            return
        self.ax_speed.set_xlim(0, self._total())
        self.canvas.draw_idle()


class ArmPanel(ttk.LabelFrame):
    def __init__(self, parent, side, app: "LabelToolApp"):
        super().__init__(parent, text=f"{side} arm", padding=6)
        self.side, self.app = side, app
        sch = app.schema
        self.phase_var = tk.StringVar(value=sch.task_phases[0])
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 3))
        for n in sch.task_phases:
            tk.Label(row, text="  ", bg=sch.all[n]["color"], relief="solid", bd=1).pack(side="left", padx=(8, 2))
            ttk.Radiobutton(row, text=f"{sch.all[n].get('key', '')} {n}", value=n, variable=self.phase_var).pack(side="left")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(0, 3))
        ttk.Label(row, text="object").pack(side="left")
        self.obj_var = tk.StringVar(value=sch.null)
        self.obj_combo = ttk.Combobox(row, textvariable=self.obj_var, state="readonly", width=16)
        self.obj_combo.pack(side="left", padx=4)
        self.obj_swatch = tk.Label(row, text="  ", bg="white", relief="solid", bd=1)
        self.obj_swatch.pack(side="left", padx=(0, 8))
        self.obj_var.trace_add("write", lambda *_: self._update_swatch())
        ttk.Label(row, text="goal").pack(side="left")
        self.goal_var = tk.StringVar(value=sch.null)
        self.goal_combo = ttk.Combobox(row, textvariable=self.goal_var, state="readonly", width=12)
        self.goal_combo.pack(side="left", padx=4)
        ttk.Button(row, text="pick from frame", command=self.pick_from_frame).pack(side="left", padx=4)
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Button(row, text="Start segment here", command=self.mark_start).pack(side="left")
        ttk.Button(row, text="Label to here ⏎", command=self.apply_range).pack(side="left", padx=4)
        self.start_var = tk.StringVar(value="from: auto")
        ttk.Label(row, textvariable=self.start_var, width=14).pack(side="left")
        ttk.Button(row, text="This frame", command=self.set_here).pack(side="left")
        ttk.Button(row, text="Clear arm", command=self.clear).pack(side="left", padx=(16, 0))
        self.readout = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.readout, foreground="#444").pack(anchor="w", pady=(3, 0))
        self._start: Optional[int] = None

    def _update_swatch(self):
        self.obj_swatch.configure(bg=self.app.plots.obj_color.get(self.obj_var.get(), "white"))

    def set_options(self, s: EpisodeSession):
        null = self.app.schema.null
        self.obj_combo["values"] = [null] + s.obj_names
        self.goal_combo["values"] = [null] + s.goal_names

    def pick_from_frame(self):
        s, t = self.app.session, self.app.current_t
        if s is None:
            return
        self.phase_var.set(s.schema.names[int(s.phase[self.side][t])])
        self.obj_var.set(s.obj[self.side][t])
        self.goal_var.set(s.goal[self.side][t])

    def mark_start(self):
        if self.app.session is None:
            return
        self._start = self.app.current_t
        self.start_var.set(f"from: {self._start}")

    def _auto_start(self) -> int:
        s, t = self.app.session, self.app.current_t
        bs = [b for b in s.boundaries(self.side) if b <= t]
        return bs[-1] if bs else 0

    def apply_range(self):
        s = self.app.session
        if s is None:
            return
        lo = self._start if self._start is not None else self._auto_start()
        s.apply_range(self.side, lo, self.app.current_t, self.phase_var.get(), self.obj_var.get(), self.goal_var.get())
        self._start = None
        self.start_var.set("from: auto")
        self.app.on_labels_changed()

    def set_here(self):
        s, t = self.app.session, self.app.current_t
        if s is None:
            return
        s.apply_range(self.side, t, t, self.phase_var.get(), self.obj_var.get(), self.goal_var.get())
        self.app.on_labels_changed()

    def clear(self):
        if self.app.session is None:
            return
        if messagebox.askyesno("Clear", f"Clear all {self.side}-arm labels? (undo works right after)"):
            self.app.session.clear(self.side)
            self.app.on_labels_changed()

    def update_readout(self, s: EpisodeSession, t: int):
        src = ["default", "auto", "manual"][int(s.src[self.side][t])]
        self.readout.set(f"frame {t}: {s.schema.names[int(s.phase[self.side][t])]} / "
                         f"{s.obj[self.side][t]} → {s.goal[self.side][t]}   [{src}]")


class LabelToolApp:
    def __init__(self, root: tk.Tk, cfg: dict, store_root: Path, schema: Schema, task_path: Path,
                 initial: Optional[str]):
        self.root, self.cfg, self.store_root, self.schema, self.task_path = root, cfg, store_root, schema, task_path
        self.session: Optional[EpisodeSession] = None
        self.current_t = 0
        self.playing = False
        self.active_side = "left"
        self._photo = None
        self._photo_extra = []

        root.title(f"teleop-intent labeling — {schema.task}")
        root.geometry("1800x1060")

        top = ttk.Frame(root, padding=6)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="episode").pack(side="left")
        self.ep_ids = discover_episodes(store_root)
        self.ep_var = tk.StringVar()
        self.ep_combo = ttk.Combobox(top, textvariable=self.ep_var, state="readonly", width=30)
        self.ep_combo.pack(side="left", padx=4)
        self.refresh_episode_list(initial)
        ttk.Button(top, text="Load", command=self.load_episode).pack(side="left")
        ttk.Button(top, text="◀ prev ep", command=lambda: self.step_episode(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="next ep ▶", command=lambda: self.step_episode(1)).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Run auto-labeler", command=self.run_auto).pack(side="left")
        ttk.Button(top, text="Import auto (keep manual)", command=lambda: self.import_auto(True)).pack(side="left", padx=4)
        ttk.Button(top, text="Reset to auto", command=lambda: self.import_auto(False)).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Save", command=self.save).pack(side="left")
        ttk.Button(top, text="Save + mark reviewed", command=lambda: self.save(True)).pack(side="left", padx=4)
        self.status = tk.StringVar(value="no episode loaded")
        ttk.Label(top, textvariable=self.status, foreground="#555").pack(side="left", padx=8)

        ttk.Label(root, text="←/→ 1 frame  shift ×10  space play  0-9 phase  ⏎ label to here  "
                             "n/p boundary  [ ] move boundary  { } ×10  f next flag  o part names  =/- zoom  z full  tab arm  click/drag plots to scrub  ctrl+s save  ctrl+z undo",
                  foreground="#888").pack(side="bottom", pady=3)

        panels = ttk.Frame(root, padding=6)
        panels.pack(side="bottom", fill="x")
        self.panel_frame = panels
        self.arm_panels: dict[str, ArmPanel] = {}

        scrub = ttk.Frame(root, padding=6)
        scrub.pack(side="bottom", fill="x")
        self.frame_var = tk.IntVar(value=0)
        self.scale = ttk.Scale(scrub, from_=0, to=1, orient="horizontal", variable=self.frame_var,
                               command=lambda v: self.goto(int(float(v))))
        self.scale.pack(fill="x")
        ctrl = ttk.Frame(scrub)
        ctrl.pack(fill="x", pady=3)
        ttk.Button(ctrl, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left")
        self.play_btn = ttk.Button(ctrl, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(ctrl, text="▶", width=3, command=lambda: self.step(1)).pack(side="left")
        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="follow cursor", variable=self.follow_var,
                        command=lambda: setattr(self.plots, "follow", self.follow_var.get())).pack(side="left", padx=(10, 0))
        ttk.Button(ctrl, text="zoom + (=)", command=lambda: self.plots.zoom(0.5)).pack(side="left", padx=(4, 0))
        ttk.Button(ctrl, text="zoom − (-)", command=lambda: self.plots.zoom(2.0)).pack(side="left")
        ttk.Button(ctrl, text="full (z)", command=lambda: self.plots.full_view()).pack(side="left")
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="part names (o)", variable=self.overlay_var, command=self.refresh).pack(side="left", padx=(10, 0))
        ttk.Label(ctrl, text="speed").pack(side="left", padx=(10, 2))
        self.speed_var = tk.StringVar(value="1x")
        ttk.Combobox(ctrl, textvariable=self.speed_var, values=["0.25x", "0.5x", "1x", "2x", "4x"], width=5,
                     state="readonly").pack(side="left")
        self.time_var = tk.StringVar(value="frame 0 / 0")
        ttk.Label(ctrl, textvariable=self.time_var).pack(side="left", padx=10)
        ttk.Button(ctrl, text="◀ boundary (p)", command=lambda: self.jump_boundary(-1)).pack(side="left", padx=(20, 0))
        ttk.Button(ctrl, text="boundary ▶ (n)", command=lambda: self.jump_boundary(1)).pack(side="left")
        ttk.Button(ctrl, text="move ◀ ([)", command=lambda: self.move_boundary(-1)).pack(side="left", padx=(10, 0))
        ttk.Button(ctrl, text="move ▶ (])", command=lambda: self.move_boundary(1)).pack(side="left")
        ttk.Button(ctrl, text="next flag (f)", command=self.next_flag).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Undo", command=self.undo).pack(side="right")

        content = ttk.Frame(root, padding=6)
        content.pack(side="top", fill="both", expand=True)
        left = ttk.Frame(content)
        left.pack(side="left", fill="y")
        self.video = ttk.Label(left)
        self.video.pack()
        self.extra_row = ttk.Frame(left)
        self.extra_row.pack(pady=(4, 0))
        ttk.Label(left, text="flags (double-click to jump)").pack(anchor="w", pady=(6, 0))
        self.flag_list = tk.Listbox(left, height=9, width=60, font=("Courier", 9))
        self.flag_list.pack(fill="x")
        self.flag_list.bind("<Double-Button-1>", self._on_flag)
        plots = ttk.Frame(content)
        plots.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.plots = ContextPlots(plots, schema)
        self.plots.on_seek = self.goto

        for k, f in {"<Left>": lambda e: self.step(-1), "<Right>": lambda e: self.step(1),
                     "<Shift-Left>": lambda e: self.step(-10), "<Shift-Right>": lambda e: self.step(10),
                     "<space>": lambda e: self.toggle_play(), "<Control-s>": lambda e: self.save(),
                     "<Control-z>": lambda e: self.undo(), "<Return>": lambda e: self._panel().apply_range(),
                     "n": lambda e: self.jump_boundary(1), "p": lambda e: self.jump_boundary(-1),
                     "f": lambda e: self.next_flag(),
                     "<equal>": lambda e: self.plots.zoom(0.5), "<plus>": lambda e: self.plots.zoom(0.5),
                     "<minus>": lambda e: self.plots.zoom(2.0), "z": lambda e: self.plots.full_view(),
                     "o": lambda e: (self.overlay_var.set(not self.overlay_var.get()), self.refresh()), "<Tab>": lambda e: self.toggle_arm()}.items():
            root.bind(k, f)
        root.bind("[", lambda e: self.move_boundary(-1))
        root.bind("]", lambda e: self.move_boundary(1))
        root.bind("{", lambda e: self.move_boundary(-10))
        root.bind("}", lambda e: self.move_boundary(10))
        for n in schema.task_phases:
            key = str(schema.all[n].get("key", ""))
            if key:
                root.bind(key, lambda e, n=n: self._panel().phase_var.set(n))

        if self.ep_var.get():
            root.after(200, self.load_episode)

    def refresh_episode_list(self, select: Optional[str] = None):
        labels = [episode_status(self.store_root, e) for e in self.ep_ids]
        self.ep_combo["values"] = labels
        cur = select or (self.ep_var.get().split()[0] if self.ep_var.get() else None)
        for l in labels:
            if cur and l.split()[0] == cur:
                self.ep_var.set(l)
                return
        if labels and not self.ep_var.get():
            self.ep_var.set(labels[0])

    def _panel(self) -> ArmPanel:
        return self.arm_panels[self.active_side]

    def _eid(self) -> str:
        return self.ep_var.get().split()[0]

    def step_episode(self, d):
        if not self.ep_ids:
            return
        if self.session is not None and not self._confirm_leave():
            return
        i = self.ep_ids.index(self._eid()) if self._eid() in self.ep_ids else 0
        i = max(0, min(len(self.ep_ids) - 1, i + d))
        self.refresh_episode_list(self.ep_ids[i])
        self.load_episode(skip_confirm=True)

    def _confirm_leave(self) -> bool:
        if self.session is not None and self.session._history:
            r = messagebox.askyesnocancel("Unsaved", "Save changes before switching?")
            if r is None:
                return False
            if r:
                self.save()
        return True

    def load_episode(self, skip_confirm=False):
        if not self.ep_var.get():
            return
        if not skip_confirm and self.session is not None and not self._confirm_leave():
            return
        path = self.store_root / self._eid() / "episode.hdf5"
        if not path.exists():
            messagebox.showerror("Not found", str(path))
            return
        if self.session is not None:
            self.session.close()
        self.status.set(f"loading {self._eid()} …")
        self.root.update_idletasks()
        self.session = EpisodeSession(path, self.cfg, self.schema)
        s = self.session
        for p in self.arm_panels.values():
            p.destroy()
        self.arm_panels = {sd: ArmPanel(self.panel_frame, sd, self) for sd in s.sides}
        for p in self.arm_panels.values():
            p.pack(side="left", fill="x", expand=True, padx=4)
            p.set_options(s)
        self.active_side = s.sides[0]
        self._mark_active()
        for w in self.extra_row.winfo_children():
            w.destroy()
        self._extra_labels = []
        for c in s.extra_cams:
            lab = ttk.Label(self.extra_row)
            lab.pack(side="left", padx=2)
            self._extra_labels.append((c, lab))
        self.plots.ax_speed.set_xlim(0, 1)
        self.plots.set_session(s, self.active_side)
        for p in self.arm_panels.values():
            p._update_swatch()
        self._fill_flags()
        self.scale.configure(to=max(1, s.T - 1))
        self.current_t = 0
        self.frame_var.set(0)
        self.status.set(f"{self._eid()}  {s.T} frames  labels from {s.loaded_from}"
                        f"{'  REVIEWED' if s.reviewed else ''}")
        self.refresh()

    def _fill_flags(self):
        self.flag_list.delete(0, "end")
        s = self.session
        if s is None:
            return
        self._flag_frames = []
        for f in sorted(s.flags, key=lambda f: f["t"]):
            self.flag_list.insert("end", f"{f['t']:6d}  {f.get('side', ''):5s} {f['kind']:22s} {f.get('object', '')}")
            self._flag_frames.append(f["t"])

    def _on_flag(self, _e):
        sel = self.flag_list.curselection()
        if sel:
            self.goto(self._flag_frames[sel[0]])

    def next_flag(self):
        if self.session is None or not getattr(self, "_flag_frames", None):
            return
        nxt = [t for t in self._flag_frames if t > self.current_t]
        t = nxt[0] if nxt else self._flag_frames[0]
        self.goto(t)
        i = self._flag_frames.index(t)
        self.flag_list.selection_clear(0, "end")
        self.flag_list.selection_set(i)
        self.flag_list.see(i)

    def toggle_arm(self):
        if self.session is None or len(self.session.sides) < 2:
            return "break"
        sides = self.session.sides
        self.active_side = sides[(sides.index(self.active_side) + 1) % len(sides)]
        self._mark_active()
        self.plots.side = self.active_side
        self.plots.update_labels(self.session)
        return "break"

    def _mark_active(self):
        for sd, p in self.arm_panels.items():
            p.configure(text=f"{sd} arm{'  ◀ active' if sd == self.active_side else ''}")

    def run_auto(self):
        if self.session is None:
            return
        folder = self.session.folder
        if not (folder / "scene.csv").exists():
            messagebox.showerror("Auto-labeler", f"No raw logs (scene.csv) in {folder}")
            return
        self.status.set("running auto-labeler …")
        self.root.update_idletasks()
        r = subprocess.run([sys.executable, str(HERE / "auto_label.py"), str(folder), "--task", str(self.task_path)],
                           capture_output=True, text=True)
        if r.returncode != 0 or "[error]" in r.stderr:
            messagebox.showerror("Auto-labeler", r.stderr[-2000:] or r.stdout[-2000:])
            return
        self.import_auto(True)

    def import_auto(self, keep_manual: bool):
        s = self.session
        if s is None:
            return
        if not keep_manual and not messagebox.askyesno("Reset", "Overwrite ALL labels (including manual) with auto labels?"):
            return
        if not s.import_auto(keep_manual=keep_manual):
            messagebox.showinfo("Auto labels", f"No {AUTO_NAME} in {s.folder}. Run the auto-labeler first.")
            return
        self.plots.set_session(s, self.active_side)
        self._fill_flags()
        self.status.set(f"imported auto labels ({'manual kept' if keep_manual else 'reset'}) — not saved yet")
        self.refresh()

    def save(self, reviewed: Optional[bool] = None):
        if self.session is None:
            return
        self.session.save(reviewed)
        self.session._history.clear()
        self.status.set(f"saved {self._eid()}{'  REVIEWED' if self.session.reviewed else ''}")
        self.refresh_episode_list(self._eid())

    def undo(self):
        if self.session is None:
            return
        if self.session.undo():
            self.on_labels_changed()

    def on_labels_changed(self):
        self.plots.update_labels(self.session)
        self.refresh()

    def jump_boundary(self, d):
        s = self.session
        if s is None:
            return
        bs = s.boundaries(self.active_side)
        if not bs:
            return
        if d > 0:
            nxt = [b for b in bs if b > self.current_t]
            self.goto(nxt[0] if nxt else bs[-1])
        else:
            prv = [b for b in bs if b < self.current_t]
            self.goto(prv[-1] if prv else bs[0])

    def move_boundary(self, delta):
        s = self.session
        if s is None:
            return
        bs = s.boundaries(self.active_side)
        if not bs:
            return
        b = min(bs, key=lambda x: abs(x - self.current_t))
        if abs(b - self.current_t) > 2:
            self.status.set("move the cursor onto a boundary first (n/p)")
            return
        nb = s.shift_boundary(self.active_side, b, delta)
        self.current_t = nb
        self.frame_var.set(nb)
        self.on_labels_changed()

    def step(self, d):
        self.goto(self.current_t + d)

    def goto(self, t):
        if self.session is None:
            return
        t = max(0, min(self.session.T - 1, int(t)))
        self.current_t = t
        self.frame_var.set(t)
        self.refresh()

    def _font(self, size):
        key = f"_font_{size}"
        if not hasattr(self, key):
            try:
                f = ImageFont.load_default(size=size)
            except TypeError:
                f = ImageFont.load_default()
            setattr(self, key, f)
        return getattr(self, key)

    def _draw_parts(self, img, t):
        s = self.session
        if s.cam is None or not s.obj_pos:
            return
        d = ImageDraw.Draw(img)
        side = self.active_side
        cur_obj, cur_goal = s.obj[side][t], s.goal[side][t]
        pts = np.array([p[t] for p in s.obj_pos]) + np.array([0, 0, 0.03])
        uv = s.project(pts, DISPLAY_SCALE)
        for (u, v), n in zip(uv, s.obj_names):
            if not np.isfinite(u):
                continue
            col = self.plots.obj_color.get(n, "#ffffff")
            hit = n == cur_obj
            r = 9 if hit else 5
            d.ellipse([u - r, v - r, u + r, v + r], outline="white" if hit else col, width=3 if hit else 2)
            txt = ContextPlots.short(n)
            f = self._font(16 if hit else 13)
            bb = d.textbbox((u + r + 3, v - 8), txt, font=f)
            d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(0, 0, 0))
            d.text((u + r + 3, v - 8), txt, fill="white" if hit else col, font=f)
        if cur_goal in s.schema.goal_pos:
            g = s.project(s.schema.goal_pos[cur_goal], DISPLAY_SCALE)[0]
            if np.isfinite(g[0]):
                u, v = g
                d.line([u - 10, v, u + 10, v], fill=(0, 230, 255), width=2)
                d.line([u, v - 10, u, v + 10], fill=(0, 230, 255), width=2)
                f = self._font(14)
                bb = d.textbbox((u - 10, v + 12), f"goal: {cur_goal}", font=f)
                d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(0, 0, 0))
                d.text((u - 10, v + 12), f"goal: {cur_goal}", fill=(0, 230, 255), font=f)

    def refresh(self):
        s = self.session
        if s is None:
            return
        t = self.current_t
        img = s.frame_image(t)
        img = img.resize((int(img.width * DISPLAY_SCALE), int(img.height * DISPLAY_SCALE)), Image.NEAREST)
        if self.overlay_var.get():
            self._draw_parts(img, t)
        self._photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self._photo)
        self._photo_extra = []
        for cam, lab in getattr(self, "_extra_labels", []):
            ph = ImageTk.PhotoImage(s.frame_image(t, cam))
            self._photo_extra.append(ph)
            lab.configure(image=ph)
        self.plots.move_cursor(t)
        ts = (int(s.timestamps[t]) - int(s.timestamps[0])) / 1e9
        self.time_var.set(f"frame {t} / {s.T - 1}   ({ts:.2f}s)")
        for sd, p in self.arm_panels.items():
            p.update_readout(s, t)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._tick()

    def _tick(self):
        s = self.session
        if not self.playing or s is None:
            return
        if self.current_t >= s.T - 1:
            self.toggle_play()
            return
        mult = float(self.speed_var.get().rstrip("x"))
        dt_ms = float(np.median(np.diff(s.timestamps[:100]))) / 1e6 if s.T > 2 else 33.0
        stride = max(1, int(round(mult))) if mult >= 1 else 1
        self.step(stride)
        self.root.after(int(dt_ms * stride / mult), self._tick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "playback.yaml"))
    ap.add_argument("--task", default=str(DEFAULT_TASK))
    ap.add_argument("--store-root", default=None)
    ap.add_argument("--episode", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = args.store_root or cfg.get("data", {}).get("store_root")
    if not store:
        ds = ROOT / "configs" / "dataset.yaml"
        store = load_config(str(ds))["data"]["store_root"] if ds.exists() else None
    if not store:
        sys.exit("pass --store-root or set data.store_root")
    schema = Schema(SCHEMA_PATH, Path(args.task))
    root = tk.Tk()
    LabelToolApp(root, cfg, Path(store), schema, Path(args.task), args.episode)
    root.mainloop()


if __name__ == "__main__":
    main()

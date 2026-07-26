"""
labeling/label_tool.py — standalone manual labeling tool (replaces segment.py).

segment.py inferred phase (IDLE/APPROACH/GRASP/TRANSPORT/PLACE) and target
object per arm from gripper-width + object-lift heuristics. That's gone now:
this tool lets a human watch the episode and mark it by eye instead.

This is deliberately NOT built on Rerun. An earlier version tried to pair a
Tkinter control panel with a spawned Rerun viewer, but Rerun has no supported
way for an external process to read back the viewer's current playhead or
reliably force it to an arbitrary point — the two windows could drift apart
with no way to reconcile them. For a tight scrub-and-mark labeling loop,
that's a fragile foundation.

Instead this is one self-contained window: a single process draws the video
frame (with gaze dot + slot boxes burned in via PIL) and small context plots
(EE speed, contact force, gripper width/cmd, EE attention, plus your own
labeled-phase/labeled-target progress) with a moving cursor line, all from
the same scrub position. There is nothing to keep in sync because there is
only one source of truth. viz/playback.py (Rerun) remains the tool for rich
multi-view review; this tool is only for producing labels.

Labels are written back to the episode HDF5 in the same schema segment.py
used to write (labels/arm_{side}_phase, labels/arm_{side}_target_name), so
nothing downstream (recompute_intent_belief.py, viz/playback.py) needs to
change.

Dependencies: h5py, numpy, pyyaml, Pillow, matplotlib (all pip-installable;
no rerun-sdk needed).

Usage:
    python labeling/label_tool.py
    python labeling/label_tool.py --episode 076
    python labeling/label_tool.py --config configs/playback.yaml
"""

from __future__ import annotations

import argparse
import csv
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
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    PHASE_IDS,
    PHASE_NAMES,
    candidate_names_from_hdf5,
    compute_ee_speed,
    compute_gaze_px,
    compute_slot_boxes,
    load_config,
)

SIDES = ("left", "right")
DISPLAY_SCALE = 2.8  # upscale the (small, downsampled) stored frame for visibility


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def resolve_store_root(cfg: dict) -> str:
    """Returns the episode store root, falling back to configs/dataset.yaml."""
    store_root = cfg["data"].get("store_root")
    if store_root:
        return store_root
    dataset_cfg_path = ROOT / "configs" / "dataset.yaml"
    if dataset_cfg_path.exists():
        dataset_cfg = load_config(str(dataset_cfg_path))
        return dataset_cfg["data"]["store_root"]
    sys.exit("data.store_root must be set in configs/playback.yaml or configs/dataset.yaml")


def load_episode_list() -> list[str]:
    """Returns episode ids from data/manifest.csv, or [] if it's missing."""
    manifest = ROOT / "data" / "manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["episode_id"] for r in rows if r.get("episode_id")]


def episode_label(episode_id: str) -> str:
    """Returns a display string for the episode picker, e.g. '076  success  56s'."""
    manifest = ROOT / "data" / "manifest.csv"
    if manifest.exists():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("episode_id") == episode_id:
                    dur = r.get("duration_s", "")
                    dur = f"{float(dur):.0f}s" if dur else ""
                    return f"{episode_id}   {r.get('success', ''):<8} {dur}"
    return episode_id


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def list_target_names(ep: h5py.File) -> list[str]:
    """Returns object + bin names from the scene/ group, in scene order.

    The loop bound must check the *pose* dataset, not the *name* dataset —
    some episodes have bin{i}_pose without a matching bin{i}_name (bins are
    only named implicitly via the color_bin_mapping attr), so checking the
    name key would silently drop bins from the target list entirely.
    """
    if "scene" not in ep:
        return []
    sc = ep["scene"]
    names = []
    for i in range(20):
        if f"obj{i}_pose" not in sc:
            break
        key = f"obj{i}_name"
        names.append((_decode(sc[key][()]).rstrip("\x00").strip() if key in sc else "") or f"object_{i + 1}")
    for i in range(20):
        if f"bin{i}_pose" not in sc:
            break
        key = f"bin{i}_name"
        names.append((_decode(sc[key][()]).rstrip("\x00").strip() if key in sc else "") or f"bin_{i + 1}")
    return names


# ---------------------------------------------------------------------------
# Frame rendering (video canvas)
# ---------------------------------------------------------------------------

def render_frame(raw_frame: np.ndarray, intent, t: int, cfg: dict, ep: h5py.File,
                 names: list[str], scale: int = DISPLAY_SCALE) -> "Image.Image":
    """Returns a PIL Image of the frame with gaze dot + slot boxes burned in,
    upscaled for visibility. Uses the same pixel math as viz/playback.py
    (common.compute_gaze_px / compute_slot_boxes) so the overlay is identical
    to what you'd see in the Rerun review tool."""
    img = Image.fromarray(raw_frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    if intent is not None:
        gp = compute_gaze_px(intent, t, cfg, raw_frame.shape)
        if gp is not None:
            gx, gy = gp
            r = cfg["gaze"].get("dot_radius_px", 4)
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(220, 30, 30))

        ee_boxes, tgt_boxes = compute_slot_boxes(intent, t, raw_frame.shape, ep, names)
        half = 10

        if tgt_boxes:
            max_b = max(b["belief"] for b in tgt_boxes) or 1.0
            for b in tgt_boxes:
                tc = b["belief"] / max_b if max_b > 1e-6 else 0.0
                color = (255, int(255 * (1.0 - 0.7 * tc)), int(255 * (1.0 - tc)))
                u, v = b["u"], b["v"]
                draw.rectangle([u - half, v - half, u + half, v + half], outline=color, width=2)
                draw.text((u - half, max(0, v - half - 11)), b["label"], fill=color)

        if ee_boxes:
            max_b = max(b["belief"] for b in ee_boxes) or 1.0
            for b in ee_boxes:
                tc = b["belief"] / max_b if max_b > 1e-6 else 0.0
                color = (int(100 * (1.0 - tc)), int(180 + 75 * tc), 255)
                u, v = b["u"], b["v"]
                draw.rectangle([u - half, v - half, u + half, v + half], outline=color, width=2)
                draw.text((u - half, max(0, v - half - 11)), b["label"], fill=color)

    if scale != 1:
        # Pillow requires integer dimensions — a float scale (e.g. 1.5) would
        # otherwise raise TypeError on every single call, which is exactly
        # what broke the video display: the crash happened before the frame
        # counter or image ever got updated, so it looked like a silent hang.
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.NEAREST)
    return img


# ---------------------------------------------------------------------------
# Episode session: holds the HDF5 handle + in-memory label arrays
# ---------------------------------------------------------------------------

class EpisodeSession:
    """One loaded episode: read-only HDF5 handle, precomputed context-plot
    signals, and the in-memory label arrays being edited (only written to
    disk on Save)."""

    def __init__(self, episode_path: Path, cfg: dict):
        self.path = episode_path
        self.cfg = cfg
        self.ep = h5py.File(episode_path, "r")

        self.obs = self.ep["observations"]
        self.act = self.ep.get("actions")
        self.imgs = self.obs["images"]
        self.intent = self.obs.get("intent")
        self.timestamps = self.obs["timestamp_ns"][:]
        self.T = len(self.timestamps)
        self.t0_ep = int(self.timestamps[0])

        self.primary_cam = cfg["camera"].get("primary", "head_cam_left")

        n_slots = int(self.intent["n_slots"][0]) if self.intent is not None and "n_slots" in self.intent else 0
        self.names = candidate_names_from_hdf5(self.intent, n_slots, cfg) if self.intent is not None else []
        self.target_names = list_target_names(self.ep) or [n for n in self.names if n not in ("ee_left", "ee_right")]

        # -- context-plot signals: cheap scalar arrays, load fully upfront --
        self.speed: dict[str, np.ndarray] = {}
        self.contact_force: dict[str, np.ndarray] = {}
        self.gripper_width: dict[str, np.ndarray] = {}
        self.gripper_cmd: dict[str, np.ndarray] = {}
        for side in SIDES:
            arm_obs = self.obs.get(f"arm_{side}")
            arm_act = self.act.get(f"arm_{side}") if self.act is not None else None
            if arm_obs is not None and "O_T_EE" in arm_obs:
                self.speed[side] = compute_ee_speed(arm_obs["O_T_EE"][:])
            else:
                self.speed[side] = np.zeros(self.T)
            if arm_obs is not None and "F_ext" in arm_obs:
                self.contact_force[side] = np.linalg.norm(arm_obs["F_ext"][:, :3], axis=1)
            else:
                self.contact_force[side] = np.zeros(self.T)
            self.gripper_width[side] = (arm_obs["gripper_width"][:]
                                        if arm_obs is not None and "gripper_width" in arm_obs
                                        else np.zeros(self.T))
            self.gripper_cmd[side] = (arm_act["gripper_cmd"][:]
                                      if arm_act is not None and "gripper_cmd" in arm_act
                                      else np.zeros(self.T))

        self.ee_belief: Optional[dict[str, np.ndarray]] = None
        if self.intent is not None and "ee_belief_ee_left" in self.intent and "ee_belief_ee_right" in self.intent:
            self.ee_belief = {
                "left": self.intent["ee_belief_ee_left"][:],
                "right": self.intent["ee_belief_ee_right"][:],
            }

        # -- labels: resume existing hand labels if present, else start blank --
        self.phase: dict[str, np.ndarray] = {}
        self.target: dict[str, list[str]] = {}
        lbl = self.ep.get("labels")
        for side in SIDES:
            pk, tk = f"arm_{side}_phase", f"arm_{side}_target_name"
            if lbl is not None and pk in lbl and len(lbl[pk]) == self.T:
                self.phase[side] = lbl[pk][:].astype(np.int8)
            else:
                self.phase[side] = np.zeros(self.T, dtype=np.int8)
            if lbl is not None and tk in lbl and len(lbl[tk]) == self.T:
                self.target[side] = [_decode(v) for v in lbl[tk][:]]
            else:
                self.target[side] = ["null"] * self.T

        self._history: list[tuple] = []  # undo stack of (side, phase_copy, target_copy)

    def auto_start(self, side: str, before_t: int) -> int:
        """Returns where the next 'Apply to range' should start from: the
        frame right after the nearest already-labeled frame *strictly before*
        before_t, or 0 if nothing before it is labeled.

        This is computed fresh from the actual label arrays every time,
        rather than tracked as a single "furthest labeled" pointer — a
        single global pointer breaks the moment you jump around labeling
        different stretches out of order (e.g. going back to fix an early
        segment after already labeling near the end): the pointer stays
        pinned far ahead, so scrubbing back to work locally looks like
        you're "behind" your own progress and either blocks you or, worse,
        silently overwrites everything in between if it doesn't check
        direction. Searching backward from wherever you currently are
        instead always proposes the locally-correct gap to fill, and by
        construction the result can never exceed before_t, so the "current
        frame is before the start" failure mode this used to hit is
        structurally impossible now.
        """
        before_t = max(0, min(self.T, before_t))
        if before_t == 0:
            return 0
        phase_labeled = self.phase[side][:before_t] != 0
        target_labeled = np.array(self.target[side][:before_t]) != "null"
        labeled = phase_labeled | target_labeled
        nz = np.nonzero(labeled)[0]
        return int(nz[-1]) + 1 if len(nz) else 0

    def close(self) -> None:
        self.ep.close()

    def get_frame_image(self, t: int) -> "Image.Image":
        t = max(0, min(self.T - 1, t))
        frame = self.imgs[self.primary_cam][t]
        return render_frame(frame, self.intent, t, self.cfg, self.ep, self.names)

    def label_text(self, t: int) -> str:
        lines = []
        for side in SIDES:
            phase_name = PHASE_NAMES.get(int(self.phase[side][t]), "?")
            lines.append(f"{side:<5}  {phase_name:<11}  {self.target[side][t]}")
        return "\n".join(lines)

    # -- label editing --
    def snapshot(self, side: str) -> None:
        self._history.append((side, self.phase[side].copy(), list(self.target[side])))
        if len(self._history) > 30:
            self._history.pop(0)

    def undo(self) -> Optional[str]:
        if not self._history:
            return None
        side, phase_copy, target_copy = self._history.pop()
        self.phase[side] = phase_copy
        self.target[side] = target_copy
        return side

    def clear_labels(self, side: str) -> None:
        """Zeros out every existing label for one arm across the whole
        episode: phase -> IDLE, target -> null. Meant for wiping out messy
        old labels before relabeling from scratch. Snapshotted like any
        other edit, so it's undoable."""
        self.snapshot(side)
        self.phase[side][:] = PHASE_IDS["IDLE"]
        self.target[side] = ["null"] * self.T

    def apply_range(self, side: str, lo: int, hi: int, phase_name: str, target_name: str) -> None:
        # Clamp each bound into [0, T-1] individually *before* sorting — if
        # only hi were clamped, a lo past the end of the episode could end
        # up as the unclamped "hi" after sorting and corrupt the range.
        lo = max(0, min(self.T - 1, lo))
        hi = max(0, min(self.T - 1, hi))
        lo, hi = sorted((lo, hi))
        self.snapshot(side)
        self.phase[side][lo:hi + 1] = PHASE_IDS[phase_name]
        for t in range(lo, hi + 1):
            self.target[side][t] = target_name

    def save(self) -> None:
        """Writes the in-memory label arrays back to the episode HDF5."""
        self.ep.close()
        with h5py.File(self.path, "r+") as ep:
            lbl = ep["labels"] if "labels" in ep else ep.create_group("labels")
            # Drop the legacy Pass-1 segment table if present — hand labels
            # only produce per-frame phase/target, not a segment table.
            if "segments" in lbl:
                del lbl["segments"]
            for side in SIDES:
                pk, tk = f"arm_{side}_phase", f"arm_{side}_target_name"
                if pk in lbl:
                    del lbl[pk]
                if tk in lbl:
                    del lbl[tk]
                ds = lbl.create_dataset(pk, data=self.phase[side])
                ds.attrs["legend"] = "  ".join(f"{v}={k}" for k, v in PHASE_NAMES.items())
                names_arr = np.array([n.encode() for n in self.target[side]], dtype="S32")
                lbl.create_dataset(tk, data=names_arr)
            lbl.attrs["pass2_complete"] = True
            lbl.attrs["label_source"] = "manual"
        self.ep = h5py.File(self.path, "r")
        self.obs = self.ep["observations"]
        self.act = self.ep.get("actions")
        self.imgs = self.obs["images"]
        self.intent = self.obs.get("intent")


# ---------------------------------------------------------------------------
# Embedded context plots (EE speed / contact force / EE attention)
# ---------------------------------------------------------------------------

class ContextPlots:
    """Small matplotlib panel embedded in the Tk window, showing the whole
    episode's phase-judgment signals plus your labeling progress, with a
    vertical cursor line tracking the current frame. Lives in the same
    process as the video canvas, so the cursor is always exactly where the
    video is — no cross-process sync.

    "labeled phase" and "labeled target" are step plots of your in-progress
    labels (not raw sensor data) — the point is to let you see, at a glance,
    what you've covered so far and spot gaps or accidental overlaps. They're
    redrawn via update_labels() after every edit; the other three panels are
    static per-episode signals, drawn once at load.
    """

    def __init__(self, parent: tk.Widget):
        # Single column, one row per signal — simpler to make genuinely
        # bigger/readable than a cramped grid, and keeps "labeled phase" and
        # "labeled target" as two distinct, equally prominent panels.
        # figsize is just the initial size — packed with fill="both",
        # expand=True in a frame with no fixed width, the canvas resizes to
        # fill whatever space is actually available (see LabelToolApp). Wide
        # but not tall: height is what was pushing the arm-panel controls
        # off the bottom of the window.
        self.fig = Figure(figsize=(7.6, 6.2), dpi=92)
        gs = self.fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1.2, 1.4])
        self.ax_speed   = self.fig.add_subplot(gs[0])
        self.ax_force   = self.fig.add_subplot(gs[1], sharex=self.ax_speed)
        self.ax_gripper = self.fig.add_subplot(gs[2], sharex=self.ax_speed)
        self.ax_belief  = self.fig.add_subplot(gs[3], sharex=self.ax_speed)
        self.ax_phase   = self.fig.add_subplot(gs[4], sharex=self.ax_speed)
        self.ax_target  = self.fig.add_subplot(gs[5], sharex=self.ax_speed)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Standard matplotlib zoom/pan toolbar. Zooming the x-range on any
        # one panel zooms all of them together for free, since they all
        # share their x-axis (sharex=self.ax_speed above) — matplotlib keeps
        # linked axes' x-limits in sync automatically.
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.cursors: dict = {}
        self._t = 0

    @staticmethod
    def _style(ax, title: str) -> None:
        ax.cla()
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)

    def set_session(self, session: EpisodeSession) -> None:
        x = np.arange(session.T)

        self._style(self.ax_speed, "EE speed")
        self.ax_speed.plot(x, session.speed["left"], linewidth=0.7, label="left")
        self.ax_speed.plot(x, session.speed["right"], linewidth=0.7, label="right")
        self.ax_speed.legend(fontsize=8, loc="upper left")
        self.cursors["speed"] = self.ax_speed.axvline(0, color="black", linewidth=1, linestyle="--")

        self._style(self.ax_force, "contact force")
        self.ax_force.plot(x, session.contact_force["left"], linewidth=0.7)
        self.ax_force.plot(x, session.contact_force["right"], linewidth=0.7)
        self.cursors["force"] = self.ax_force.axvline(0, color="black", linewidth=1, linestyle="--")

        self._style(self.ax_gripper, "gripper (solid=width, dashed=cmd)")
        self.ax_gripper.plot(x, session.gripper_width["left"], linewidth=0.9, color="C0", label="left")
        self.ax_gripper.plot(x, session.gripper_cmd["left"], linewidth=0.9, color="C0", linestyle="--")
        self.ax_gripper.plot(x, session.gripper_width["right"], linewidth=0.9, color="C1", label="right")
        self.ax_gripper.plot(x, session.gripper_cmd["right"], linewidth=0.9, color="C1", linestyle="--")
        self.ax_gripper.legend(fontsize=8, loc="upper left")
        self.cursors["gripper"] = self.ax_gripper.axvline(0, color="black", linewidth=1, linestyle="--")

        self._style(self.ax_belief, "EE attention")
        if session.ee_belief is not None:
            self.ax_belief.plot(x, session.ee_belief["left"], linewidth=0.7)
            self.ax_belief.plot(x, session.ee_belief["right"], linewidth=0.7)
        self.cursors["belief"] = self.ax_belief.axvline(0, color="black", linewidth=1, linestyle="--")

        self.update_labels(session, redraw=False)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_labels(self, session: EpisodeSession, redraw: bool = True) -> None:
        """Redraws just the labeled-phase and labeled-target panels from
        current in-memory state — call this after every edit (apply_range,
        set_here, undo) so labeling progress is visible as you go.

        Preserves whatever x-zoom is currently active: _style() clears the
        axes (ax.cla()), and re-plotting fresh data onto a cleared axes
        autoscales it back to the full range — which, because all six
        panels share their x-axis, would silently reset your zoom on
        *every* panel on every single label edit if left alone.
        """
        xlim = self.ax_speed.get_xlim()
        x = np.arange(session.T)

        self._style(self.ax_phase, "labeled phase")
        self.ax_phase.plot(x, session.phase["left"], linewidth=0.8, drawstyle="steps-post", label="left")
        self.ax_phase.plot(x, session.phase["right"], linewidth=0.8, drawstyle="steps-post", label="right")
        self.ax_phase.set_yticks(list(PHASE_NAMES.keys()))
        self.ax_phase.set_yticklabels(list(PHASE_NAMES.values()), fontsize=8)
        self.ax_phase.legend(fontsize=8, loc="upper left")
        self.cursors["phase"] = self.ax_phase.axvline(self._t, color="black", linewidth=1, linestyle="--")

        names = ["null"] + session.target_names
        code = {n: i for i, n in enumerate(names)}
        self._style(self.ax_target, "labeled target")
        self.ax_target.plot(x, [code.get(n, 0) for n in session.target["left"]],
                            linewidth=0.8, drawstyle="steps-post", label="left")
        self.ax_target.plot(x, [code.get(n, 0) for n in session.target["right"]],
                            linewidth=0.8, drawstyle="steps-post", label="right")
        self.ax_target.set_yticks(range(len(names)))
        self.ax_target.set_yticklabels(names, fontsize=8)
        self.ax_target.legend(fontsize=8, loc="upper left")
        self.cursors["target"] = self.ax_target.axvline(self._t, color="black", linewidth=1, linestyle="--")

        # Re-applying to any one shared-x axes re-syncs the whole group —
        # a no-op on first load (xlim was just captured from the same
        # freshly-autoscaled data), but restores your zoom on every
        # subsequent call.
        self.ax_speed.set_xlim(xlim)

        if redraw:
            self.fig.tight_layout()
            self.canvas.draw()

    def move_cursor(self, t: int) -> None:
        self._t = t
        for c in self.cursors.values():
            c.set_xdata([t, t])
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Tkinter control widgets
# ---------------------------------------------------------------------------

class ArmPanel(ttk.LabelFrame):
    """Phase/target controls for one arm."""

    def __init__(self, parent, side: str, app: "LabelToolApp"):
        super().__init__(parent, text=f"{side} arm", padding=8)
        self.side = side
        self.app = app

        self.phase_var = tk.StringVar(value="IDLE")
        phase_row = ttk.Frame(self)
        phase_row.pack(fill="x", pady=(0, 4))
        for name in PHASE_NAMES.values():
            ttk.Radiobutton(phase_row, text=name, value=name, variable=self.phase_var).pack(side="left")

        target_row = ttk.Frame(self)
        target_row.pack(fill="x", pady=(0, 4))
        ttk.Label(target_row, text="target:").pack(side="left")
        self.target_var = tk.StringVar(value="null")
        self.target_combo = ttk.Combobox(target_row, textvariable=self.target_var, state="readonly", width=14)
        self.target_combo.pack(side="left", padx=4)

        self.start_var = tk.StringVar(value="continues from: 0")
        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Start new segment here", command=self.mark_start).pack(side="left")
        ttk.Button(btn_row, text="Label to here", command=self.apply_range).pack(side="left", padx=4)
        self.start_label = ttk.Label(btn_row, textvariable=self.start_var)
        self.start_label.pack(side="left", padx=6)
        ttk.Button(btn_row, text="Set this frame only", command=self.set_here).pack(side="left")
        ttk.Button(btn_row, text="Clear all labels for this arm", command=self.clear_labels).pack(side="left", padx=(20, 0))

        self.readout_var = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.readout_var, foreground="#555").pack(anchor="w", pady=(4, 0))

        self._start_t: Optional[int] = None  # explicit segment start; None = auto-continue

    def set_target_options(self, names: list[str]) -> None:
        opts = ["null"] + names
        self.target_combo["values"] = opts
        if self.target_var.get() not in opts:
            self.target_var.set("null")

    def mark_start(self) -> None:
        """Pins the start of the next 'Label to here' to the current frame.

        You need this whenever the segment you're about to label doesn't
        start right where your last one ended — most commonly, after an
        idle/rest stretch between two pick-place actions. Those gaps are
        real and should stay unlabeled (IDLE is the default), so "Label to
        here" deliberately won't reach back across one on its own: doing
        that automatically would silently swallow the idle gap into
        whatever phase you label next. Within one continuous action
        (APPROACH -> GRASP -> TRANSPORT -> PLACE, no gaps between them) you
        don't need this — "Label to here" already continues from your last
        labeled frame by itself.
        """
        if self.app.session is None:
            return
        self._start_t = self.app.current_t
        self.start_var.set(f"starts at: {self._start_t} (set)")

    def _auto_start(self) -> int:
        """Where 'Label to here' would start from right now: the nearest
        already-labeled frame before the current one, plus one. Computed
        fresh from the actual label data each time (session.auto_start), so
        it's always correct for wherever you're currently scrubbed to —
        whether that's continuing forward past your furthest progress or
        jumping back to fill a gap earlier in the episode."""
        return self.app.session.auto_start(self.side, self.app.current_t)

    def apply_range(self) -> None:
        """Labels from your last labeled frame (or from 'Start new segment
        here', if you clicked it) through the current frame. This is the
        one click you need for every phase transition *within* one
        continuous action; 'Start new segment here' is the one extra click
        needed only when there's a real gap before this segment."""
        if self.app.session is None:
            return
        hi = self.app.current_t
        lo = self._start_t if self._start_t is not None else self._auto_start()
        self.app.session.apply_range(self.side, lo, hi, self.phase_var.get(), self.target_var.get())
        self._start_t = None  # consumed; next click auto-continues from here again
        self.app.on_labels_changed()

    def set_here(self) -> None:
        if self.app.session is None:
            return
        t = self.app.current_t
        self.app.session.apply_range(self.side, t, t, self.phase_var.get(), self.target_var.get())
        self.app.on_labels_changed()

    def clear_labels(self) -> None:
        """Wipes every existing label for this arm (whole episode) back to
        IDLE / null. For cleaning up old/messy labels before relabeling —
        confirmed first since it touches the full episode, not just a
        range, and undo only remembers the last 30 edits."""
        if self.app.session is None:
            return
        if not messagebox.askyesno(
            "Clear all labels",
            f"Clear ALL {self.side}-arm labels for this episode?\n"
            "Phase -> IDLE, target -> null, for every frame.\n"
            "This can be undone with Undo right after, but not later.",
        ):
            return
        self.app.session.clear_labels(self.side)
        self._start_t = None
        self.app.on_labels_changed()

    def update_readout(self, phase_id: int, target_name: str) -> None:
        self.readout_var.set(f"current frame: {PHASE_NAMES.get(phase_id, '?')} / {target_name}")
        if self._start_t is None and self.app.session is not None:
            self.start_var.set(f"continues from: {self._auto_start()}")


class LabelToolApp:
    def __init__(self, root: tk.Tk, cfg: dict, store_root: str, initial_episode: Optional[str]):
        self.root = root
        self.cfg = cfg
        self.store_root = store_root
        self.session: Optional[EpisodeSession] = None
        self.current_t = 0
        self.playing = False
        self._photo = None  # keep a reference — Tk drops the image otherwise

        root.title("teleop-intent — manual labeling")
        root.geometry("1780x1050")
        root.minsize(1600, 1000)

        # -- episode picker (top) --
        top = ttk.Frame(root, padding=8)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="episode:").pack(side="left")
        episode_ids = load_episode_list()
        labels = [episode_label(e) for e in episode_ids]
        default_label = episode_label(initial_episode) if initial_episode else (labels[0] if labels else "")
        self.episode_var = tk.StringVar(value=default_label)
        self.episode_combo = ttk.Combobox(top, textvariable=self.episode_var, state="readonly", width=28,
                                          values=labels or [default_label])
        self.episode_combo.pack(side="left", padx=4)
        ttk.Button(top, text="Load", command=self.load_episode).pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="no episode loaded")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").pack(side="left", padx=8)

        # -- everything below is packed side="bottom", in reverse visual
        # order, and BEFORE the video/plots area. That reserves their space
        # first no matter how tall the plots want to be, so the controls can
        # never get pushed off the bottom of the window again — previously
        # the expanding video+plots area was packed first and could grow to
        # claim the whole window, leaving these with nowhere to go.
        ttk.Label(root, text="←/→ step 1   shift+←/→ step 10   space play/pause   ctrl+s save   ctrl+z undo",
                  foreground="#888").pack(side="bottom", pady=4)

        # -- per-arm panels --
        panels = ttk.Frame(root, padding=8)
        panels.pack(side="bottom", fill="x")
        self.arm_panels = {side: ArmPanel(panels, side, self) for side in SIDES}
        for p in self.arm_panels.values():
            p.pack(side="left", fill="x", expand=True, padx=4)

        # -- scrubber --
        scrub = ttk.Frame(root, padding=8)
        scrub.pack(side="bottom", fill="x")
        self.frame_var = tk.IntVar(value=0)
        self.scale = ttk.Scale(scrub, from_=0, to=1, orient="horizontal",
                               variable=self.frame_var, command=self._on_scale)
        self.scale.pack(fill="x")
        ctrl = ttk.Frame(scrub)
        ctrl.pack(fill="x", pady=4)
        ttk.Button(ctrl, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left")
        self.play_btn = ttk.Button(ctrl, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(ctrl, text="▶", width=3, command=lambda: self.step(1)).pack(side="left")
        self.time_var = tk.StringVar(value="frame 0 / 0")
        ttk.Label(ctrl, textvariable=self.time_var).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Undo", command=self.undo).pack(side="right")
        ttk.Button(ctrl, text="Save", command=self.save).pack(side="right", padx=4)

        # -- main content: video canvas (left) + context plots (right) —
        # packed LAST so it only fills whatever's left above the reserved
        # controls, instead of claiming the whole window and pushing them out.
        content = ttk.Frame(root, padding=8)
        content.pack(side="top", fill="both", expand=True)
        video_frame = ttk.Frame(content)
        video_frame.pack(side="left", fill="y")
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        self.label_text_var = tk.StringVar(value="")
        ttk.Label(video_frame, textvariable=self.label_text_var, font=("Courier", 11),
                  justify="left").pack(anchor="w", pady=(6, 0))

        # No fixed width / pack_propagate here on purpose: this frame fills
        # whatever space is left after the video column (which just hugs its
        # own fixed image size), and the embedded matplotlib canvas resizes
        # to fill it — so the plots use all the leftover width automatically,
        # including on window resize, without hand-tuning a pixel width.
        plots_frame = ttk.Frame(content)
        plots_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.context_plots = ContextPlots(plots_frame)

        root.bind("<Left>", lambda e: self.step(-1))
        root.bind("<Right>", lambda e: self.step(1))
        root.bind("<Shift-Left>", lambda e: self.step(-10))
        root.bind("<Shift-Right>", lambda e: self.step(10))
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-z>", lambda e: self.undo())

        if self.episode_var.get():
            root.after(200, self.load_episode)

    # -- episode lifecycle --
    def load_episode(self) -> None:
        raw = self.episode_var.get().split()[0]  # strip the "  success  56s" suffix if present
        ep_path = Path(self.store_root) / raw / "episode.hdf5"
        if not ep_path.exists():
            messagebox.showerror("Episode not found", str(ep_path))
            return
        if self.session is not None:
            self.session.close()

        self.status_var.set(f"loading {raw}…")
        self.root.update_idletasks()

        self.session = EpisodeSession(ep_path, self.cfg)
        self.context_plots.set_session(self.session)

        for panel in self.arm_panels.values():
            panel.set_target_options(self.session.target_names)

        self.scale.configure(to=max(1, self.session.T - 1))
        self.current_t = 0
        self.frame_var.set(0)
        self.status_var.set(f"{raw}  —  {self.session.T} frames")
        self.refresh_frame()

    def save(self) -> None:
        if self.session is None:
            return
        self.session.save()
        self.status_var.set(f"saved  ({self.session.T} frames)")

    def undo(self) -> None:
        if self.session is None:
            return
        side = self.session.undo()
        if side:
            self.on_labels_changed()
            self.status_var.set(f"undid last edit ({side})")

    def on_labels_changed(self) -> None:
        """Called after any edit that changes the in-memory label arrays —
        refreshes both the progress plots and the per-frame readouts."""
        if self.session is None:
            return
        self.context_plots.update_labels(self.session)
        self.refresh_frame()

    # -- scrubbing --
    def _on_scale(self, value: str) -> None:
        self.goto(int(float(value)))

    def step(self, delta: int) -> None:
        self.goto(self.current_t + delta)

    def goto(self, t: int) -> None:
        if self.session is None:
            return
        t = max(0, min(self.session.T - 1, t))
        self.current_t = t
        self.frame_var.set(t)
        self.refresh_frame()

    def refresh_frame(self) -> None:
        """Redraws the video canvas, moves the plot cursor, and updates the
        label readouts — all from the single self.current_t. Everything the
        user sees comes from this one call, so there's nothing to fall out
        of sync."""
        if self.session is None:
            return
        img = self.session.get_frame_image(self.current_t)
        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)
        self.context_plots.move_cursor(self.current_t)
        self.label_text_var.set(self.session.label_text(self.current_t))

        ts_s = (int(self.session.timestamps[self.current_t]) - self.session.t0_ep) / 1e9
        self.time_var.set(f"frame {self.current_t} / {self.session.T - 1}   ({ts_s:.2f}s)")
        for side in SIDES:
            self.arm_panels[side].update_readout(
                int(self.session.phase[side][self.current_t]),
                self.session.target[side][self.current_t],
            )

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing or self.session is None:
            return
        if self.current_t >= self.session.T - 1:
            self.playing = False
            self.play_btn.configure(text="▶ Play")
            return
        self.step(1)
        self.root.after(33, self._play_tick)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/playback.yaml", help="Path to playback config YAML")
    ap.add_argument("--episode", default=None, help="Episode folder name to preload, e.g. 076")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store_root = resolve_store_root(cfg)

    root = tk.Tk()
    LabelToolApp(root, cfg, store_root, args.episode)
    root.mainloop()


if __name__ == "__main__":
    main()

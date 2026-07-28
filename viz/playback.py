"""
viz/playback.py  -  Frame-by-frame episode playback with intent overlay in Rerun.

Loads an episode HDF5, steps through every timestep at the configured rate, and logs to Rerun:
  - Primary camera image (head_cam_left by default, configurable)
  - Optional wrist camera images
  - Gaze point overlaid on the primary camera
  - Intent posterior as a bar-chart annotation per candidate
  - EE pose and gripper width as scalar timeseries
  - Model entropy as a scalar timeseries

If a model is configured it is instantiated, reset, and stepped once per
frame to produce live predictions.  With model.name = "none" the intent
panels are populated from the logged slot_belief_* values in the HDF5 for
sanity-checking (clearly labelled as the logger's own filter, not a
learned model).

Usage:
    python viz/playback.py --episode 076
    python viz/playback.py --episode 076 --config configs/playback.yaml
    python viz/playback.py --episode 076 --camera head_cam_right
    python viz/playback.py --episode 076 --no-gaze
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    sys.exit("rerun-sdk is required: pip install rerun-sdk")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    PHASE_NAMES,
    candidate_names_from_cfg,
    candidate_names_from_hdf5,
    compute_ee_speed,
    compute_gaze_px,
    compute_slot_boxes,
    load_config,
)
from models.base import CandidateFeatures, IntentModel, IntentPrediction


def resolve_episode_path(cfg: dict, episode_id: str) -> Path:
    """Returns the episode.hdf5 path for the given episode id."""
    store_root = cfg["data"].get("store_root") or os.environ.get("TELEOP_STORE_ROOT")
    if not store_root:
        sys.exit("data.store_root must be set in config or TELEOP_STORE_ROOT env var")
    return Path(store_root) / episode_id / "episode.hdf5"


def load_model(cfg: dict) -> Optional[IntentModel]:
    """Instantiates and returns the configured model, or None for 'none'."""
    name = cfg["model"].get("name", "none")
    if name == "none":
        return None
    module_path, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model: IntentModel = cls()
    checkpoint = cfg["model"].get("checkpoint")
    if checkpoint:
        model.load(checkpoint)
    return model


def build_candidate_features(intent: h5py.Group, t: int) -> Optional[CandidateFeatures]:
    """Extracts CandidateFeatures for timestep t from the intent group."""
    if "n_slots" not in intent:
        return None
    n_slots = int(intent["n_slots"][t])
    if n_slots == 0:
        return None

    feature_keys = [
        k for k in intent.keys()
        if k.startswith("slot_dist_") or k.startswith("slot_px_") or
           k.startswith("ee_") or k.startswith("gripper_")
    ]
    feature_vecs = []
    for k in sorted(feature_keys):
        arr = intent[k][t]
        feature_vecs.append(np.atleast_1d(arr).astype(np.float32))

    if not feature_vecs:
        return None

    max_n = feature_vecs[0].shape[0]
    features = np.stack(feature_vecs, axis=-1)[:max_n]
    mask = np.zeros(max_n, dtype=bool)
    mask[:n_slots] = True

    slot_types = intent["slot_type_0"][t:t+1] if "slot_type_0" in intent else np.zeros(max_n)
    candidate_types = np.zeros(max_n, dtype=np.int32)

    gaze_valid = bool(intent["gaze_valid"][t]) if "gaze_valid" in intent else True
    ts = int(intent["timestamp_arrival_ns"][t]) if "timestamp_arrival_ns" in intent else 0

    return CandidateFeatures(
        features=features,
        mask=mask,
        candidate_types=candidate_types,
        timestamp_ns=ts,
        gaze_valid=gaze_valid,
    )


def log_camera(name: str, frame: np.ndarray, path: str) -> None:
    """Logs one camera frame to a Rerun image entity."""
    rr.log(path, rr.Image(frame))


def log_gaze(intent: h5py.Group, t: int, cfg: dict, img_shape: tuple, camera: str) -> None:
    """Logs the gaze point as a 2D point overlay on the primary camera.
    Pixel math lives in common.compute_gaze_px — shared with the standalone
    labeler so the two tools can never disagree on where the dot goes.
    """
    gp = compute_gaze_px(intent, t, cfg, img_shape)
    if gp is None:
        return
    gx, gy = gp
    radius = cfg["gaze"].get("dot_radius_px", 5)
    rr.log("camera/primary/gaze", rr.Points2D([[gx, gy]], radii=[radius], colors=[[220, 30, 30, 255]]))


def log_slot_boxes(intent: h5py.Group, t: int, img_shape: tuple, ep_file: h5py.File, names: list[str]) -> None:
    """Overlays projected slot bounding boxes on the primary camera image.
    Projection + belief lookup lives in common.compute_slot_boxes; this
    function only maps belief -> Rerun color and logs the boxes.
    """
    ee_boxes, tgt_boxes = compute_slot_boxes(intent, t, img_shape, ep_file, names)
    box_half = 12.0

    if tgt_boxes:
        max_b = max(b["belief"] for b in tgt_boxes) or 1.0
        colors = []
        for b in tgt_boxes:
            tc = b["belief"] / max_b if max_b > 1e-6 else 0.0
            colors.append([255, int(255 * (1.0 - 0.7 * tc)), int(255 * (1.0 - tc)), 200])
        rr.log("camera/primary/slots_target", rr.Boxes2D(
            centers=[[b["u"], b["v"]] for b in tgt_boxes],
            half_sizes=[[box_half, box_half]] * len(tgt_boxes),
            colors=colors, labels=[b["label"] for b in tgt_boxes],
        ))

    if ee_boxes:
        max_b = max(b["belief"] for b in ee_boxes) or 1.0
        colors = []
        for b in ee_boxes:
            tc = b["belief"] / max_b if max_b > 1e-6 else 0.0
            colors.append([int(100 * (1.0 - tc)), int(180 + 75 * tc), 255, 200])
        rr.log("camera/primary/slots_ee", rr.Boxes2D(
            centers=[[b["u"], b["v"]] for b in ee_boxes],
            half_sizes=[[box_half, box_half]] * len(ee_boxes),
            colors=colors, labels=[b["label"] for b in ee_boxes],
        ))


def log_intent_posterior(
    posterior: np.ndarray,
    mask: np.ndarray,
    names: list[str],
    entropy: float,
    t: int,
) -> None:
    """Logs per-candidate intent posteriors and entropy as Rerun scalars."""
    for i, (p, valid) in enumerate(zip(posterior, mask)):
        if not valid:
            continue
        label = names[i] if i < len(names) else f"slot_{i}"
        rr.log(f"intent/posterior/{label}", rr.Scalars(float(p)))

    rr.log("intent/entropy", rr.Scalars(entropy))


def log_belief_from_hdf5(intent: h5py.Group, t: int, names: list[str]) -> None:
    """Logs EE and target beliefs from HDF5 (two separate Bayesian streams, name-keyed).

    After recompute, datasets are:
      ee_belief_{name}   — end-effector attention (ee_left, ee_right, null)
      tgt_belief_{name}  — object/bin attention   (object_*, bin_*, null)
    Falls back to legacy slot_belief_* if new datasets are absent.
    """
    has_split = any(k.startswith("ee_belief_") for k in intent.keys())

    if has_split:
        for k in sorted(k for k in intent.keys() if k.startswith("ee_belief_")):
            label = k[len("ee_belief_"):]          # strip prefix → "ee_left", "null", etc.
            rr.log(f"intent/ee_belief/{label}", rr.Scalars(float(intent[k][t])))
        for k in sorted(k for k in intent.keys() if k.startswith("tgt_belief_")):
            label = k[len("tgt_belief_"):]
            rr.log(f"intent/target_belief/{label}", rr.Scalars(float(intent[k][t])))
    else:
        # Legacy fallback: monolithic slot_belief_* (numeric, sorted properly)
        belief_keys = sorted((k for k in intent.keys() if k.startswith("slot_belief_")),
                             key=lambda k: int(k.rsplit("_", 1)[-1]))
        for i, k in enumerate(belief_keys):
            label = names[i] if i < len(names) else f"slot_{i}"
            if i > len(names):
                break
            rr.log(f"intent/logger_belief/{label}", rr.Scalars(float(intent[k][t])))


def log_intent_summary(intent: h5py.Group, t: int) -> None:
    """Logs a text panel showing the highest-attention EE and target with probabilities."""
    has_split = any(k.startswith("ee_belief_") for k in intent.keys())

    if has_split:
        ee_items  = [(k[len("ee_belief_"):],  float(intent[k][t]))
                     for k in intent.keys()
                     if k.startswith("ee_belief_") and k != "ee_belief_null"]
        tgt_items = [(k[len("tgt_belief_"):], float(intent[k][t]))
                     for k in intent.keys()
                     if k.startswith("tgt_belief_") and k != "tgt_belief_null"]

        ee_null  = float(intent["ee_belief_null"][t])  if "ee_belief_null"  in intent else 0.0
        tgt_null = float(intent["tgt_belief_null"][t]) if "tgt_belief_null" in intent else 0.0

        ee_top  = max(ee_items,  key=lambda x: x[1]) if ee_items  else ("—", 0.0)
        tgt_top = max(tgt_items, key=lambda x: x[1]) if tgt_items else ("—", 0.0)

        tgt_others = sorted([x for x in tgt_items if x[0] != tgt_top[0] and x[1] > 0.15],
                            key=lambda x: -x[1])
        lines = [
            f"EE:     {ee_top[0]}  ({ee_top[1]:.2f})",
            f"Target: {tgt_top[0]}  ({tgt_top[1]:.2f})  null={tgt_null:.2f}",
        ]
        if tgt_others:
            lines.append("  also: " + "  ".join(f"{n} {p:.2f}" for n, p in tgt_others))
    else:
        lines = ["(recompute to see split EE/target belief)"]

    rr.log("intent/summary", rr.TextDocument("\n".join(lines)))


def load_segments(ep: h5py.File) -> dict | None:
    """Load per-frame label data from the labels/ group.

    Segments (labels/segments/*) are an optional legacy table from the old
    heuristic auto-labeler; hand labels (labeling/label_tool.py) only write
    the per-frame arm_{side}_phase / arm_{side}_target_name arrays, so both
    are treated as optional here — only the arrays that exist are returned.

    Returns dict with keys:
      segments:      list of segment dicts (empty unless a legacy segments table exists)
      arm_phase:     {side: np.ndarray [T] int8}
      arm_target:    {side: np.ndarray [T] bytes}  — target name per frame
      label_source:  str  — "manual", "heuristic", or "" if unset
    or None if the episode has no labels at all.
    """
    if "labels" not in ep:
        return None
    lbl = ep["labels"]

    segs = []
    if "segments" in lbl:
        sg = lbl["segments"]
        n  = int(sg.attrs.get("n_segments", 0))
        if n > 0:
            arms  = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in sg["arm"][:]]
            names = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in sg["object_name"][:]]
            g_fr  = sg["grasp_frame"][:].tolist()
            r_fr  = sg["release_frame"][:].tolist()
            dests = [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in sg["destination"][:]]
            for i in range(n):
                segs.append({"arm": arms[i], "object_name": names[i],
                             "grasp_frame": g_fr[i], "release_frame": r_fr[i],
                             "destination": dests[i] or None})

    arm_phase  = {}
    arm_target = {}
    for side in ("left", "right"):
        pk = f"arm_{side}_phase"
        tk = f"arm_{side}_target_name"
        if pk in lbl:
            arm_phase[side]  = lbl[pk][:]
        if tk in lbl:
            arm_target[side] = lbl[tk][:]

    if not segs and not arm_phase and not arm_target:
        return None

    label_source = str(lbl.attrs.get("label_source", ""))
    return {"segments": segs, "arm_phase": arm_phase,
            "arm_target": arm_target, "label_source": label_source}


def log_label_signals(seg_data: dict, t: int) -> None:
    """Log a compact per-frame label summary (phase + target per arm) as a
    single text panel — a numeric table is easier to read at a glance than
    stepped timeseries plots for a handful of discrete states."""
    arm_phase  = seg_data.get("arm_phase",  {})
    arm_target = seg_data.get("arm_target", {})
    source     = seg_data.get("label_source", "")

    lines = []
    for side in ("left", "right"):
        phase_arr  = arm_phase.get(side)
        target_arr = arm_target.get(side)

        if target_arr is not None and t < len(target_arr):
            raw = target_arr[t]
            tgt = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        else:
            tgt = "null"

        phase_name = PHASE_NAMES.get(int(phase_arr[t]) if phase_arr is not None and t < len(phase_arr) else 0, "?")
        lines.append(f"{side:<5}  {phase_name:<11}  {tgt}")

    header = "arm    phase        target"
    if source:
        header += f"   ({source})"
    rr.log("labels/summary", rr.TextDocument(header + "\n" + "\n".join(lines)))


def log_telemetry(obs: h5py.Group, act: h5py.Group, t: int, arm: str) -> None:
    """Logs EE pose, gripper, velocity, and contact forces for one arm."""
    arm_obs = obs.get(arm)
    arm_act = act.get(arm) if act is not None else None
    base = f"telemetry/{arm}"

    if arm_obs is not None:
        if "O_T_EE" in arm_obs:
            ee = arm_obs["O_T_EE"][t]
            rr.log(f"{base}/ee_pos/x", rr.Scalars(float(ee[12])))
            rr.log(f"{base}/ee_pos/y", rr.Scalars(float(ee[13])))
            rr.log(f"{base}/ee_pos/z", rr.Scalars(float(ee[14])))
        # dq (joint velocities) intentionally not logged — Cartesian twist not available
        if "F_ext" in arm_obs:
            f_ext = arm_obs["F_ext"][t]
            rr.log(f"{base}/contact_force/fx", rr.Scalars(float(f_ext[0])))
            rr.log(f"{base}/contact_force/fy", rr.Scalars(float(f_ext[1])))
            rr.log(f"{base}/contact_force/fz", rr.Scalars(float(f_ext[2])))
        if "gripper_width" in arm_obs:
            rr.log(f"{base}/gripper/width", rr.Scalars(float(arm_obs["gripper_width"][t])))
        if "state" in arm_obs:
            rr.log(f"{base}/state", rr.Scalars(float(arm_obs["state"][t])))

    if arm_act is not None:
        if "O_T_EE_cmd" in arm_act:
            ee_cmd = arm_act["O_T_EE_cmd"][t]
            rr.log(f"{base}/ee_pos/x_cmd", rr.Scalars(float(ee_cmd[12])))
            rr.log(f"{base}/ee_pos/y_cmd", rr.Scalars(float(ee_cmd[13])))
            rr.log(f"{base}/ee_pos/z_cmd", rr.Scalars(float(ee_cmd[14])))
        if "gripper_cmd" in arm_act:
            rr.log(f"{base}/gripper/cmd", rr.Scalars(float(arm_act["gripper_cmd"][t])))


def log_signals(obs: h5py.Group, t: int, arm: str, speed: Optional[np.ndarray]) -> None:
    """Logs phase-judgment aid signals for one arm: EE speed and contact-force
    magnitude. These aren't raw telemetry (that's the right-hand column) —
    they're derived signals meant to help a human eyeball phase transitions
    while labeling or reviewing."""
    side = arm.replace("arm_", "")
    if speed is not None and t < len(speed):
        rr.log(f"signals/ee_speed/{side}", rr.Scalars(float(speed[t])))

    arm_obs = obs.get(arm)
    if arm_obs is not None and "F_ext" in arm_obs:
        f_ext = arm_obs["F_ext"][t][:3]
        rr.log(f"signals/contact_force/{side}", rr.Scalars(float(np.linalg.norm(f_ext))))


def labeling_progress_view() -> rrb.TimeSeriesView:
    """Returns a compact per-arm phase-id strip (0=IDLE..4=PLACE) — a labeling-
    tool-only view so you can see at a glance which stretches of the episode
    have been labeled vs. still default/IDLE. Not used in normal playback."""
    return rrb.TimeSeriesView(name="labeled phase (progress)", origin="labels/phase_id")


def build_blueprint(cfg: dict, has_wrists: bool, extra_label_views: Optional[list] = None) -> rrb.Blueprint:
    """3-column layout: cameras | status + signals + attention | telemetry.

    Phase and target are discrete, low-cardinality states, so they're shown
    as a plain text table (label summary) rather than stepped timeseries
    plots — a number is easier to read at a glance than a staircase. Label
    summary and intent summary sit side by side since they're both compact
    text panels. EE speed and contact force are added as phase-judgment aids
    (transport ~ high speed, grasp/place ~ a contact spike) — signals that
    used to be invisible or dead code. Attention plots get the rest of the
    column's vertical space.

    extra_label_views lets a caller (e.g. the labeling tool) append its own
    views to the bottom of this column without duplicating the layout.
    """
    head_view = rrb.Spatial2DView(name="head camera", origin="camera/primary")

    if has_wrists and cfg["camera"].get("show_wrists", True):
        camera_column = rrb.Vertical(
            head_view,
            rrb.Horizontal(
                rrb.Spatial2DView(name="wrist left",  origin="camera/wrist_left"),
                rrb.Spatial2DView(name="wrist right", origin="camera/wrist_right"),
            ),
            row_shares=[3, 2],
        )
    else:
        camera_column = rrb.Vertical(head_view)

    # ── Status + signals + attention column ─────────────────────────────────
    extra_label_views = extra_label_views or []
    label_rows = [
        rrb.Horizontal(
            rrb.TextDocumentView(name="label summary",  origin="labels/summary"),
            rrb.TextDocumentView(name="intent summary", origin="intent/summary"),
        ),
        rrb.TimeSeriesView(name="EE speed",       origin="signals/ee_speed"),
        rrb.TimeSeriesView(name="contact force",  origin="signals/contact_force"),
        rrb.TimeSeriesView(name="EE attention",     origin="intent/ee_belief"),
        rrb.TimeSeriesView(name="target attention", origin="intent/target_belief"),
        *extra_label_views,
    ]
    row_shares = [1, 1.5, 1.5, 2.5, 2.5] + [1.5] * len(extra_label_views)
    label_column = rrb.Vertical(*label_rows, row_shares=row_shares)

    # ── Telemetry column (compact) ─────────────────────────────────────────
    telemetry_column = rrb.Vertical(
        rrb.TimeSeriesView(name="EE pos — left",    origin="telemetry/arm_left/ee_pos"),
        rrb.TimeSeriesView(name="EE pos — right",   origin="telemetry/arm_right/ee_pos"),
        rrb.TimeSeriesView(name="gripper — left",   origin="telemetry/arm_left/gripper"),
        rrb.TimeSeriesView(name="gripper — right",  origin="telemetry/arm_right/gripper"),
        row_shares=[3, 3, 2, 2],
    )

    return rrb.Blueprint(
        rrb.Horizontal(camera_column, label_column, telemetry_column, column_shares=[2, 2, 1.5]),
        collapse_panels=True,
    )


def run(episode_path: Path, cfg: dict, model: Optional[IntentModel]) -> None:
    """Main playback loop: opens HDF5, resets model, steps through all frames."""
    if not episode_path.exists():
        sys.exit(f"Episode not found: {episode_path}")

    episode_id = episode_path.parent.name
    primary_cam = cfg["camera"].get("primary", "head_cam_left")
    fps = cfg["display"].get("fps", 30)
    frame_dt = 1.0 / fps

    with h5py.File(episode_path, "r") as ep:
        obs = ep["observations"]
        act = ep.get("actions")
        imgs = obs["images"]
        intent = obs.get("intent")
        timestamps = obs["timestamp_ns"][:]
        T = len(timestamps)

        has_wrists = "wrist_cam_left" in imgs and "wrist_cam_right" in imgs
        n_slots = int(intent["n_slots"][0]) if intent and "n_slots" in intent else 0
        names = candidate_names_from_hdf5(intent, n_slots, cfg) if intent is not None else candidate_names_from_cfg(cfg, n_slots)

        seg_data = load_segments(ep)
        if seg_data is not None:
            n_segs = len(seg_data["segments"])
            print(f"Loaded {n_segs} segment(s) from labels/segments")

        speed_arrs = {}
        for arm in ("arm_left", "arm_right"):
            arm_obs = obs.get(arm)
            if arm_obs is not None and "O_T_EE" in arm_obs:
                speed_arrs[arm] = compute_ee_speed(arm_obs["O_T_EE"][:])

        rr.init(f"teleop-intent · episode {episode_id}", spawn=not args.save)
        if args.save:
            rr.save(args.save)
            print(f"Saving recording to {args.save}")
        blueprint = build_blueprint(cfg, has_wrists)
        rr.send_blueprint(blueprint)

        if model is not None:
            model.reset()

        t0_wall = time.time()
        t0_ep = timestamps[0]

        try:
            for t in range(T):
                ts_ns = int(timestamps[t])
                ts_s = (ts_ns - t0_ep) / 1e9
                rr.set_time("episode_time", duration=ts_s)

                if primary_cam in imgs:
                    frame = imgs[primary_cam][t]
                    log_camera(primary_cam, frame, "camera/primary")

                    if intent is not None:
                        log_gaze(intent, t, cfg, frame.shape, primary_cam)
                        log_slot_boxes(intent, t, frame.shape, ep, names)

                if has_wrists and cfg["camera"].get("show_wrists", True):
                    if "wrist_cam_left" in imgs:
                        log_camera("wrist_cam_left", imgs["wrist_cam_left"][t], "camera/wrist_left")
                    if "wrist_cam_right" in imgs:
                        log_camera("wrist_cam_right", imgs["wrist_cam_right"][t], "camera/wrist_right")

                for arm in ("arm_left", "arm_right"):
                    log_telemetry(obs, act, t, arm)
                    log_signals(obs, t, arm, speed_arrs.get(arm))

                if intent is not None:
                    if model is not None:
                        cf = build_candidate_features(intent, t)
                        if cf is not None:
                            pred: IntentPrediction = model.step(cf)
                            n = int(intent["n_slots"][t]) if "n_slots" in intent else len(pred.object_posterior)
                            mask = np.zeros(len(pred.object_posterior), dtype=bool)
                            mask[:n] = True
                            log_intent_posterior(pred.object_posterior, mask, names, pred.object_entropy(), t)
                    else:
                        log_belief_from_hdf5(intent, t, names)
                    if intent is not None:
                        log_intent_summary(intent, t)

                if seg_data is not None:
                    log_label_signals(seg_data, t)

                sleep = frame_dt - (time.time() - t0_wall - ts_s)
                if sleep > 0:
                    time.sleep(sleep)

            print(f"Playback complete: {T} frames, {ts_s:.2f}s")

        except KeyboardInterrupt:
            print("\nPlayback stopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="Episode folder name, e.g. 076")
    ap.add_argument("--config", default="configs/playback.yaml", help="Path to playback config YAML")
    ap.add_argument("--camera", default=None, help="Override camera.primary from config")
    ap.add_argument("--no-gaze", action="store_true", help="Disable gaze overlay regardless of config")
    ap.add_argument("--model", default=None, help="Override model.name, e.g. models.bayesian.BayesianFilter")
    ap.add_argument("--checkpoint", default=None, help="Override model.checkpoint path")
    ap.add_argument("--save", metavar="FILE", nargs="?", const="recording.rrd",
                    help="Save recording to .rrd file instead of spawning viewer (default: recording.rrd)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    store_root = cfg["data"].get("store_root")
    if not store_root:
        dataset_cfg_path = ROOT / "configs" / "dataset.yaml"
        if dataset_cfg_path.exists():
            with open(dataset_cfg_path) as f:
                dataset_cfg = yaml.safe_load(f)
            cfg["data"]["store_root"] = dataset_cfg["data"]["store_root"]

    if args.camera: cfg["camera"]["primary"] = args.camera
    if args.no_gaze: cfg["gaze"]["show"] = False
    if args.model: cfg["model"]["name"] = args.model
    if args.checkpoint: cfg["model"]["checkpoint"] = args.checkpoint

    model = load_model(cfg)
    episode_path = resolve_episode_path(cfg, args.episode)
    run(episode_path, cfg, model)


if __name__ == "__main__":
    main()

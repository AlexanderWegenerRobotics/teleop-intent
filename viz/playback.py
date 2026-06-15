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

from models.base import CandidateFeatures, IntentModel, IntentPrediction


def load_config(path: str) -> dict:
    """Returns merged config: file defaults overridden by CLI flags later."""
    with open(path) as f:
        return yaml.safe_load(f)


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


def candidate_names_from_cfg(cfg: dict, n_slots: int) -> list[str]:
    """Returns per-candidate display names from config, padding with slot indices."""
    names = cfg["display"].get("candidate_names") or []
    result = list(names[:n_slots])
    for i in range(len(result), n_slots):
        result.append(f"slot_{i}")
    return result


def gaze_color(cfg: dict) -> list[int]:
    """Returns the [R, G, B] gaze dot color from config."""
    return cfg["gaze"].get("color", [245, 158, 11])


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

    UE5 logs gaze as: gaze_px_x = GazeUV.X * full_stereo_texture_width.
    The stored images are per-eye (half that width).  gaze_px_ref_width may be
    stored as either the full stereo width or the per-eye width; we normalise
    to full-stereo space before splitting by eye.

    Left eye:  valid when raw_x in [0, eye_w); gx = raw_x
    Right eye: valid when raw_x in [eye_w, 2*eye_w); gx = raw_x - eye_w
    """
    if not cfg["gaze"].get("show", True):
        return
    if "gaze_px_x" not in intent or "gaze_px_y" not in intent:
        return
    if "gaze_valid" in intent and not bool(intent["gaze_valid"][t]):
        return

    stored_h, stored_w = img_shape[:2]
    px_normalized = bool(intent.attrs.get("px_normalized", False))

    raw_x = float(intent["gaze_px_x"][t])
    raw_y = float(intent["gaze_px_y"][t])

    if px_normalized:
        gaze_uv_x = raw_x
        gaze_uv_y = raw_y
    else:
        gaze_ref_w = float(intent.attrs.get("gaze_px_ref_width", stored_w * 2))
        gaze_ref_h = float(intent.attrs.get("gaze_px_ref_height", stored_h))
        full_w = gaze_ref_w if gaze_ref_w > stored_w else stored_w * 2
        gaze_uv_x = raw_x / full_w
        gaze_uv_y = raw_y / gaze_ref_h

    is_right_eye = camera.endswith("_right")
    if is_right_eye:
        if gaze_uv_x < 0.5 or gaze_uv_x > 1.0:
            return
        gx = (gaze_uv_x - 0.5) * 2.0 * stored_w
    else:
        if gaze_uv_x < 0.0 or gaze_uv_x >= 0.5:
            return
        gx = gaze_uv_x * 2.0 * stored_w

    gy = gaze_uv_y * stored_h

    radius = cfg["gaze"].get("dot_radius_px", 5)
    rr.log(
        "camera/primary/gaze",
        rr.Points2D([[gx, gy]], radii=[radius], colors=[[220, 30, 30, 255]]),
    )


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
        label = names[i] if i < len(names) else f"slot {i}"
        rr.log(f"intent/posterior/{label}", rr.Scalars(float(p)))

    rr.log("intent/entropy", rr.Scalars(entropy))


def log_belief_from_hdf5(intent: h5py.Group, t: int, names: list[str]) -> None:
    """Logs slot_belief_* from HDF5 when no model is active (logger filter, labelled clearly)."""
    belief_keys = sorted(k for k in intent.keys() if k.startswith("slot_belief_"))
    for i, k in enumerate(belief_keys):
        label = names[i] if i < len(names) else f"slot {i}"
        rr.log(f"intent/logger_belief/{label}", rr.Scalars(float(intent[k][t])))


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
        if "dq" in arm_obs:
            dq = arm_obs["dq"][t]
            for i, v in enumerate(dq):
                rr.log(f"{base}/velocity/joint_{i}", rr.Scalars(float(v)))
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


def _arm_column(arm: str, label: str) -> rrb.Vertical:
    """Returns a vertical stack of telemetry panels for one arm."""
    base = f"telemetry/{arm}"
    return rrb.Vertical(
        rrb.TimeSeriesView(name=f"{label} · EE pos",        origin=f"{base}/ee_pos"),
        rrb.TimeSeriesView(name=f"{label} · gripper",       origin=f"{base}/gripper"),
        rrb.TimeSeriesView(name=f"{label} · velocity",      origin=f"{base}/velocity"),
        rrb.TimeSeriesView(name=f"{label} · contact force", origin=f"{base}/contact_force"),
    )


def build_blueprint(cfg: dict, has_wrists: bool) -> rrb.Blueprint:
    """Returns a Rerun blueprint: cameras left (1/3), telemetry + intent right (2/3)."""
    head_view = rrb.Spatial2DView(name="head camera", origin="camera/primary")

    if has_wrists and cfg["camera"].get("show_wrists", True):
        camera_column = rrb.Vertical(
            head_view,
            rrb.Horizontal(
                rrb.Spatial2DView(name="wrist left",  origin="camera/wrist_left"),
                rrb.Spatial2DView(name="wrist right", origin="camera/wrist_right"),
            ),
            row_shares=[2, 1],
        )
    else:
        camera_column = rrb.Vertical(head_view)

    intent_column = rrb.Vertical(
        rrb.TimeSeriesView(name="intent posterior", origin="intent/posterior"),
        rrb.TimeSeriesView(name="intent logger belief", origin="intent/logger_belief"),
        rrb.TimeSeriesView(name="entropy", origin="intent/entropy"),
    )

    data_column = rrb.Vertical(
        rrb.Horizontal(
            _arm_column("arm_left",  "left arm"),
            _arm_column("arm_right", "right arm"),
        ),
        intent_column,
        row_shares=[3, 2],
    )

    return rrb.Blueprint(
        rrb.Horizontal(
            camera_column,
            data_column,
            column_shares=[1, 2],
        ),
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
        names = candidate_names_from_cfg(cfg, n_slots)

        rr.init(f"teleop-intent · episode {episode_id}", spawn=True)
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

                if has_wrists and cfg["camera"].get("show_wrists", True):
                    if "wrist_cam_left" in imgs:
                        log_camera("wrist_cam_left", imgs["wrist_cam_left"][t], "camera/wrist_left")
                    if "wrist_cam_right" in imgs:
                        log_camera("wrist_cam_right", imgs["wrist_cam_right"][t], "camera/wrist_right")

                for arm in ("arm_left", "arm_right"):
                    log_telemetry(obs, act, t, arm)

                if intent is not None:
                    if model is not None:
                        cf = build_candidate_features(intent, t)
                        if cf is not None:
                            pred: IntentPrediction = model.step(cf)
                            n = int(intent["n_slots"][t]) if "n_slots" in intent else len(pred.object_posterior)
                            mask = np.zeros(len(pred.object_posterior), dtype=bool)
                            mask[:n] = True
                            log_intent_posterior(
                                pred.object_posterior, mask, names,
                                pred.object_entropy(), t,
                            )
                    else:
                        log_belief_from_hdf5(intent, t, names)

                sleep = frame_dt - (time.time() - t0_wall - ts_s)
                if sleep > 0:
                    time.sleep(sleep)

            print(f"Playback complete: {T} frames, {ts_s:.2f}s")

        except KeyboardInterrupt:
            print("\nPlayback stopped.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--episode", required=True,
                    help="Episode folder name, e.g. 076")
    ap.add_argument("--config", default="configs/playback.yaml",
                    help="Path to playback config YAML")
    ap.add_argument("--camera", default=None,
                    help="Override camera.primary from config")
    ap.add_argument("--no-gaze", action="store_true",
                    help="Disable gaze overlay regardless of config")
    ap.add_argument("--model", default=None,
                    help="Override model.name, e.g. models.bayesian.BayesianFilter")
    ap.add_argument("--checkpoint", default=None,
                    help="Override model.checkpoint path")
    args = ap.parse_args()

    cfg = load_config(args.config)

    store_root = cfg["data"].get("store_root")
    if not store_root:
        dataset_cfg_path = ROOT / "configs" / "dataset.yaml"
        if dataset_cfg_path.exists():
            with open(dataset_cfg_path) as f:
                dataset_cfg = yaml.safe_load(f)
            cfg["data"]["store_root"] = dataset_cfg["data"]["store_root"]

    if args.camera:
        cfg["camera"]["primary"] = args.camera
    if args.no_gaze:
        cfg["gaze"]["show"] = False
    if args.model:
        cfg["model"]["name"] = args.model
    if args.checkpoint:
        cfg["model"]["checkpoint"] = args.checkpoint

    model = load_model(cfg)
    episode_path = resolve_episode_path(cfg, args.episode)
    run(episode_path, cfg, model)


if __name__ == "__main__":
    main()

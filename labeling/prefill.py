"""
labeling/prefill.py — loads the RandomForest models trained by
train_prefill_models.py and uses them to draft phase/target labels for an
arm of a loaded episode.

This is deliberately separate from common.py: it needs scikit-learn +
joblib, which viz/playback.py (the Rerun review tool) has no reason to
depend on.

The output of predict() is a DRAFT, not ground truth — held-out validation
(see train_prefill_models.py --validate) put frame accuracy around 83-86%
for phase and 86-89% for target (roughly 77% on the harder non-null target
frames). That's good enough to turn labeling into "scrub and fix mistakes"
rather than "label from scratch," but it will get things wrong, especially
right at phase transitions. Always review before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"

TGT_NAMES = ["null", "bin_1", "bin_2", "object_1", "object_2", "object_3", "object_4"]
KIN_FEATS = ["speed", "force", "gwidth", "gwidth_rate", "gcmd", "ee_belief", "ee_belief_other"]
BEL_FEATS = [f"bel_{n}" for n in TGT_NAMES]

_model_cache: dict[str, object] = {}


def available() -> bool:
    """Whether trained model files exist on disk (run train_prefill_models.py first if not)."""
    return MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib"))


def _load(name: str):
    if name not in _model_cache:
        try:
            import joblib
        except ImportError as e:
            raise RuntimeError("Pre-fill requires scikit-learn + joblib: pip install scikit-learn joblib") from e
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python labeling/train_prefill_models.py` first "
                "to train pre-fill models from your manually-labeled episodes."
            )
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]


def _build_features(session, side: str) -> dict[str, np.ndarray]:
    """Rebuilds the exact same feature set used at training time, reusing
    the signals EpisodeSession already computed (same compute_ee_speed
    call, same F_ext/gripper columns) so inference can't silently drift
    from what the models were trained on."""
    other = "right" if side == "left" else "left"
    T = session.T
    feats = {
        "speed": session.speed[side],
        "force": session.contact_force[side],
        "gwidth": session.gripper_width[side],
        "gwidth_rate": np.gradient(session.gripper_width[side]),
        "gcmd": session.gripper_cmd[side],
        "ee_belief": session.ee_belief[side] if session.ee_belief is not None else np.zeros(T),
        "ee_belief_other": session.ee_belief[other] if session.ee_belief is not None else np.zeros(T),
    }
    for name in TGT_NAMES:
        key = f"tgt_belief_{name}"
        feats[f"bel_{name}"] = (session.intent[key][:]
                                 if session.intent is not None and key in session.intent
                                 else np.zeros(T))
    return feats


def predict(session, side: str) -> tuple[np.ndarray, list[str]]:
    """Returns (phase_ids, target_names) predicted for one arm across the
    whole episode. Pure prediction only — the caller (EpisodeSession.prefill)
    is responsible for snapshotting for undo and for treating this as a
    draft rather than a save-ready result."""
    feats = _build_features(session, side)

    phase_model = _load(f"phase_{side}")
    X_phase = np.column_stack([feats[k] for k in KIN_FEATS])
    phase_pred = phase_model.predict(X_phase).astype(np.int8)

    target_model = _load(f"target_{side}")
    X_target = np.column_stack([feats[k] for k in KIN_FEATS + BEL_FEATS])
    target_pred = list(target_model.predict(X_target))

    return phase_pred, target_pred

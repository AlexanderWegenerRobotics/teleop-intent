"""Tests for the GRU intent model.

Split deliberately into two halves:

  * feature assembly, masking and label mapping -- pure numpy, runs anywhere,
    including on a machine with no PyTorch.
  * network behaviour -- requires torch, SKIPPED (not failed) without it.

The torch half carries the properties that would otherwise only be assumed:

    CAUSALITY.  The sequence forward pass must equal repeated single-step
    calls, and altering a future frame must not change an earlier output. A
    causal model that is accidentally non-causal scores beautifully offline
    and is worthless deployed, and nothing else in the pipeline would catch
    it -- eval/score.py steps frame by frame and would simply report the
    better number.

    PERMUTATION EQUIVARIANCE.  Candidate index order is an artefact of the
    logger. Reordering the candidates must permute the target probabilities
    identically, or the model is learning something about slot positions.

    MASKING.  An excluded candidate must receive exactly zero probability --
    the same property whose absence in the HMM's sticky filter produced the
    "still targeting the object in the gripper" failure, asserted here before
    it can happen again.

    LEARNABILITY.  The model must be able to overfit a tiny deterministic
    dataset. This does not show it will generalise; it shows the loss, the
    masking and the gradient path are wired together correctly, which is the
    thing that is actually easy to get wrong.

Run:
    python -m pytest tests/test_gru.py -v
    python tests/test_gru.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._stub_contracts import install, Phase, NULL_TARGET  # noqa: E402

install()

# models.gru.train imports h5py at module scope for the episode reader. None of
# the functions tested here touch it, and stubbing keeps these tests runnable
# on a machine without the hdf5 stack -- the same reason the contracts are
# stubbed. Only registered if h5py is genuinely absent, so a real install is
# never shadowed.
if "h5py" not in sys.modules:
    try:
        import h5py  # noqa: F401
    except ImportError:
        import types
        _h5 = types.ModuleType("h5py")
        _h5.File = object
        sys.modules["h5py"] = _h5

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from models.gru.features import (candidate_features, candidate_feature_names,  # noqa: E402
                                  CAND_FEATURE_NAMES)
from models.gru.train import _target_index, compute_norm_stats, phase_class_weights  # noqa: E402


class _Frame:
    """Minimal SensorFrame stand-in; only the fields features.py reads."""

    def __init__(self, n=3, gaze_valid=True, types=None, world=None):
        from tests._stub_contracts import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
        self.candidate_mask = np.ones(n, dtype=bool)
        self.candidate_types = np.zeros(n, dtype=int) if types is None else np.asarray(types)
        self.candidate_features = np.zeros((n, len(CANDIDATE_FEATURE_NAMES)))
        self.candidate_features[:, CANDIDATE_FEATURE_NAMES.index("px_u")] = np.linspace(0.1, 0.9, n)
        self.candidate_features[:, CANDIDATE_FEATURE_NAMES.index("px_v")] = 0.4
        self.candidate_features[:, CANDIDATE_FEATURE_NAMES.index("dist_left")] = np.linspace(0.2, 0.6, n)
        self.candidate_features[:, CANDIDATE_FEATURE_NAMES.index("dist_right")] = np.linspace(0.3, 0.7, n)
        self.global_features = np.zeros(len(GLOBAL_FEATURE_NAMES))
        self.global_features[GLOBAL_FEATURE_NAMES.index("gaze_x")] = 0.1
        self.global_features[GLOBAL_FEATURE_NAMES.index("gaze_y")] = 0.4
        self.gaze_valid = gaze_valid
        self.candidate_world_pos = (np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
                                    if world is None else world)
        self.grasp_confirmed = {}
        self.proprio = {"arm_left": np.zeros(39), "arm_right": np.zeros(39)}
        self.timestamp_ns = 0


# --------------------------------------------------------------------------
# Feature assembly (no torch)
# --------------------------------------------------------------------------

def test_candidate_features_are_finite_and_masked():
    """Invalid candidates must come out as exact zeros, not stale values: a
    masking mistake downstream then trains on nothing rather than on noise."""
    f = _Frame(n=4)
    f.candidate_mask[2] = False
    c = candidate_features(f, "left", np.array([0.2, 0.0, 0.0]), np.zeros(3))
    assert c.shape == (4, len(CAND_FEATURE_NAMES))
    assert np.isfinite(c).all()
    assert (c[2] == 0).all()


def test_alignment_is_reported_unavailable_when_the_arm_is_slow():
    """Below the speed gate the direction of motion is noise. The cosine must
    read as unavailable rather than as a jittery number the network would have
    to learn to ignore -- the same gate the sticky filter used."""
    f = _Frame(n=3)
    idx_cos = CAND_FEATURE_NAMES.index("align_cos")
    idx_ok = CAND_FEATURE_NAMES.index("align_valid")
    slow = candidate_features(f, "left", np.array([1e-4, 0.0, 0.0]), np.zeros(3))
    fast = candidate_features(f, "left", np.array([0.5, 0.0, 0.0]), np.zeros(3))
    assert (slow[:, idx_ok] == 0).all() and (slow[:, idx_cos] == 0).all()
    assert (fast[:, idx_ok] == 1).all()
    assert np.isclose(fast[0, idx_cos], 1.0)      # driving straight at it


def test_invalid_gaze_zeroes_distance_and_flag_together():
    """gaze_dist is meaningless when gaze is invalid. Both the value and its
    flag must go to zero as a pair, which is what lets the network learn
    'zero with flag zero means absent' instead of 'the target is at zero'."""
    f = _Frame(n=3, gaze_valid=False)
    c = candidate_features(f, "left", np.array([0.3, 0.0, 0.0]), np.zeros(3))
    assert (c[:, CAND_FEATURE_NAMES.index("gaze_valid")] == 0).all()
    assert (c[:, CAND_FEATURE_NAMES.index("gaze_dist")] == 0).all()


def test_rich_features_are_additive_and_off_by_default():
    """The default feature set must match what the HMM could see, or the model
    comparison measures an input change rather than a model change."""
    assert candidate_feature_names(False) == CAND_FEATURE_NAMES
    rich = candidate_feature_names(True)
    assert rich[:len(CAND_FEATURE_NAMES)] == CAND_FEATURE_NAMES and len(rich) > len(CAND_FEATURE_NAMES)


def test_null_target_maps_to_the_last_column():
    """The network puts null last, matching the sticky filter's layout so the
    downstream renormalisation is shared. An off-by-one here trains the model
    to predict the wrong candidate everywhere and still converges."""
    idx = _target_index(np.array([0, 2, NULL_TARGET, 1]), n_cand=4)
    assert list(idx) == [0, 2, 4, 1]


def test_unreachable_target_labels_are_dropped_from_the_loss():
    """A label pointing at a candidate the mask excludes is unwinnable: the
    softmax gives it exactly zero probability by construction. Left in it
    contributes a ~1e9 loss term and a gradient that pushes every VALID option
    down, teaching 'nothing here is correct'. It must be marked ignorable, and
    reachable frames must be untouched."""
    from models.gru.train import target_index_masked, IGNORE_INDEX
    mask = np.array([[True, True, False],
                     [True, True, False],
                     [False, False, False],
                     [True, True, True]])
    target = np.array([0, 2, NULL_TARGET, 1])       # frame 1 points at a masked candidate
    idx, bad = target_index_masked(target, mask)
    assert list(bad) == [False, True, False, False]
    assert idx[0] == 0 and idx[3] == 1
    assert idx[1] == IGNORE_INDEX
    assert idx[2] == 3, "null must still map to the last column, never to ignore"


def test_out_of_range_target_label_is_treated_as_unreachable():
    """A stale label naming a candidate that no longer exists must not index
    off the end of the logits -- it is the same disagreement, and crashing on
    it in the middle of epoch 40 would be a bad way to find out."""
    from models.gru.train import target_index_masked, IGNORE_INDEX
    mask = np.ones((2, 2), dtype=bool)
    idx, bad = target_index_masked(np.array([5, 0]), mask)
    assert bad[0] and idx[0] == IGNORE_INDEX
    assert not bad[1] and idx[1] == 0


def test_norm_stats_ignore_masked_candidates_and_constant_channels():
    rng = np.random.default_rng(0)
    data = [{"arm": rng.normal(size=(50, 6)).astype(np.float32),
             "cand": rng.normal(size=(50, 3, 4)).astype(np.float32),
             "mask": rng.random((50, 3)) > 0.3} for _ in range(3)]
    for d in data:
        d["arm"][:, 0] = 7.0                 # constant channel
        d["cand"][~d["mask"]] = 999.0        # masked rows must not enter the statistics
    a_mean, a_std, c_mean, c_std = compute_norm_stats(data)
    assert a_std[0] == 1.0, "constant channel must get std 1, not 0"
    assert (a_std > 0).all() and (c_std > 0).all()
    assert np.abs(c_mean).max() < 5.0, "masked 999s leaked into the candidate statistics"


def test_class_weights_lift_the_rare_phases():
    """Unweighted cross-entropy optimises accuracy, and accuracy on this
    dataset is won by predicting idle -- the same failure the HMM's emission
    bias produced by a different mechanism."""
    data = [{"phase": np.array([Phase.IDLE] * 900 + [Phase.PLACE] * 100)}]
    w, counts = phase_class_weights(data, "inverse")
    assert w[Phase.PLACE] > w[Phase.IDLE]
    assert np.isclose(w[Phase.PLACE] / w[Phase.IDLE], counts[Phase.IDLE] / counts[Phase.PLACE])
    assert (phase_class_weights(data, "none")[0] == 1).all()


# --------------------------------------------------------------------------
# Network behaviour (needs torch)
# --------------------------------------------------------------------------

class Skipped(Exception):
    """Raised by _need_torch when PyTorch is absent.

    Skipping is reported separately from passing, deliberately: a suite that
    prints PASS for a test it never executed is worse than one that prints
    nothing, because it manufactures confidence in exactly the properties
    (causality, masking) that most need checking.
    """


_STANDALONE = False


def _need_torch():
    if HAVE_TORCH:
        return
    if not _STANDALONE:
        try:
            import pytest
            pytest.skip("torch not installed")
        except ImportError:
            pass
    raise Skipped("torch not installed")


def _net(d_arm=6, d_cand=4, hidden=16):
    from models.gru.model import IntentGRU
    torch.manual_seed(0)
    net = IntentGRU(d_arm, d_cand, hidden=hidden, dropout=0.0, cand_hidden=8)
    net.eval()
    return net


def _batch(T=20, n=3, d_arm=6, d_cand=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    arm = torch.randn(1, T, d_arm, generator=g)
    cand = torch.randn(1, T, n, d_cand, generator=g)
    mask = torch.ones(1, T, n, dtype=torch.bool)
    return arm, cand, mask


def test_sequence_forward_equals_stepping_one_frame_at_a_time():
    """THE CAUSALITY TEST. Training runs whole sequences; deployment steps one
    frame at a time carrying the hidden state. If those two computations ever
    diverge, every offline number is measuring a model that does not exist at
    runtime."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch()
    with torch.no_grad():
        pl_seq, tl_seq, _ = net(arm, cand, mask)
        h, pl_step, tl_step = None, [], []
        for t in range(arm.shape[1]):
            p, q, h = net(arm[:, t:t + 1], cand[:, t:t + 1], mask[:, t:t + 1], h)
            pl_step.append(p); tl_step.append(q)
    assert torch.allclose(pl_seq, torch.cat(pl_step, 1), atol=1e-5)
    assert torch.allclose(tl_seq, torch.cat(tl_step, 1), atol=1e-5)


def test_a_future_frame_cannot_change_an_earlier_output():
    """The other half of causality, stated directly rather than inferred from
    the architecture: perturb the last frame and everything before it must be
    bit-identical."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch()
    arm2 = arm.clone(); arm2[0, -1] += 5.0
    cand2 = cand.clone(); cand2[0, -1] += 5.0
    with torch.no_grad():
        a, b, _ = net(arm, cand, mask)
        c, d, _ = net(arm2, cand2, mask)
    assert torch.allclose(a[:, :-1], c[:, :-1], atol=1e-6)
    assert torch.allclose(b[:, :-1], d[:, :-1], atol=1e-6)


def test_target_head_is_permutation_equivariant():
    """Candidate ordering is a logging artefact. Reordering the candidates must
    permute the outputs identically, or the model is learning slot positions --
    which would not survive a scene with a different layout."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=5, n=4)
    perm = [2, 0, 3, 1]
    with torch.no_grad():
        _, t1, _ = net(arm, cand, mask)
        _, t2, _ = net(arm, cand[:, :, perm], mask[:, :, perm])
    assert torch.allclose(t1[..., perm], t2[..., :4], atol=1e-5)
    assert torch.allclose(t1[..., -1], t2[..., -1], atol=1e-5)   # null unaffected


def test_masked_candidates_get_exactly_zero_probability():
    """The failure that cost the HMM a whole checkpoint series: an excluded
    candidate that keeps probability mass. Asserted at the softmax here."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=6, n=4)
    mask[:, :, 1] = False
    with torch.no_grad():
        _, tl, _ = net(arm, cand, mask)
        p = torch.softmax(tl, -1)
    assert torch.all(p[..., 1] == 0.0)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5)


def test_null_state_survives_when_every_candidate_is_masked():
    """A frame can legitimately have no valid candidate. The softmax must stay
    finite and put everything on null rather than producing NaN."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=4, n=3)
    mask[:] = False
    with torch.no_grad():
        _, tl, _ = net(arm, cand, mask)
        p = torch.softmax(tl, -1)
    assert torch.isfinite(p).all()
    assert torch.allclose(p[..., -1], torch.ones_like(p[..., -1]), atol=1e-5)


def test_model_can_overfit_a_tiny_deterministic_dataset():
    """Not a generalisation claim -- a wiring claim. If the loss, the masking
    and the gradient path are connected correctly the model must be able to
    memorise a handful of frames; if any of them is wrong it will plateau
    while still reporting a falling loss."""
    _need_torch()
    import torch.nn.functional as F
    from models.gru.model import IntentGRU
    torch.manual_seed(0)
    T, n, d_arm, d_cand = 40, 3, 4, 3
    # Phase and target are deterministic functions of the arm feature, so a
    # correct model can reach zero loss and an incorrect one cannot.
    phase = torch.arange(T) % Phase.N_CLASSES
    target = torch.arange(T) % n
    arm = F.one_hot(phase, Phase.N_CLASSES).float()[:, :d_arm].unsqueeze(0)
    cand = torch.zeros(1, T, n, d_cand)
    cand[0, torch.arange(T), target, 0] = 1.0
    mask = torch.ones(1, T, n, dtype=torch.bool)

    net = IntentGRU(d_arm, d_cand, hidden=32, dropout=0.0, cand_hidden=16)
    opt = torch.optim.AdamW(net.parameters(), lr=0.02)
    for _ in range(400):
        pl, tl, _ = net(arm, cand, mask)
        loss = (F.cross_entropy(pl[0], phase) + F.cross_entropy(tl[0], target))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pl, tl, _ = net(arm, cand, mask)
    assert (pl[0].argmax(-1) == phase).float().mean() > 0.95
    assert (tl[0].argmax(-1) == target).float().mean() > 0.95


def test_checkpoint_round_trips_and_rejects_a_feature_mismatch():
    _need_torch()
    import tempfile
    from models.gru.model import GRUIntentModel
    m = GRUIntentModel({"hidden": 16, "cand_hidden": 8, "dropout": 0.0})
    m.build(len(__import__("models.gru.features", fromlist=["x"]).ARM_FEATURE_NAMES),
            len(CAND_FEATURE_NAMES))
    m.norm_mean = np.zeros(m.net.d_arm); m.norm_std = np.ones(m.net.d_arm)
    m.cand_mean = np.zeros(m.net.d_cand); m.cand_std = np.ones(m.net.d_cand)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.pt")
        m.save(p)
        m2 = GRUIntentModel(); m2.load(p)
        assert m2.net.d_arm == m.net.d_arm and m2.net.d_cand == m.net.d_cand
        for a, b in zip(m.net.state_dict().values(), m2.net.state_dict().values()):
            assert torch.allclose(a, b)
        # Asking for a configuration the checkpoint was not fit with must
        # fail loudly. Silently adopting the checkpoint's config instead is
        # how someone ends up certain they evaluated the rich variant when
        # they evaluated the plain one -- and the weights only make sense for
        # the features they actually saw.
        try:
            GRUIntentModel({"rich": True}).load(p)
        except ValueError as e:
            assert "rich" in str(e)
        else:
            raise AssertionError("feature-set mismatch loaded without complaint")

        # ...while constructing with no config must adopt the checkpoint's,
        # which is what eval/score.py's loader does.
        plain = GRUIntentModel()
        plain.load(p)
        assert plain.config["rich"] is False


if __name__ == "__main__":
    _STANDALONE = True
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Skipped as e:
            skipped += 1
            print(f"SKIP  {name}  ({e})")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n      {e}")
    passed = len(fns) - failed - skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped  (of {len(fns)})")
    if skipped:
        print("Install PyTorch and rerun -- the skipped tests are the ones that check "
              "causality, masking and permutation equivariance.")
    sys.exit(1 if failed else 0)

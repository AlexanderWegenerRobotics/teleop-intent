"""Tests for the transformer intent model.

The properties here are the ones a windowed causal transformer can silently
get wrong while still training to a good-looking number:

    CAUSALITY. Attention masks are easy to get backwards, and a model that
    peeks at the future scores beautifully offline and is worthless deployed.
    Asserted two ways -- perturbing a future frame must not move an earlier
    output, and the training-time forward must equal the frame-by-frame
    buffered path step() actually runs.

    WINDOW BOUNDS. A frame must see exactly the last W frames: no more (or
    training and serving diverge the moment an episode outruns step()'s
    buffer) and no fewer.

    WINDOW = 1 IS MEMORYLESS. It is the control the whole experiment rests on,
    so it has to be genuinely memoryless rather than approximately so.

    SHARED HEADS. The target head must stay permutation-equivariant and mask
    correctly, exactly as the GRU's does -- if the two differed, a target
    accuracy gap between the models would be uninterpretable.

Requires torch; skipped (not passed) without it.

    python -m pytest tests/test_transformer.py -v
    python tests/test_transformer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._stub_contracts import install, Phase  # noqa: E402

install()

if "h5py" not in sys.modules:
    try:
        import h5py  # noqa: F401
    except ImportError:
        import types
        _h5 = types.ModuleType("h5py")
        _h5.File = object
        sys.modules["h5py"] = _h5

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class Skipped(Exception):
    """Raised when torch is absent; reported separately from a pass."""


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


def _net(d_arm=6, d_cand=4, window=8, d_model=16, layers=2):  # noqa: D401
    from models.transformer.model import IntentTransformer
    torch.manual_seed(0)
    net = IntentTransformer(d_arm, d_cand, d_model=d_model, layers=layers, heads=2,
                            ff_mult=2, dropout=0.0, cand_hidden=8, window=window)
    net.eval()
    return net


def _batch(T=20, n=3, d_arm=6, d_cand=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, T, d_arm, generator=g),
            torch.randn(1, T, n, d_cand, generator=g),
            torch.ones(1, T, n, dtype=torch.bool))


def test_window_mask_blocks_future_and_distant_past():
    """The mask is the whole safety property. Frame t may see [t-W+1, t] and
    nothing else."""
    _need_torch()
    from models.transformer.model import causal_window_mask
    W, T = 4, 10
    m = causal_window_mask(T, W).numpy()
    for q in range(T):
        for k in range(T):
            allowed = (k <= q) and (q - k < W)
            assert m[q, k] == (not allowed), f"query {q} key {k}: expected allowed={allowed}"
    assert not m[5, 5], "a frame must always be able to see itself"


def test_window_of_one_sees_only_the_current_frame():
    """The memoryless control has to be genuinely memoryless."""
    _need_torch()
    from models.transformer.model import causal_window_mask
    m = causal_window_mask(6, 1).numpy()
    assert (~m == np.eye(6, dtype=bool)).all()


def test_a_future_frame_cannot_change_an_earlier_output():
    """CAUSALITY, stated directly rather than inferred from the mask."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch()
    arm2, cand2 = arm.clone(), cand.clone()
    arm2[0, -1] += 5.0
    cand2[0, -1] += 5.0
    with torch.no_grad():
        a, b, _ = net(arm, cand, mask)
        c, d, _ = net(arm2, cand2, mask)
    assert torch.allclose(a[:, :-1], c[:, :-1], atol=1e-5)
    assert torch.allclose(b[:, :-1], d[:, :-1], atol=1e-5)


def test_a_frame_outside_the_window_cannot_change_the_output():
    """The other bound. If distant history still leaked in, step()'s bounded
    buffer would quietly compute something different from training."""
    _need_torch()
    W = 4
    net = _net(window=W)
    arm, cand, mask = _batch(T=12)
    arm2, cand2 = arm.clone(), cand.clone()
    arm2[0, 0] += 100.0                    # 11 frames back, far outside W=4
    cand2[0, 0] += 100.0
    with torch.no_grad():
        a, b, _ = net(arm, cand, mask)
        c, d, _ = net(arm2, cand2, mask)
    assert torch.allclose(a[:, -1], c[:, -1], atol=1e-4)
    assert torch.allclose(b[:, -1], d[:, -1], atol=1e-4)


def test_buffered_stepping_matches_the_full_sequence_forward():
    """Training runs whole sequences; step() replays a bounded buffer. If the
    two diverge, every offline number describes a model that does not exist at
    runtime -- and nothing else in the pipeline would notice.

    The buffer is the RECEPTIVE FIELD, not the per-layer window: two stacked
    local-attention layers see layers*(W-1)+1 frames, because layer 2 attends
    over positions that each already summarised W. Slicing only W frames was
    the first version of this test and it failed -- correctly."""
    _need_torch()
    W, T = 4, 20
    net = _net(window=W, layers=2)
    R = net.receptive_field
    assert R == 2 * (W - 1) + 1
    arm, cand, mask = _batch(T=T)
    with torch.no_grad():
        p_full, t_full, _ = net(arm, cand, mask)
        for t in range(T):
            lo = max(0, t - R + 1)
            p_win, t_win, _ = net(arm[:, lo:t + 1], cand[:, lo:t + 1], mask[:, lo:t + 1])
            assert torch.allclose(p_full[0, t], p_win[0, -1], atol=1e-4), f"phase differs at t={t}"
            assert torch.allclose(t_full[0, t], t_win[0, -1], atol=1e-4), f"target differs at t={t}"


def test_attention_bias_depends_only_on_relative_position():
    """The property that makes buffered stepping exact. An absolute positional
    encoding gives the same frame a different vector depending on where the
    tensor starts, which is what broke the first version of this model."""
    _need_torch()
    from models.transformer.model import attention_bias
    T, W, H = 16, 5, 2
    full = attention_bias(T, W, H)
    for t in range(T):
        lo = max(0, t - W + 1)
        sub = attention_bias(t - lo + 1, W, H)
        assert torch.allclose(full[:, t, lo:t + 1], sub[:, -1, :], equal_nan=True)


def test_no_attention_row_is_entirely_masked():
    """An all -inf row makes softmax return NaN. Every frame must at least see
    itself, whatever the window."""
    _need_torch()
    from models.transformer.model import attention_bias
    for W in (1, 3, 64):
        b = attention_bias(10, W, 2)
        assert torch.isfinite(b).any(dim=-1).all(), f"fully-masked row at window {W}"


def test_train_and_eval_modes_compute_the_same_function():
    """REGRESSION. With dropout at zero the only difference between modes
    should be nothing at all. nn.TransformerEncoderLayer's fused fast path
    engages in eval only and, in several PyTorch versions, silently applies
    post-norm regardless of norm_first -- producing a model that trained to
    0.001 loss and then scored 0.60 on the very data it had memorised. Every
    val number and every eval/score.py number would have inherited that."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=12)
    net.train()
    with torch.no_grad():
        p_train, t_train, _ = net(arm, cand, mask)
    net.eval()
    with torch.no_grad():
        p_eval, t_eval, _ = net(arm, cand, mask)
    assert torch.allclose(p_train, p_eval, atol=1e-6), "phase logits differ between train and eval"
    assert torch.allclose(t_train, t_eval, atol=1e-6), "target logits differ between train and eval"


def test_overfit_result_is_the_same_in_both_modes():
    """The specific symptom, asserted directly: a model that has memorised its
    input must report that in eval mode, not only while training."""
    _need_torch()
    import torch.nn.functional as F
    from models.transformer.model import IntentTransformer
    torch.manual_seed(0)
    T, n, d_arm = 24, 3, 5
    phase = torch.arange(T) % Phase.N_CLASSES
    arm = F.one_hot(phase, Phase.N_CLASSES).float()[:, :d_arm].unsqueeze(0)
    cand = torch.zeros(1, T, n, 3)
    mask = torch.ones(1, T, n, dtype=torch.bool)
    net = IntentTransformer(d_arm, 3, d_model=32, layers=2, heads=2, ff_mult=2,
                            dropout=0.0, cand_hidden=16, window=8)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(600):
        pl, _tl, _ = net(arm, cand, mask)
        loss = F.cross_entropy(pl[0], phase)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        net.train()
        acc_train = float((net(arm, cand, mask)[0][0].argmax(-1) == phase).float().mean())
        net.eval()
        acc_eval = float((net(arm, cand, mask)[0][0].argmax(-1) == phase).float().mean())
    assert abs(acc_train - acc_eval) < 1e-6, (
        f"train accuracy {acc_train:.2f} but eval accuracy {acc_eval:.2f} -- the model "
        f"computes a different function in eval mode")


def test_target_head_is_permutation_equivariant():
    """Shared with the GRU by design; asserted here too so the two target
    accuracies stay comparable."""
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=5, n=4)
    perm = [2, 0, 3, 1]
    with torch.no_grad():
        _, t1, _ = net(arm, cand, mask)
        _, t2, _ = net(arm, cand[:, :, perm], mask[:, :, perm])
    assert torch.allclose(t1[..., perm], t2[..., :4], atol=1e-4)
    assert torch.allclose(t1[..., -1], t2[..., -1], atol=1e-4)


def test_masked_candidates_get_exactly_zero_probability():
    _need_torch()
    net = _net()
    arm, cand, mask = _batch(T=6, n=4)
    mask[:, :, 1] = False
    with torch.no_grad():
        _, tl, _ = net(arm, cand, mask)
        p = torch.softmax(tl, -1)
    assert torch.all(p[..., 1] == 0.0)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5)


def test_presets_bracket_the_gru_parameter_count():
    """'matched' exists to hold capacity at the GRU's ~28k so a difference is
    attributable to architecture. If it drifts far from that, the comparison
    it was built for stops being a comparison."""
    _need_torch()
    from models.transformer.model import IntentTransformer, PRESETS
    counts = {}
    for name, cfg in PRESETS.items():
        net = IntentTransformer(11, 7, window=512,
                                **{k: v for k, v in cfg.items()})
        counts[name] = sum(p.numel() for p in net.parameters())
    gru_params = 28_454
    assert abs(counts["matched"] - gru_params) / gru_params < 0.15, (
        f"matched preset is {counts['matched']} params against the GRU's {gru_params}; "
        f"the comparison it exists for is no longer capacity-controlled")
    assert 2.5 < counts["small"] / counts["matched"] < 5.0, (
        f"small preset is {counts['small']}, only {counts['small'] / counts['matched']:.1f}x "
        f"matched -- too close to be a meaningful capacity contrast")


def test_model_can_overfit_a_tiny_deterministic_dataset():
    """A wiring claim, not a generalisation one: if the mask, the loss and the
    gradient path are connected correctly the model must be able to memorise a
    handful of frames whose labels are a deterministic function of the input.

    Given a longer schedule than the GRU's equivalent on purpose. A pre-norm
    transformer trained with AdamW and no warmup is unstable at the learning
    rate that suits a 64-unit GRU, so a failure there would say "the optimiser
    settings were wrong", not "the model is miswired" -- and this test exists
    only to answer the second question.

    Warmup AND cosine decay, not warmup alone. With the rate held at its peak
    the model reached zero training loss and then walked away from it: Adam's
    update is g / (sqrt(v) + eps), so once the gradient and its second moment
    both collapse the ratio stays order one and the optimiser keeps taking
    full-size steps across a flat landscape. That produced the memorable
    combination of a 0.000 training loss and 0.60 evaluation accuracy. Decaying
    to zero pins the solution once it is found.

    The failure message reports both accuracies so a future regression is
    diagnosable at a glance: near chance (0.2 phase / 0.33 target) means
    something is genuinely disconnected, while 0.8-0.9 means it simply needed
    longer.
    """
    _need_torch()
    import torch.nn.functional as F
    from models.transformer.model import IntentTransformer
    torch.manual_seed(0)
    T, n, d_arm, d_cand = 40, 3, 5, 3
    phase = torch.arange(T) % Phase.N_CLASSES
    target = torch.arange(T) % n
    arm = F.one_hot(phase, Phase.N_CLASSES).float()[:, :d_arm].unsqueeze(0)
    cand = torch.zeros(1, T, n, d_cand)
    cand[0, torch.arange(T), target, 0] = 1.0
    mask = torch.ones(1, T, n, dtype=torch.bool)

    net = IntentTransformer(d_arm, d_cand, d_model=32, layers=2, heads=2, ff_mult=2,
                            dropout=0.0, cand_hidden=16, window=8)
    import math
    steps, warmup, peak = 2000, 200, 3e-3

    def schedule(i):
        if i < warmup:
            return (i + 1) / warmup
        progress = (i - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    opt = torch.optim.AdamW(net.parameters(), lr=peak, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    first = last = None
    for i in range(steps):
        pl, tl, _ = net(arm, cand, mask)
        loss = F.cross_entropy(pl[0], phase) + F.cross_entropy(tl[0], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        value = loss.item()
        first = value if first is None else first
        last = value

    net.eval()
    with torch.no_grad():
        pl, tl, _ = net(arm, cand, mask)
    p_acc = float((pl[0].argmax(-1) == phase).float().mean())
    t_acc = float((tl[0].argmax(-1) == target).float().mean())
    detail = (f"phase {p_acc:.2f} (chance {1 / Phase.N_CLASSES:.2f}), "
              f"target {t_acc:.2f} (chance {1 / n:.2f}), "
              f"loss {first:.3f} -> {last:.3f}")
    assert p_acc > 0.95, f"phase head did not memorise the input: {detail}"
    assert t_acc > 0.95, f"target head did not memorise the input: {detail}"


def test_checkpoint_round_trips_and_rejects_a_config_conflict():
    _need_torch()
    import tempfile
    from models.transformer.model import TransformerIntentModel
    m = TransformerIntentModel({"preset": "matched", "window": 32, "dropout": 0.0})
    m.build(11, 7)
    m.norm_mean, m.norm_std = np.zeros(11), np.ones(11)
    m.cand_mean, m.cand_std = np.zeros(7), np.ones(7)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pt")
        m.save(p)
        m2 = TransformerIntentModel()
        m2.load(p)
        assert m2.config["window"] == 32
        for a, b in zip(m.net.state_dict().values(), m2.net.state_dict().values()):
            assert torch.allclose(a, b)
        try:
            TransformerIntentModel({"window": 512}).load(p)
        except ValueError as e:
            assert "window" in str(e)
        else:
            raise AssertionError("config conflict loaded without complaint")


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
    print(f"\n{len(fns) - failed - skipped} passed, {failed} failed, {skipped} skipped  (of {len(fns)})")
    sys.exit(1 if failed else 0)

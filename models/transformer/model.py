"""TransformerIntentModel: a causal windowed-attention intent model behind the
same IntentModel contract as the HMM and the GRU.

WHAT THIS IS TESTING
--------------------
The GRU's advantage over the structured baseline was measured and localised:
features are worth +0.54 macro F1 over chance, memory adds +0.06 on top, and a
memoryless GRU lands on the HMM to within 0.011. So the open question is not
"is a bigger model better" but a specific one:

    the GRU compresses all history into a fixed 64-vector, updated once per
    frame. Attention instead looks back at every frame in a window directly.
    On a task whose dependencies we measured as a few seconds long, does that
    buy anything?

The honest prior is no. Duration statistics put phase lengths at 57-137
frames, the boundary analysis showed the residual errors are segment EXTENT
rather than missed segments, and 116 training sequences is thin for
attention. A null result here is a finding, not a failure -- it says the task
is short-horizon, which is exactly the argument for taking sequence models to
assembly instead.

WINDOWED CAUSAL ATTENTION
-------------------------
Frame t attends to [t - W + 1, t]. Two reasons W is finite and 512 by default:

  * it spans a full pick-place cycle, so the model CAN see the previous
    grasp-transport-place when predicting the next one. A shorter window would
    make a null result uninformative -- it would only show that attention adds
    nothing over the GRU's own memory, which we already know.
  * step() is called once per frame in a 33 ms control loop. Unbounded
    attention makes per-frame cost grow with episode length, so a model that
    scores well at frame 200 misses its deadline at frame 3000. Bounded
    context is a deployment requirement, not a shortcut.

POSITION IS RELATIVE (ALiBi), NOT ABSOLUTE, and that is a correctness
requirement rather than a preference. An absolute encoding gives frame t a
vector indexed by t, so the same frame encoded inside a full episode and
inside step()'s buffer receives DIFFERENT positions and produces a different
answer -- the first version of this file did exactly that, and
test_buffered_stepping_matches_the_full_sequence_forward caught it. ALiBi adds
a per-head linear penalty on (query - key) distance directly to the attention
scores, so the computation depends only on how far back a frame is, never on
where the tensor happens to start. There is no separate positional embedding.

EFFECTIVE CONTEXT IS layers x window, not window. Each layer looks back W
frames, and a second layer looks back W frames over positions that themselves
already looked back W, so the receptive field is layers*(W-1)+1. step() must
buffer that many frames or it computes something training never saw. The
default W=512 over 2 layers gives 1023 frames -- about 34 s at 30 Hz, roughly
two full pick-place cycles. That is deliberately generous: the experiment is
asking whether long-range attention helps, so the honest version gives it more
reach than the measured dependency length, not less.

The deployment cost follows from the same number. step() re-encodes a
1023-frame buffer every frame, against the GRU's constant per-frame work. If
the two score alike, that asymmetry is a reason to deploy the GRU regardless
of which wins offline.

WINDOW = 1 IS THE MEMORYLESS CONTROL, and it is the direct analogue of the
GRU's --memoryless: identical parameters, attention restricted to the current
frame, so every difference is attributable to looking backwards.

Two heads, same shapes as the GRU: phase is a linear map, target scores each
candidate by additive attention so it stays permutation-equivariant and
handles a variable candidate count. Sharing that design is deliberate -- if
the target heads differed, a target-accuracy gap would be uninterpretable.
"""

from __future__ import annotations

import math

import numpy as np

from teleop_orchestrator.contracts import SensorFrame, ArmIntent, IntentOutput, Phase

from ..base import TrainableIntentModel
from ..gru.features import ARM_FEATURE_NAMES, candidate_feature_names, EpisodeFeatureBuilder

SIDES = ("left", "right")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:                                     # pragma: no cover
    torch = None
    nn = None
    _TORCH = False


def _require_torch():
    if not _TORCH:
        raise ImportError(
            "models.transformer needs PyTorch. Install the build for your CUDA version from "
            "https://pytorch.org (the HMM has no such dependency and still runs without it).")


# Two configurations, both reported. `matched` holds capacity at the GRU's
# ~28k parameters so any difference is attributable to architecture alone;
# `small` is a conventionally-sized transformer. Running both separates
# "attention does not help here" from "it was starved" or "it overfitted
# 116 sequences" -- the first objection a reviewer raises to a null result.
PRESETS = {
    # d_model 40 x 2 layers lands at ~28.8k parameters, against the GRU's
    # 28454 -- close enough that a score difference cannot be attributed to
    # capacity. test_transformer.py asserts this stays true if anyone edits it.
    "matched": dict(d_model=40, layers=2, heads=4, ff_mult=2, dropout=0.1, cand_hidden=32),
    # ~103k: a conventional small transformer, and roughly 3.6x the matched
    # budget. Higher dropout because 116 training sequences is thin for it.
    "small":   dict(d_model=64, layers=2, heads=4, ff_mult=4, dropout=0.2, cand_hidden=32),
}

DEFAULTS = dict(PRESETS["matched"], window=512, rich=False, preset="matched")


if _TORCH:
    def causal_window_mask(t_len: int, window: int, device=None) -> "torch.Tensor":
        """[T, T] bool, True where attention is FORBIDDEN.

        Blocks the future (strictly causal) and anything older than `window`
        frames. Both bounds matter: without the first the model is not
        deployable, and without the second training and serving would disagree
        the moment an episode ran longer than the buffer step() keeps.
        """
        idx = torch.arange(t_len, device=device)
        delta = idx[None, :] - idx[:, None]          # key position minus query position
        return (delta > 0) | (delta < -(window - 1))

    def alibi_slopes(heads: int, device=None) -> "torch.Tensor":
        """One geometric slope per head, the standard ALiBi schedule.

        Different heads penalise distance at different rates, so some attend
        locally and others stay nearly flat across the window -- which is what
        lets a single layer represent both "what just happened" and "what
        happened earlier in this cycle" without a learned embedding.
        """
        return torch.tensor([2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)],
                            device=device)

    def attention_bias(t_len: int, window: int, heads: int,
                       device=None, dtype=None) -> "torch.Tensor":
        """[heads, T, T] additive attention bias: ALiBi inside the window,
        -inf outside it. Broadcasts over the batch axis at the call site.

        Every row keeps at least its diagonal entry finite (a frame always sees
        itself), so no row is entirely -inf and the softmax cannot produce NaN.
        """
        idx = torch.arange(t_len, device=device)
        delta = (idx[None, :] - idx[:, None]).to(dtype or torch.float32)   # key - query, <=0 in the past
        forbidden = causal_window_mask(t_len, window, device=device)
        bias = alibi_slopes(heads, device=device)[:, None, None] * delta[None]
        bias = bias.masked_fill(forbidden[None], float("-inf"))
        return bias

    class _CausalBlock(nn.Module):
        """One pre-norm attention + feed-forward block, written out explicitly.

        NOT nn.TransformerEncoderLayer, and that is the point. That module
        carries a fused "fast path" which engages in EVAL MODE ONLY and, in
        several PyTorch versions, implements post-norm regardless of
        norm_first. The result is a model that trains to zero loss and then
        computes something different the moment it is switched to eval -- which
        is exactly what happened here: 0.001 training loss against 0.60
        evaluation accuracy on data the model had provably memorised. Every
        validation score and every eval/score.py number would have carried that
        error silently.

        Spelling the block out costs about twenty lines and removes the whole
        class of version-dependent, mode-dependent behaviour. Parameter counts
        are unchanged (qkv + proj is 4d^2 + 4d, exactly what MultiheadAttention
        holds), so the matched-capacity comparison is unaffected.
        """

        def __init__(self, d_model: int, heads: int, ff_dim: int, dropout: float):
            super().__init__()
            if d_model % heads:
                raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")
            self.heads, self.d_head = heads, d_model // heads
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                    nn.Linear(ff_dim, d_model))
            self.drop = nn.Dropout(dropout)

        def forward(self, x, bias):
            B, T, D = x.shape
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            shape = (B, T, self.heads, self.d_head)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
            a = a.transpose(1, 2).reshape(B, T, D)
            x = x + self.drop(self.proj(a))
            return x + self.drop(self.ff(self.norm2(x)))

    class IntentTransformer(nn.Module):
        """Free of SensorFrame and episode handling so it can be tested on
        plain tensors, same as IntentGRU."""

        def __init__(self, d_arm: int, d_cand: int, d_model: int = 48, layers: int = 2,
                     heads: int = 4, ff_mult: int = 2, dropout: float = 0.1,
                     cand_hidden: int = 32, window: int = 512):
            super().__init__()
            self.d_arm, self.d_cand, self.d_model, self.window = d_arm, d_cand, d_model, window
            self.layers, self.heads = layers, heads
            self.inp = nn.Linear(d_arm, d_model)
            self.blocks = nn.ModuleList([
                _CausalBlock(d_model, heads, d_model * ff_mult, dropout) for _ in range(layers)])
            self.drop = nn.Dropout(dropout)
            self.phase_head = nn.Linear(d_model, Phase.N_CLASSES)
            self.tgt_h = nn.Linear(d_model, cand_hidden, bias=False)
            self.tgt_c = nn.Linear(d_cand, cand_hidden, bias=True)
            self.tgt_v = nn.Linear(cand_hidden, 1, bias=False)
            self.null_head = nn.Linear(d_model, 1)

        @property
        def receptive_field(self) -> int:
            """Frames the output at t actually depends on: layers*(W-1)+1.

            Stacking local attention compounds -- layer 2 attends over
            positions that already summarised W frames each. step() buffers
            exactly this, and the training chunker overlaps by exactly this,
            so all three paths compute the same function.
            """
            return self.layers * (self.window - 1) + 1

        def forward(self, arm, cand, mask, h0=None):
            """arm [B,T,d_arm]; cand [B,T,n,d_cand]; mask [B,T,n] bool.

            Returns (phase_logits [B,T,5], target_logits [B,T,n+1], None).
            The trailing None keeps the signature interchangeable with
            IntentGRU's, which returns a hidden state; a transformer carries no
            state between calls, so step() replays a window of raw features
            instead. h0 is accepted and ignored for the same reason.
            """
            T = arm.shape[1]
            z = self.inp(arm)
            bias = attention_bias(T, self.window, self.heads, device=arm.device, dtype=z.dtype)
            for block in self.blocks:
                z = block(z, bias)
            out = self.drop(z)

            phase_logits = self.phase_head(out)
            joint = torch.tanh(self.tgt_h(out).unsqueeze(2) + self.tgt_c(cand))
            cand_logits = self.tgt_v(joint).squeeze(-1)
            cand_logits = cand_logits.masked_fill(~mask, -1e9)
            target_logits = torch.cat([cand_logits, self.null_head(out)], dim=-1)
            return phase_logits, target_logits, None


class TransformerIntentModel(TrainableIntentModel):
    """Causal windowed-attention intent model, one feature buffer per arm.

    step() keeps the last `window` frames of RAW FEATURES per arm and re-runs
    the encoder over that buffer each frame, returning the last position's
    output. Re-encoding rather than caching keys and values is deliberate for
    now: it is trivially identical to the training-time computation, which is
    the property test_transformer.py asserts, and correctness matters more
    than throughput at 28k-100k parameters. KV caching is the obvious
    optimisation if the deployed loop ever needs it.

    Note the deployment cost this implies. The GRU's per-frame work is
    constant; this grows with the window, so a transformer that wins offline
    can still be the wrong choice for a 33 ms control loop. That trade belongs
    in the writeup alongside the accuracy numbers.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        preset = cfg.get("preset", DEFAULTS["preset"])
        self._requested = cfg
        merged = dict(DEFAULTS)
        merged.update(PRESETS.get(preset, {}))
        merged.update(cfg)
        self.config = merged
        self.net = None
        self.norm_mean = self.norm_std = self.cand_mean = self.cand_std = None
        self._builders = {s: EpisodeFeatureBuilder(s, rich=self.config["rich"]) for s in SIDES}
        self._buf = {s: None for s in SIDES}
        self._prev_phase = {s: None for s in SIDES}

    def reset(self) -> None:
        if self.net is None:
            raise RuntimeError("TransformerIntentModel has no weights -- call fit() or load() first")
        for s in SIDES:
            self._builders[s].reset()
            self._buf[s] = {"arm": [], "cand": [], "mask": []}
            self._prev_phase[s] = None

    def _normalise(self, arm, cand):
        return (arm - self.norm_mean) / self.norm_std, (cand - self.cand_mean) / self.cand_std

    def step(self, obs: SensorFrame) -> IntentOutput:
        _require_torch()
        # Not `window` -- see IntentTransformer.receptive_field. Buffering only
        # W frames would silently truncate the second layer's context and make
        # step() compute a different function from the one trained.
        W = self.net.receptive_field
        arms, null_probs, pool_masks = {}, {}, {}
        for side in SIDES:
            arm, cand, mask = self._builders[side].step(obs, self._prev_phase[side])
            buf = self._buf[side]
            for key, val in (("arm", arm), ("cand", cand), ("mask", mask)):
                buf[key].append(val)
                if len(buf[key]) > W:
                    buf[key].pop(0)

            a = np.stack(buf["arm"])[None]
            c = np.stack(buf["cand"])[None]
            m = np.stack(buf["mask"])[None]
            a, c = self._normalise(a, c)
            with torch.no_grad():
                p_logits, t_logits, _ = self.net(
                    torch.as_tensor(a, dtype=torch.float32),
                    torch.as_tensor(c, dtype=torch.float32),
                    torch.as_tensor(m, dtype=torch.bool))
            phase_posterior = torch.softmax(p_logits[0, -1], -1).numpy().astype(np.float64)
            full = torch.softmax(t_logits[0, -1], -1).numpy().astype(np.float64)

            null_prob = float(full[-1])
            remaining = 1.0 - null_prob
            if remaining > 1e-9:
                target_posterior = full[:-1] / remaining
            else:
                n_valid = max(int(mask.sum()), 1)
                target_posterior = np.where(mask, 1.0 / n_valid, 0.0)

            self._prev_phase[side] = int(np.argmax(phase_posterior))
            arms[side] = ArmIntent(phase_posterior=phase_posterior,
                                   target_posterior=target_posterior)
            null_probs[side] = null_prob
            pool_masks[side] = mask

        return IntentOutput(left=arms["left"], right=arms["right"],
                            extras={"target_null_prob": null_probs,
                                    "target_pool_mask": pool_masks})

    def build(self, d_arm: int, d_cand: int) -> None:
        _require_torch()
        c = self.config
        self.net = IntentTransformer(d_arm, d_cand, d_model=c["d_model"], layers=c["layers"],
                                     heads=c["heads"], ff_mult=c["ff_mult"], dropout=c["dropout"],
                                     cand_hidden=c["cand_hidden"], window=int(c["window"]))

    def save(self, path: str) -> None:
        _require_torch()
        torch.save({
            "state_dict": self.net.state_dict(), "config": self.config,
            "d_arm": self.net.d_arm, "d_cand": self.net.d_cand,
            "norm_mean": self.norm_mean, "norm_std": self.norm_std,
            "cand_mean": self.cand_mean, "cand_std": self.cand_std,
            "arm_feature_names": list(ARM_FEATURE_NAMES),
            "cand_feature_names": candidate_feature_names(self.config["rich"]),
        }, path)

    def load(self, path: str) -> None:
        _require_torch()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck_config = dict(DEFAULTS, **ck["config"])
        if list(ck["arm_feature_names"]) != list(ARM_FEATURE_NAMES):
            raise ValueError(
                f"checkpoint {path} was fit on arm features {ck['arm_feature_names']} but "
                f"features.py now produces {list(ARM_FEATURE_NAMES)} -- re-fit it.")
        produced = candidate_feature_names(ck_config["rich"])
        if list(ck["cand_feature_names"]) != produced:
            raise ValueError(
                f"checkpoint {path} was fit on candidate features {ck['cand_feature_names']} "
                f"but features.py now produces {produced} -- re-fit it.")
        for key, want in self._requested.items():
            if key in ck_config and ck_config[key] != want:
                raise ValueError(
                    f"checkpoint {path} was fit with {key}={ck_config[key]!r}, not {want!r}. "
                    f"The weights only make sense for the configuration they saw -- construct "
                    f"with no config to adopt the checkpoint's, or re-fit.")
        self.config = ck_config
        self.build(ck["d_arm"], ck["d_cand"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        for k in ("norm_mean", "norm_std", "cand_mean", "cand_std"):
            setattr(self, k, np.asarray(ck[k], dtype=np.float64))
        self._builders = {s: EpisodeFeatureBuilder(s, rich=self.config["rich"]) for s in SIDES}
        self._buf = {s: None for s in SIDES}
        self._prev_phase = {s: None for s in SIDES}

    def fit(self, train_data, val_data, config: dict) -> dict:
        from .train import fit_model
        return fit_model(self, train_data, val_data, dict(self.config, **(config or {})))

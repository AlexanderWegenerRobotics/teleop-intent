"""GRUIntentModel: a causal recurrent intent model behind the same IntentModel
contract as the HMM, so eval/score.py, boundary_analysis.py and viz/playback.py
run it unchanged.

WHAT REPLACES WHAT
------------------
The HMM carried a belief b_t: a probability distribution over 21 discrete
states, updated in closed form by Bayes' rule, with every parameter estimated
by counting. This carries a hidden vector h_t in R^H updated by a learned
gated function. The trade is explicit: the HMM's state means something and its
dynamics are constrained to what the transition matrix allows; h_t means
whatever training makes it mean, and can in principle hold anything -- "the
gripper closed and reopened 400 ms ago", "this is the third parcel" -- that no
5-phase Markov state can represent. That freedom is the whole argument for the
model, and also the whole risk, because 58 episodes is not much to pin down an
unconstrained memory.

    z_t = sigma(W_z x_t + U_z h_{t-1})                  update gate
    r_t = sigma(W_r x_t + U_r h_{t-1})                  reset gate
    n_t = tanh(W_n x_t + r_t * (U_n h_{t-1}))           candidate state
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}               blend

The last line is why a GRU and not a plain RNN: h_t reaches h_{t-1} through a
multiplication by z_t rather than by a weight matrix, so a unit that sets
z_t ~ 1 copies its state forward with gradient ~ 1 and can hold information for
hundreds of frames. A plain RNN multiplies by U every step, and the gradient
vanishes or explodes geometrically. In HMM terms, z_t is a LEARNED, INPUT-
DEPENDENT self-transition -- the thing whose fixed value forced geometric
durations and cost us the whole sub-state chain construction.

STRICTLY CAUSAL, ON PURPOSE. One direction, one layer by default. A
bidirectional GRU would score better on every offline metric here and be
useless: the deployed module has no access to future frames. Nothing in this
file may ever look forward, and test_gru.py asserts it numerically by checking
that the sequence forward pass equals repeated single-step calls.

TWO HEADS, DIFFERENT SHAPES
---------------------------
Phase is a fixed 5-way classification: a linear map off h_t.

Target is not. The candidate set changes size and its ordering is arbitrary,
so a fixed output layer would both break on a different candidate count and
learn meaningless things about index positions. The target head instead SCORES
each candidate from its own features together with the hidden state, additive-
attention style:

    s_j = v^T tanh(W_h h_t + W_c c_j)        for each candidate j
    s_null = w_null^T h_t
    p = softmax([s_j for valid j] + [s_null])

This is permutation-equivariant by construction (reorder the candidates and
the probabilities follow) and works for any candidate count -- the same two
properties the sticky filter had for free, recovered here deliberately rather
than assumed.

Per arm, shared weights, separate hidden state -- matching how the HMM pooled
parameters across arms but kept belief separate. Coupling the two arms is a
plausible extension (what the left arm is doing is informative about the
right) and deliberately not done yet: it changes the comparison and should be
earned on its own evidence.
"""

from __future__ import annotations

import numpy as np

from teleop_orchestrator.contracts import (
    SensorFrame, ArmIntent, IntentOutput, NULL_TARGET, Phase,
)

from ..base import TrainableIntentModel
from .features import (ARM_FEATURE_NAMES, candidate_feature_names, EpisodeFeatureBuilder)

SIDES = ("left", "right")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:                                     # pragma: no cover
    torch = None
    nn = None
    _TORCH = False


def _require_torch():
    if not _TORCH:
        raise ImportError(
            "models.gru needs PyTorch. Install the build for your CUDA version from "
            "https://pytorch.org (the HMM has no such dependency and still runs without it).")


DEFAULTS = dict(
    hidden=64,          # ~28k parameters total; see class docstring on why small
    layers=1,
    dropout=0.1,
    cand_hidden=32,     # width of the target head's scoring MLP
    rich=False,         # matched-to-HMM candidate features by default
    memoryless=False,   # ABLATION: identical parameters, no memory. See IntentGRU.
)


if _TORCH:
    class IntentGRU(nn.Module):
        """The network. Kept free of SensorFrame and of any episode handling so
        it can be unit-tested on plain tensors."""

        def __init__(self, d_arm: int, d_cand: int, hidden: int = 64, layers: int = 1,
                     dropout: float = 0.1, cand_hidden: int = 32, memoryless: bool = False):
            super().__init__()
            self.d_arm, self.d_cand, self.hidden, self.layers = d_arm, d_cand, hidden, layers
            # MEMORYLESS ABLATION. The shuffled-feature control answers "are the
            # features informative?" (they are: chance is 0.24, the model gets
            # 0.85). It does NOT answer the question this model class was chosen
            # for -- "is the MEMORY doing the work, or is this just a better
            # per-frame classifier than the HMM's emission?"
            #
            # Setting this restarts the recurrence from zero at every timestep,
            # so each frame is classified from its own features alone. Same
            # weights, same nonlinearity, same parameter count -- the only thing
            # removed is the ability to carry information across frames, which
            # isolates exactly the property under test.
            #
            # If a memoryless run scores close to the full one, the GRU is not
            # winning because it remembers anything, and the case for sequence
            # models on assembly needs to rest on something other than this
            # dataset.
            self.memoryless = memoryless
            self.inp = nn.Linear(d_arm, hidden)
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.phase_head = nn.Linear(hidden, Phase.N_CLASSES)
            # Additive attention over candidates: W_h on the state, W_c on the
            # candidate, one shared v. Bias only on the candidate branch so the
            # sum is not doubly parameterised.
            self.tgt_h = nn.Linear(hidden, cand_hidden, bias=False)
            self.tgt_c = nn.Linear(d_cand, cand_hidden, bias=True)
            self.tgt_v = nn.Linear(cand_hidden, 1, bias=False)
            self.null_head = nn.Linear(hidden, 1)

        def forward(self, arm, cand, mask, h0=None):
            """arm [B,T,d_arm]; cand [B,T,n,d_cand]; mask [B,T,n] bool.

            Returns (phase_logits [B,T,5], target_logits [B,T,n+1], h_T).
            The null state is the LAST column of target_logits, matching the
            sticky filter's convention so downstream code is shared.
            """
            z = torch.tanh(self.inp(arm))
            if self.memoryless:
                B, T, H = z.shape
                # Every frame becomes its own length-1 sequence with a zero
                # initial state, so no information can cross a timestep.
                flat, _ = self.gru(z.reshape(B * T, 1, H))
                out = flat.reshape(B, T, -1)
                hT = torch.zeros(self.layers, B, self.hidden, dtype=z.dtype, device=z.device)
            else:
                out, hT = self.gru(z, h0)
            out = self.drop(out)

            phase_logits = self.phase_head(out)

            # [B,T,1,C] + [B,T,n,C] broadcast -> one score per candidate
            joint = torch.tanh(self.tgt_h(out).unsqueeze(2) + self.tgt_c(cand))
            cand_logits = self.tgt_v(joint).squeeze(-1)                    # [B,T,n]
            # -1e9, not finfo.min. Both drive the softmax to exactly zero, but
            # finfo.min (-3.4e38) is one subtraction away from overflowing to
            # -inf, and any frame whose LABEL lands on a masked candidate then
            # reports a cross-entropy of ~1e38 or inf. That made the training
            # loss unreadable while the gradients stayed healthy -- the worst
            # kind of fault, since the number you watch is broken and the thing
            # it is meant to warn you about is invisible.
            cand_logits = cand_logits.masked_fill(~mask, -1e9)
            target_logits = torch.cat([cand_logits, self.null_head(out)], dim=-1)
            return phase_logits, target_logits, hT


class GRUIntentModel(TrainableIntentModel):
    """Causal recurrent intent model, one hidden state per arm, shared weights.

    Constructed with no arguments by eval/score.py's loader, so every
    hyperparameter needed to rebuild the network is stored in the checkpoint
    and restored by load(). Training lives in train.py; this class only knows
    how to run and to serialise.
    """

    def __init__(self, config: dict | None = None):
        # Kept separately from the merged config so load() can tell a value the
        # caller actually asked for from one that is merely a default. Without
        # that distinction the feature-set check below compares the checkpoint
        # against itself and can never fail.
        self._requested = dict(config or {})
        self.config = dict(DEFAULTS, **(config or {}))
        self.net = None
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None
        self.cand_mean: np.ndarray | None = None
        self.cand_std: np.ndarray | None = None
        self._builders = {s: EpisodeFeatureBuilder(s, rich=self.config["rich"]) for s in SIDES}
        self._h = {s: None for s in SIDES}
        # CPU only, deliberately. The network is ~28k parameters stepped one
        # arm-sequence at a time, so runtime is dominated by Python-level loop
        # and kernel-launch overhead rather than by arithmetic -- a GPU is
        # typically no faster here and often slower. Inference must run on CPU
        # regardless: step() is called once per frame inside the orchestrator's
        # control loop, where a host-device round trip per frame would cost far
        # more than the forward pass. If a much larger variant ever needs the
        # GPU, add it in train.py only and keep this path on CPU.

    # -- inference --------------------------------------------------------
    def reset(self) -> None:
        if self.net is None:
            raise RuntimeError("GRUIntentModel has no weights -- call fit() or load() first")
        for s in SIDES:
            self._builders[s].reset()
            self._h[s] = None

    def _normalise(self, arm: np.ndarray, cand: np.ndarray):
        """Standardise inputs with statistics frozen from the training split.

        Unlike the HMM -- whose per-dimension diagonal Gaussian was scale
        invariant, so raw metres and newtons mixed freely -- a shared linear
        input layer is not. Contact force ranges over ~20 while gripper delta
        ranges over ~0.003; without this the force channel would dominate the
        first layer purely by magnitude.
        """
        arm = (arm - self.norm_mean) / self.norm_std
        cand = (cand - self.cand_mean) / self.cand_std
        return arm, cand

    def step(self, obs: SensorFrame) -> IntentOutput:
        _require_torch()
        arms, null_probs = {}, {}
        for side in SIDES:
            arm, cand, mask = self._builders[side].step(obs)
            arm_n, cand_n = self._normalise(arm[None, None, :], cand[None, None, :, :])
            with torch.no_grad():
                p_logits, t_logits, h = self.net(
                    torch.as_tensor(arm_n, dtype=torch.float32),
                    torch.as_tensor(cand_n, dtype=torch.float32),
                    torch.as_tensor(mask[None, None, :], dtype=torch.bool),
                    self._h[side])
            self._h[side] = h
            phase_posterior = torch.softmax(p_logits[0, 0], -1).numpy().astype(np.float64)
            full = torch.softmax(t_logits[0, 0], -1).numpy().astype(np.float64)

            null_prob = float(full[-1])
            remaining = 1.0 - null_prob
            if remaining > 1e-9:
                target_posterior = full[:-1] / remaining
            else:
                n_valid = max(int(mask.sum()), 1)
                target_posterior = np.where(mask, 1.0 / n_valid, 0.0)

            arms[side] = ArmIntent(phase_posterior=phase_posterior,
                                   target_posterior=target_posterior)
            null_probs[side] = null_prob
        return IntentOutput(left=arms["left"], right=arms["right"],
                            extras={"target_null_prob": null_probs})

    # -- persistence ------------------------------------------------------
    def build(self, d_arm: int, d_cand: int) -> None:
        _require_torch()
        c = self.config
        self.net = IntentGRU(d_arm, d_cand, hidden=c["hidden"], layers=c["layers"],
                             dropout=c["dropout"], cand_hidden=c["cand_hidden"],
                             memoryless=bool(c.get("memoryless", False)))

    def save(self, path: str) -> None:
        _require_torch()
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "d_arm": self.net.d_arm, "d_cand": self.net.d_cand,
            "norm_mean": self.norm_mean, "norm_std": self.norm_std,
            "cand_mean": self.cand_mean, "cand_std": self.cand_std,
            # Recorded so a checkpoint fit on one feature vector cannot be
            # silently scored against another -- the same guard the HMM's
            # load() has, and for the same reason.
            "arm_feature_names": list(ARM_FEATURE_NAMES),
            "cand_feature_names": candidate_feature_names(self.config["rich"]),
        }, path)

    def load(self, path: str) -> None:
        _require_torch()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck_config = dict(DEFAULTS, **ck["config"])

        # Two distinct failures, both silent if unchecked.
        #
        # 1. CODE DRIFT. Does today's features.py still produce the feature
        #    vector this checkpoint was fit on? If a feature is added, removed
        #    or reordered, every column of the input layer is misaligned and
        #    the model degrades quietly rather than erroring. Compared against
        #    the CHECKPOINT's own config, since that is what it was fit with.
        if list(ck["arm_feature_names"]) != list(ARM_FEATURE_NAMES):
            raise ValueError(
                f"checkpoint {path} was fit on arm features {ck['arm_feature_names']} but "
                f"features.py now produces {list(ARM_FEATURE_NAMES)} -- re-fit it.")
        produced = candidate_feature_names(ck_config["rich"])
        if list(ck["cand_feature_names"]) != produced:
            raise ValueError(
                f"checkpoint {path} was fit on candidate features {ck['cand_feature_names']} "
                f"but features.py now produces {produced} for rich={ck_config['rich']} "
                f"-- re-fit it.")

        # 2. CALLER CONFLICT. Asking for a configuration the checkpoint was not
        #    fit with. The checkpoint has to win -- the weights only make sense
        #    for the features they saw -- so say so rather than silently
        #    ignoring the request, which is how someone ends up convinced they
        #    evaluated the rich variant when they did not.
        for key, want in self._requested.items():
            if key in ck_config and ck_config[key] != want:
                raise ValueError(
                    f"checkpoint {path} was fit with {key}={ck_config[key]!r}, but this "
                    f"model was constructed with {key}={want!r}. The checkpoint's "
                    f"{'candidate features' if key == 'rich' else 'architecture'} cannot be "
                    f"changed after fitting -- construct with no config to adopt the "
                    f"checkpoint's, or re-fit.")

        self.config = ck_config
        self.build(ck["d_arm"], ck["d_cand"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        for k in ("norm_mean", "norm_std", "cand_mean", "cand_std"):
            setattr(self, k, np.asarray(ck[k], dtype=np.float64))
        self._builders = {s: EpisodeFeatureBuilder(s, rich=self.config["rich"]) for s in SIDES}
        self._h = {s: None for s in SIDES}

    def fit(self, train_data, val_data, config: dict) -> dict:
        """Delegates to train.fit_model, which owns the optimisation loop.

        Kept thin deliberately: this class is what the orchestrator and the
        eval harness instantiate, and neither should drag in an optimiser, a
        scheduler or a training-time dependency just to run inference.
        """
        from .train import fit_model
        return fit_model(self, train_data, val_data, dict(self.config, **(config or {})))

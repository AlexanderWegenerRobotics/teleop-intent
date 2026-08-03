"""A learned, causal discrete HMM for the phase head.

Not deep learning: the 5x5 transition matrix is estimated by counting
label-to-label transitions across labeled episodes, and the per-class
emission model is fit in closed form (MLE) to labeled frames -- no EM, since
phase is fully observed in the training data (see labels.py). Inference is
the standard causal forward algorithm: online, uses only information up to
the current step, exactly what Module.step requires.

WHAT CHANGED AND WHY (the v7 -> v8 emission rework)
---------------------------------------------------
v7's emission was a plain per-class diagonal Gaussian over all nine features.
Measured on the shipped v7 checkpoint, that model could not recognise its own
training means: initialised at 100% belief in APPROACH and fed the exact
APPROACH class mean every frame, the forward filter collapsed into IDLE
within ~30 frames, and then stayed pinned at IDLE with probability 1.000 even
when subsequently fed the exact GRASP class mean. That is a property of the
emission model, not of the data, and it is the direct cause of the observed
"drops to idle between approach and grasp" failure.

The mechanism is the Gaussian normalizer. A diagonal Gaussian's log-density
carries a -sum_j log(sigma_kj) term that depends only on class k, never on
the observation, so it acts as a constant per-class bonus on every frame.
Summed over v7's nine features it was worth 24.31 nats for IDLE against 18.93
for GRASP: a 5.4-nat (~220:1) head start for IDLE before any evidence is
considered. Meanwhile the actual IDLE-vs-GRASP mean separation was at most
0.75 pooled std on any single feature, so the Mahalanobis term could never
overturn it. The transition prior does not rescue this either:
transition[APPROACH, IDLE] = 0.0001 looks like a 9-nat wall, but IDLE's
residual belief mass never decays (transition[IDLE, IDLE] = 0.9989), so from
a confident APPROACH belief the predict step favoured GRASP over IDLE by only
0.87 nats -- against 5.4 nats of unconditional emission bias the wrong way.
The same mechanism explains PLACE's 6% recall: PLACE had the lowest
normalizer total of all five classes (18.28), and was predicted on 210 frames
out of ~27,000.

Three changes, each independently switchable so the contribution of each can
be ablated rather than assumed:

1. BERNOULLI FOR BINARY FEATURES (features.IS_BINARY_FEATURE). is_holding is
   {0, 1}; forcing it through a Gaussian let its per-class std collapse
   (0.095 under IDLE) and blow up the normalizer. Under a Bernoulli the
   analogous term is bounded by the probability floor. is_holding,
   gripper_width and gripper_width_delta together supplied 3.7 of the 5.4
   nats, and they are three views of one physical signal.

2. RELATIVE STD FLOOR (std_floor_rel). v7's floor was absolute, 1e-3 -- which
   is meaningless for gripper_width_delta, whose real scale IS ~1e-3 (v7 std
   0.001 under IDLE against 0.0031 under GRASP, worth another 1.15 nats).
   Flooring each class std at a fraction of that feature's GLOBAL std bounds
   the normalizer spread per feature at log(1/std_floor_rel) by construction.

3. EMISSION TEMPERATURE (emission_temp, fit on val -- see
   fit_emission_temperature). The features are strongly correlated (three
   gripper-derived, three velocity components), and a diagonal emission
   treats them as independent evidence, so the likelihood is overconfident by
   roughly the redundancy factor and systematically overwhelms the transition
   prior. Raising the whole emission likelihood to a power < 1 is the
   standard correction, and it is the single highest-leverage knob here.

tie_covariance is offered as a diagnostic rather than a default: sharing one
pooled within-class covariance across all five classes makes the normalizer
term identical for every class, so it cancels exactly out of the posterior.
If a tied fit recovers most of the gap, the normalizer bias was the whole
story; if it does not, something else is also wrong. Cheap to run, and worth
running once before concluding anything about the model class.

Missing values (features.py emits NaN for undefined distances) are skipped
per-dimension -- BOTH the Mahalanobis term and the matching normalizer term,
since dropping only the first would reintroduce exactly the observation-
independent per-class bias this rework exists to remove.

SUB-STATE CHAINS: FIXING THE DURATION MODEL (v10)
--------------------------------------------------
A first-order HMM does not model how long a phase lasts. Duration is whatever
falls out of the self-transition rho, which forces it to be geometric:

    P(D = d) = rho^(d-1) (1 - rho),   E[D] = 1/(1-rho),   CV[D] = sqrt(rho)

The mean is free -- fitting rho sets it to anything you like -- but the SHAPE
is not. Two properties are fixed no matter what rho is:

  * the mode is always d = 1. The single most likely duration for any phase is
    one frame, always.
  * the coefficient of variation is always ~1, i.e. the standard deviation is
    as large as the mean.

Measured on the training labels (scripts/duration_stats.py), the observed CVs
are approach 0.44, transport 0.32, place 0.50 -- against the 1.0 the model is
forced to assume. Transport's fitted duration spread is 3x too wide. The model
is spending probability mass on durations that never occur, and the only thing
holding a phase together against that is the emission, which is why the fitted
emission temperature had to fall to 0.08.

THE FIX, without leaving the HMM. Replace each phase k with a CHAIN of N_k
sub-states traversed in order, all sharing phase k's emission:

    k_1 --> k_2 --> ... --> k_N --> (other phases)
    (each with self-loop rho'_k, advancing with probability 1 - rho'_k)

Time in the phase is now the sum of N_k independent geometric holds, which is
a negative binomial (the discrete Erlang / "method of stages"):

    E[D] = N_k / (1 - rho'_k),    CV[D] = sqrt(rho'_k / N_k) ~ 1 / sqrt(N_k)

So N_k is chosen to hit the observed CV, N_k ~ 1/CV_k^2, and rho'_k is then
set to PRESERVE the mean the counted transition matrix already implies:

    1 - rho'_k = N_k (1 - rho_k)

Two things follow. The mean duration is unchanged from the plain HMM by
construction, so the only difference between this model and v9 is the shape of
the duration distribution -- which makes the ablation clean. And the mode
moves off d=1 to the interior, so the model can finally hold the belief "a
place lasts about fifty frames" rather than "a place most likely ends
immediately".

WHY THIS IS NOT AN HSMM, AND WHY THAT IS THE POINT. The expanded process is
still Markov in the sub-state space, so inference is the same forward
algorithm on a larger transition matrix -- no new inference code, no duration
distribution to fit. It also adds NO new continuous parameters: emissions are
shared across a chain, and each phase still contributes exactly one
self-transition scalar. N_k is an integer read off the data. An HSMM would fit
an explicit duration distribution per phase, which is strictly more parameters
from the same ~240 segments, to buy a flexibility the measured CVs say is not
needed. Earn that later, if this falls short.

WHAT IT WILL NOT FIX. Sub-states can only make a duration MORE regular
(CV shrinks as 1/sqrt(N)). Phases whose observed CV is already >= 1 -- grasp
at 1.03, idle at 1.73 -- get N = 1 and are left exactly as they were. Their
durations are not mismodelled; their labels cover more than one behaviour (a
102-frame mean grasp with a 190-frame p90 is not one action), and no
transition structure fixes that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from teleop_orchestrator.contracts import Phase

from metrics import classification_metrics, segment_durations

from .features import FEATURE_NAMES, IS_BINARY_FEATURE

# Last-resort guard against an exactly-degenerate dimension; the meaningful
# floor is the RELATIVE one below (a class std can never fall below
# std_floor_rel times that feature's global std), which is what actually
# bounds the per-class normalizer spread.
_STD_FLOOR_ABS = 1e-9

# Default: no class std may be smaller than a quarter of the feature's global
# std. This caps the normalizer spread for that feature at log(1 / 0.25) =
# 1.39 nats across classes, against the unbounded spread v7 permitted.
DEFAULT_STD_FLOOR_REL = 0.25

# Bernoulli probabilities are clipped into [floor, 1 - floor], bounding a
# single binary feature's log-likelihood contribution at |log(0.02)| = 3.9
# nats instead of letting a pure class contribute unboundedly.
DEFAULT_BERNOULLI_FLOOR = 0.02

_TRANS_SMOOTHING = 1.0  # Laplace smoothing count added to every transition

# Searched by fit_emission_temperature. Spans "essentially ignore the
# emission" to "unchanged from v7" (1.0), so the sweep can always fall back
# to the old behaviour if it genuinely scores best.
DEFAULT_TEMP_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0)

# Temperature 0 switches the emission off entirely (every class gets the same
# likelihood), leaving the filter running on the transition and duration
# structure alone. Always scored as a REFERENCE row and never selectable: it
# is the "predict the task script without looking at the robot" floor, and the
# gap between it and the selected temperature is how much the sensing is
# actually worth. This matters more than it sounds -- a stereotyped
# pick-and-place task can be predicted surprisingly well from structure alone,
# and a model doing that would fail exactly on the deviations (fumbles,
# regrasps, hesitation) that an operator-nudging system exists to catch.
PRIOR_ONLY_TEMP = 0.0

# Longest sub-state chain sub_states_from_durations will propose. A chain of N
# reproduces CV = 1/sqrt(N), so 12 covers down to CV ~ 0.29. Below that the
# transition matrix is encoding a duration law that an explicit HSMM duration
# distribution would express with fewer parameters -- hitting this cap is the
# evidence that the HSMM is finally worth its cost, not a number to raise.
DEFAULT_MAX_SUB_STATES = 12


@dataclass
class PhaseHMMParams:
    """Fitted parameters: everything PhaseHMM.step needs, nothing it doesn't."""

    transition: np.ndarray        # [5, 5] transition[i, j] = P(phase_t=j | phase_{t-1}=i)
    prior: np.ndarray             # [5] initial belief, used by reset()
    emission_mean: np.ndarray     # [5, n_features] -- Gaussian dims only (binary dims unused)
    emission_std: np.ndarray      # [5, n_features] -- Gaussian dims only (binary dims unused)
    # P(feature = 1 | class) for dimensions flagged in is_binary; the Gaussian
    # mean/std entries for those columns are still populated (so the arrays
    # stay rectangular and old diagnostics keep working) but are not read.
    bernoulli_p: np.ndarray | None = None      # [5, n_features]
    is_binary: np.ndarray | None = None        # [n_features] bool
    # Exponent applied to the emission likelihood; 1.0 reproduces v7 exactly.
    emission_temp: float = 1.0
    tie_covariance: bool = False
    # [n_classes] chain length per phase; None or all-ones is the plain
    # geometric-duration HMM (v9 and earlier) exactly.
    sub_states: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    @property
    def chain_lengths(self) -> np.ndarray:
        """sub_states, defaulting to one sub-state per phase (plain HMM)."""
        n = self.transition.shape[0]
        if self.sub_states is None:
            return np.ones(n, dtype=int)
        return np.asarray(self.sub_states, dtype=int)

    def duration_stats(self) -> dict:
        """Mean and CV of each phase's duration under the CURRENT structure.

        E[D] = N/(1-rho'), CV[D] = sqrt(rho'/N), with 1-rho' = N(1-rho) so the
        mean matches the plain HMM by construction. Worth printing next to the
        observed durations after any fit: if the CV column still reads ~1.0 for
        a phase whose measured CV is 0.3, the chain length is not doing its job.
        """
        n = self.transition.shape[0]
        N = self.chain_lengths
        rho = np.diag(self.transition)
        q = np.minimum(N * (1.0 - rho), 1.0)          # per-sub-state advance probability
        rho_prime = 1.0 - q
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(q > 0, N / q, np.inf)
            cv = np.sqrt(np.maximum(rho_prime, 0.0) / N)
        return {"sub_states": N, "rho": rho, "rho_prime": rho_prime,
                "mean_duration": mean, "cv_duration": cv,
                "n_expanded_states": int(N.sum()), "n_phases": n}

    @property
    def binary_mask(self) -> np.ndarray:
        """is_binary, defaulting to all-Gaussian for checkpoints written
        before binary features existed (v1-v7)."""
        if self.is_binary is None:
            return np.zeros(self.emission_mean.shape[1], dtype=bool)
        return np.asarray(self.is_binary, dtype=bool)

    def normalizer_bias(self) -> np.ndarray:
        """[5] the observation-INDEPENDENT part of each class's emission
        log-likelihood, assuming every Gaussian dimension is observed.

        This is the quantity that broke v7 (24.31 for IDLE against 18.93 for
        GRASP). Exposed as a first-class method because it should be looked
        at after every fit -- see scripts/diagnose_emissions.py. A healthy fit
        has a small spread here; a large spread means the model has an
        opinion about which phase it is in before it has looked at anything.
        """
        gauss = ~self.binary_mask
        if not gauss.any():
            return np.zeros(self.emission_mean.shape[0])
        return -np.log(self.emission_std[:, gauss]).sum(axis=1) * self.emission_temp


def expand_chains(transition: np.ndarray, prior: np.ndarray,
                  sub_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the sub-state transition matrix, prior and owner map.

    Returns (B [S, S], prior_expanded [S], owner [S]) where S = sum(sub_states)
    and owner[s] is the phase that sub-state s belongs to.

    Construction, for phase k with chain length N and base self-transition rho:

      * advance probability q = N (1 - rho), so that the chain's mean duration
        N/q equals the plain model's 1/(1-rho) exactly. Only the SHAPE of the
        duration changes; nothing about the fit's mean is being altered.
      * each sub-state holds with probability rho' = 1 - q and advances to the
        next with q.
      * the LAST sub-state exits with q, distributed over other phases' FIRST
        sub-states in proportion to the base matrix's off-diagonal row,
        renormalised. The destination distribution is therefore preserved
        exactly too -- the expansion changes when a phase ends, never where it
        goes next.

    The prior is spread uniformly across each chain, because an episode can
    begin at any point within a phase and equal self-loops make time spent in
    each sub-state equal; putting all the mass on k_1 would assert that every
    recording starts exactly at a phase boundary.
    """
    n = transition.shape[0]
    N = np.asarray(sub_states, dtype=int)
    if N.shape[0] != n or (N < 1).any():
        raise ValueError(f"sub_states must be {n} positive integers, got {sub_states}")

    offset = np.concatenate([[0], np.cumsum(N)[:-1]]).astype(int)
    owner = np.repeat(np.arange(n), N)
    S = int(N.sum())
    B = np.zeros((S, S))

    for k in range(n):
        rho = float(transition[k, k])
        # A chain of N stages cannot be longer than the mean duration it has to
        # reproduce; clamping keeps the matrix stochastic, and train.py's chain
        # sizing already avoids ever hitting this.
        q = min(float(N[k]) * (1.0 - rho), 1.0)
        rho_prime = 1.0 - q
        others = np.delete(np.arange(n), k)
        out_weight = transition[k, others]
        out_total = float(out_weight.sum())

        for j in range(int(N[k])):
            s = offset[k] + j
            B[s, s] = rho_prime
            if j < N[k] - 1:
                B[s, s + 1] = q
            elif out_total > 0:
                for l, w in zip(others, out_weight):
                    B[s, offset[l]] = q * float(w) / out_total
            else:
                B[s, s] = 1.0  # absorbing in the base model too; preserve that

    prior_expanded = np.asarray(prior, dtype=float)[owner] / N[owner]
    total = prior_expanded.sum()
    prior_expanded = prior_expanded / total if total > 0 else np.full(S, 1.0 / S)
    return B, prior_expanded, owner


class PhaseHMM:
    """Causal forward-filtered HMM for one arm.

    Belief is carried in the SUB-STATE space (size sum(sub_states)); step()
    collapses it back to a Phase.N_CLASSES posterior on the way out, so every
    caller -- the orchestrator, the eval harness, the diagnostics -- keeps
    seeing exactly the five-phase contract it always did. With sub_states all
    ones the expanded matrix is the base matrix and this is bit-for-bit the
    previous model.
    """

    def __init__(self, params: PhaseHMMParams):
        self.params = params
        self._transition, self._prior, self._owner = expand_chains(
            params.transition, params.prior, params.chain_lengths)
        self._n_phases = params.transition.shape[0]
        self.state_belief = self._prior.copy()

    @property
    def belief(self) -> np.ndarray:
        """The five-phase posterior; the sub-state decomposition is internal."""
        return self._collapse(self.state_belief)

    def _collapse(self, state_belief: np.ndarray) -> np.ndarray:
        return np.bincount(self._owner, weights=state_belief, minlength=self._n_phases)

    def reset(self) -> None:
        """Resets belief to the fitted prior; call at episode boundaries."""
        self.state_belief = self._prior.copy()

    def step(self, features: np.ndarray) -> np.ndarray:
        """Consumes one frame's feature vector, returns the updated phase posterior."""
        predicted = self.state_belief @ self._transition
        # All sub-states of a phase share that phase's emission, so the
        # five-value likelihood is simply indexed out to sub-state space --
        # no extra emission parameters exist to evaluate.
        emission = _emission_likelihood(features, self.params)[self._owner]
        posterior = predicted * emission
        total = posterior.sum()
        self.state_belief = posterior / total if total > 1e-12 else predicted
        return self._collapse(self.state_belief)

    @classmethod
    def fit(cls, feature_seqs: list[np.ndarray], label_seqs: list[np.ndarray], *,
            std_floor_rel: float = DEFAULT_STD_FLOOR_REL,
            tie_covariance: bool = False,
            is_binary: np.ndarray | None = None,
            bernoulli_floor: float = DEFAULT_BERNOULLI_FLOOR,
            sub_states: np.ndarray | None = None,
            feature_names: list[str] | None = None) -> "PhaseHMMParams":
        """Estimates transition, prior, and emission parameters by counting
        and per-class MLE across labeled (feature_seq, label_seq) episode pairs.

        feature_seqs[i]: [T_i, n_features], label_seqs[i]: [T_i] Phase ints.
        NaN entries are treated as missing-at-random and excluded from that
        feature's per-class statistics (never imputed -- an imputed value is
        an invented observation, and inference skips the dimension anyway).

        is_binary defaults to features.IS_BINARY_FEATURE when the feature
        count matches it, so a caller that just passes the standard feature
        vector gets the Bernoulli treatment without having to ask; pass an
        explicit all-False mask to force the old all-Gaussian behaviour for
        an ablation.
        """
        n = Phase.N_CLASSES
        n_feat = feature_seqs[0].shape[1]

        if is_binary is None:
            is_binary = (IS_BINARY_FEATURE if n_feat == len(IS_BINARY_FEATURE)
                         else np.zeros(n_feat, dtype=bool))
        is_binary = np.asarray(is_binary, dtype=bool)
        if is_binary.shape[0] != n_feat:
            raise ValueError(f"is_binary has {is_binary.shape[0]} entries for {n_feat} features")

        trans_counts = np.full((n, n), _TRANS_SMOOTHING)
        prior_counts = np.full(n, _TRANS_SMOOTHING)
        for labels in label_seqs:
            prior_counts[labels[0]] += 1
            for a, b in zip(labels[:-1], labels[1:]):
                trans_counts[a, b] += 1
        transition = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        prior = prior_counts / prior_counts.sum()

        all_features = np.concatenate(feature_seqs, axis=0)
        all_labels = np.concatenate(label_seqs, axis=0)

        with np.errstate(invalid="ignore"):
            global_mean = np.nanmean(all_features, axis=0)
            global_std = np.nanstd(all_features, axis=0)
        global_mean = np.nan_to_num(global_mean, nan=0.0)          # a feature that is NaN everywhere
        global_std = np.maximum(np.nan_to_num(global_std, nan=1.0), _STD_FLOOR_ABS)

        emission_mean = np.zeros((n, n_feat))
        emission_var = np.zeros((n, n_feat))
        bernoulli_p = np.zeros((n, n_feat))
        class_counts = np.zeros((n, n_feat))
        for k in range(n):
            rows = all_features[all_labels == k]
            counts = np.isfinite(rows).sum(axis=0) if rows.size else np.zeros(n_feat)
            class_counts[k] = counts
            with np.errstate(invalid="ignore"):
                mean_k = np.nanmean(rows, axis=0) if rows.size else np.full(n_feat, np.nan)
                std_k = np.nanstd(rows, axis=0) if rows.size else np.full(n_feat, np.nan)
            # A class with no frames at all, or a feature never observed for
            # this class, falls back to the pooled statistics: an
            # uninformative emission, rather than a NaN that would poison the
            # whole posterior.
            emission_mean[k] = np.where(counts > 0, np.nan_to_num(mean_k, nan=0.0), global_mean)
            emission_var[k] = np.where(counts > 1, np.nan_to_num(std_k, nan=0.0), global_std) ** 2
            bernoulli_p[k] = np.where(counts > 0, np.nan_to_num(mean_k, nan=0.5), 0.5)

        if tie_covariance:
            # One pooled within-class variance shared by every class, so the
            # -sum log(sigma) normalizer is identical across classes and
            # cancels out of the posterior entirely. Weighted by how many
            # frames each class actually contributed to each feature.
            w = class_counts / np.maximum(class_counts.sum(axis=0, keepdims=True), 1.0)
            pooled_var = (w * emission_var).sum(axis=0)
            emission_var = np.tile(pooled_var, (n, 1))

        emission_std = np.sqrt(np.maximum(emission_var, 0.0))
        emission_std = np.maximum(emission_std, std_floor_rel * global_std)
        emission_std = np.maximum(emission_std, _STD_FLOOR_ABS)
        bernoulli_p = np.clip(bernoulli_p, bernoulli_floor, 1.0 - bernoulli_floor)

        return PhaseHMMParams(
            transition=transition, prior=prior,
            emission_mean=emission_mean, emission_std=emission_std,
            bernoulli_p=bernoulli_p, is_binary=is_binary,
            emission_temp=1.0, tie_covariance=tie_covariance,
            sub_states=(np.asarray(sub_states, dtype=int) if sub_states is not None
                        else np.ones(n, dtype=int)),
            feature_names=list(feature_names) if feature_names is not None else list(FEATURE_NAMES),
        )


def sub_states_from_durations(label_seqs: list[np.ndarray], n_classes: int, *,
                              max_sub_states: int = DEFAULT_MAX_SUB_STATES,
                              min_segments: int = 5) -> tuple[np.ndarray, dict]:
    """Chooses each phase's chain length from its observed segment durations.

    N_k = round(1 / CV_k^2), because a chain of N stages has CV = 1/sqrt(N):
    this is the shortest chain whose duration spread matches what the labels
    actually show. Three constraints are applied on top:

      * N >= 1 always, so a phase whose durations are already as irregular as
        a geometric (CV >= 1) is left completely untouched. Chains can only
        make a duration MORE regular, so there is nothing to gain there.
      * N <= max_sub_states. Past that point the transition matrix is being
        asked to encode a duration law that an explicit HSMM duration
        distribution would represent with fewer parameters -- that is the
        signal to stop stretching this model, not to stretch it further.
      * N < mean duration, since a chain of N stages takes at least N frames
        to traverse and cannot reproduce a mean shorter than itself.

    Phases with too few observed segments keep N = 1: guessing a duration law
    from three examples is worse than admitting there is no evidence.
    """
    durations: dict[int, list[int]] = {k: [] for k in range(n_classes)}
    for labels in label_seqs:
        for k, d in segment_durations(labels):
            if 0 <= k < n_classes:
                durations[k].append(d)

    chosen = np.ones(n_classes, dtype=int)
    info = {}
    for k, ds in durations.items():
        if len(ds) < min_segments:
            info[k] = {"n_segments": len(ds), "mean": float("nan"), "cv": float("nan"),
                       "sub_states": 1, "reason": "too few segments"}
            continue
        arr = np.asarray(ds, dtype=float)
        mean, cv = float(arr.mean()), float(arr.std() / max(arr.mean(), 1e-9))
        n = int(round(1.0 / max(cv, 1e-6) ** 2))
        n = max(1, min(n, max_sub_states, int(mean) - 1))
        chosen[k] = n
        reason = ("duration already geometric-like" if n == 1 else
                  "capped -- consider an explicit duration model" if n == max_sub_states else "")
        info[k] = {"n_segments": len(ds), "mean": mean, "cv": cv, "sub_states": n, "reason": reason}
    return chosen, info


def emission_loglik(x: np.ndarray, params: PhaseHMMParams) -> np.ndarray:
    """Per-class emission log-likelihood [n_classes] for one frame.

    Gaussian dimensions contribute the usual -z^2/2 - log(sigma); Bernoulli
    dimensions contribute x*log(p) + (1-x)*log(1-p). NaN dimensions
    contribute NOTHING AT ALL -- crucially including their normalizer term,
    so that whether a dimension happens to be observed this frame cannot by
    itself shift the posterior between classes. The dropped constant
    -0.5*log(2*pi) per Gaussian dimension is identical across classes within
    a frame and so cannot affect the normalised posterior.

    The result is scaled by emission_temp (see the module docstring).
    """
    x = np.asarray(x, dtype=np.float64)
    observed = np.isfinite(x)
    binary = params.binary_mask
    gauss_dims = observed & ~binary
    bern_dims = observed & binary

    ll = np.zeros(params.emission_mean.shape[0])

    if gauss_dims.any():
        mean = params.emission_mean[:, gauss_dims]
        std = params.emission_std[:, gauss_dims]
        z2 = ((x[gauss_dims][None, :] - mean) / std) ** 2
        ll += -0.5 * z2.sum(axis=1) - np.log(std).sum(axis=1)

    if bern_dims.any() and params.bernoulli_p is not None:
        p = params.bernoulli_p[:, bern_dims]
        xb = x[bern_dims][None, :]
        ll += (xb * np.log(p) + (1.0 - xb) * np.log1p(-p)).sum(axis=1)

    return ll * params.emission_temp


def _emission_likelihood(x: np.ndarray, params: PhaseHMMParams) -> np.ndarray:
    """Normalised (max-shifted) emission likelihood, safe to exponentiate."""
    ll = emission_loglik(x, params)
    ll = ll - ll.max()
    return np.exp(ll)


def filter_episode(params: PhaseHMMParams, features: np.ndarray) -> np.ndarray:
    """Runs the causal forward filter over one episode, returning the
    posterior at every step [T, n_classes]. Exactly what PhaseHMM.step does
    frame by frame -- shared so temperature selection, the eval harness and
    the diagnostics can never drift from the deployed recursion."""
    hmm = PhaseHMM(params)
    hmm.reset()
    return np.stack([hmm.step(f) for f in features])


def fit_emission_temperature(params: PhaseHMMParams, episodes: list[dict], *,
                             grid=DEFAULT_TEMP_GRID,
                             metric: str = "macro_f1") -> tuple[float, list[dict]]:
    """Picks the emission temperature that maximises `metric` on held-out
    episodes, returning (best_temp, per-temperature score table).

    Selection is on VAL, never train: temperature trades emission confidence
    against the transition prior, and on train (where the emission was fit)
    higher confidence always looks better. Default metric is macro_f1 rather
    than accuracy for the reasons in metrics.py -- an accuracy-selected
    temperature would happily keep the IDLE-collapse behaviour, since always
    predicting IDLE scores well on a set this imbalanced.

    PRIOR_ONLY_TEMP (0.0) is always scored and always excluded from selection:
    it is the floor this model reaches with the emission switched off, i.e.
    from transition and duration structure alone. Report the gap. A small gap
    means the phase estimate is mostly replaying the task's script rather than
    reading the robot, which frame-level accuracy will happily conceal on a
    stereotyped task -- and which would fail precisely on the deviations that
    make an intent module worth having.

    episodes: the same per-(episode, side) dicts train.py builds, needing
    only 'phase_features' and 'phase_labels'.
    """
    def score(temp: float) -> dict:
        trial = replace(params, emission_temp=float(temp))
        true, pred = [], []
        for d in episodes:
            post = filter_episode(trial, d["phase_features"])
            pred.extend(np.argmax(post, axis=1).tolist())
            true.extend(np.asarray(d["phase_labels"], dtype=int).tolist())
        m = classification_metrics(true, pred, Phase.N_CLASSES)
        return {"emission_temp": float(temp), "accuracy": m["accuracy"],
                "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"]}

    selectable = [score(t) for t in grid if t > 0]
    best = max(selectable, key=lambda r: (r[metric] if np.isfinite(r[metric]) else -np.inf))

    reference = score(PRIOR_ONLY_TEMP)
    reference["prior_only"] = True
    table = sorted(selectable + [reference], key=lambda r: r["emission_temp"])
    return best["emission_temp"], table

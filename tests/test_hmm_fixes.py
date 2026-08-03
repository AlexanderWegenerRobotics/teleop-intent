"""Regression tests for the two bugs that caused the v7 failure modes.

Each test names the symptom it locks down, because both bugs were invisible
in aggregate metrics -- v7's phase accuracy (0.74/0.69) looked like a model
that needed more data, and the target regression showed up only as a two-point
drop between checkpoint versions. Run:

    python -m pytest tests/test_hmm_fixes.py -v
    python tests/test_hmm_fixes.py          # no pytest needed

No hardware, no store-root, no teleop_orchestrator build required -- the
contracts are stubbed (tests/_stub_contracts.py) since everything under test
is filter arithmetic.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._stub_contracts import install, Phase, NULL_TARGET  # noqa: E402

install()

from models.hmm.phase import PhaseHMM, PhaseHMMParams, filter_episode, emission_loglik  # noqa: E402
from models.hmm.target import (TargetStickyFilter, TargetFilterParams,  # noqa: E402
                                _score, _score_grid)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# Target head: exclusion must remove a candidate, not promote it.
# --------------------------------------------------------------------------

def _params(**kw):
    base = dict(rho_loose=0.7, rho_tight=0.9, sigma=0.1, null_prior=0.2)
    base.update(kw)
    return TargetFilterParams(**base)


def test_excluded_candidate_loses_all_belief_with_gaze_on_it():
    """THE PLACE BUG. Operator carries an object and keeps looking at it. Once
    the object is excluded from the pool it must hold zero probability, even
    though the gaze evidence still points straight at it. The old code gave
    masked candidates a likelihood of 1.0, so this candidate stayed at ~0.99
    forever and the model reported 'targeting the object in the gripper'
    throughout the place phase."""
    px = np.array([[0.50, 0.50], [0.20, 0.30], [0.80, 0.30]])  # carried object, correct bin, other bin
    f = TargetStickyFilter(_params())
    f.reset()
    gaze = px[0] + np.array([0.02, 0.0])  # gaze parked on the carried object

    for t in range(60):
        mask = np.array([t < 20, True, True])  # excluded from t = 20 onward
        belief = f.step(gaze, px, mask, gaze_valid=True, phase_estimate=Phase.TRANSPORT)

    assert belief[0] == 0.0, f"excluded candidate kept {belief[0]:.3f} of the belief"
    assert np.isclose(belief.sum(), 1.0)
    assert int(np.argmax(belief[:3])) != 0


def test_excluded_candidate_zeroed_even_with_no_evidence():
    """Same bug via the other path: on frames with no evidence at all (gaze
    invalid, alignment disabled) the old code returned `predicted` untouched,
    so zeroing only the likelihood would not have been enough. The predict
    step must drop invalid states itself."""
    px = np.array([[0.5, 0.5], [0.2, 0.3]])
    f = TargetStickyFilter(_params())
    f.reset()
    f.step(np.array([0.5, 0.5]), px, np.array([True, True]), True, Phase.GRASP)
    for _ in range(10):
        belief = f.step(np.array([0.5, 0.5]), px, np.array([False, True]),
                        gaze_valid=False, phase_estimate=Phase.TRANSPORT)
    assert belief[0] == 0.0
    assert np.isclose(belief.sum(), 1.0)


def test_step_and_score_grid_agree():
    """THE ROOT CAUSE. rho/sigma/null_prior were fit by _score_grid but served
    by step(), and the two disagreed on what a masked candidate means. Any
    future divergence between the fitting recursion and the serving recursion
    breaks the fit silently, so assert they are numerically identical."""
    T, n = 80, 3
    ep = {
        "gaze_xy": RNG.random((T, 2)),
        "candidate_px": np.tile(RNG.random((n, 2)), (T, 1, 1)),
        "candidate_mask": np.stack([np.array([t < 30, True, t % 7 != 0]) for t in range(T)]),
        "gaze_valid": RNG.random(T) > 0.1,
        "phase_label": RNG.integers(0, Phase.N_CLASSES, T),
        "target_label": np.where(RNG.random(T) < 0.4, NULL_TARGET, RNG.integers(0, n, T)),
    }
    p = _params()
    scalar = _score(p, [ep])
    grid = _score_grid(np.array([p.rho_loose]), np.array([p.rho_tight]),
                       np.array([p.sigma]), np.array([p.null_prior]), [ep])
    assert np.isclose(scalar, grid[0], atol=1e-9), f"{scalar} != {grid[0]}"


def test_alignment_channel_does_not_leak_mass_to_null():
    """The alignment likelihood is <= 1 for every candidate while null used to
    get an implicit 1.0, so on gaze-invalid frames the channel pushed mass to
    null -- the opposite of its purpose. null_prior is now applied whenever
    any channel fires, keeping null on the same footing as the candidates."""
    px = np.array([[0.2, 0.3], [0.8, 0.3]])
    world = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    ee_pos = np.zeros(3)
    ee_vel = np.array([0.5, 0.0, 0.0])  # driving straight at candidate 0
    f = TargetStickyFilter(_params(sigma_align=0.3))
    f.reset()
    for _ in range(30):
        belief = f.step(np.zeros(2), px, np.array([True, True]), gaze_valid=False,
                        phase_estimate=Phase.APPROACH, ee_vel=ee_vel, ee_pos=ee_pos,
                        candidate_world_pos=world)
    assert int(np.argmax(belief)) == 0, f"alignment evidence did not win: {belief}"
    assert belief[0] > belief[-1]


# --------------------------------------------------------------------------
# Phase head: the emission must not have an opinion before it sees anything.
# --------------------------------------------------------------------------

# Per-phase statistics lifted from the fitted v7 checkpoint, so the fixture
# reproduces the real failure rather than a tidier version of it: is_holding
# rates from emission column 8, gripper widths from column 3, and the
# near-degenerate gripper-delta scale from column 4's per-class std (0.0010
# under IDLE against 0.0031 under GRASP -- the pair that made v7's absolute
# 1e-3 std floor useless). EE speed separability is deliberately poor,
# matching the measured 0.26 pooled-std gap between IDLE and GRASP.
_HOLD_RATE = {Phase.IDLE: 0.01, Phase.APPROACH: 0.002, Phase.GRASP: 0.24,
              Phase.TRANSPORT: 0.93, Phase.PLACE: 0.35}
_GRIPPER_W = {Phase.IDLE: 0.0795, Phase.APPROACH: 0.0791, Phase.GRASP: 0.0725,
              Phase.TRANSPORT: 0.0582, Phase.PLACE: 0.0706}
_DELTA_STD = {Phase.IDLE: 0.0010, Phase.APPROACH: 0.0014, Phase.GRASP: 0.0031,
              Phase.TRANSPORT: 0.0011, Phase.PLACE: 0.0027}
_EE_SPEED = {Phase.IDLE: 0.00, Phase.APPROACH: 0.05, Phase.GRASP: 0.01,
             Phase.TRANSPORT: 0.07, Phase.PLACE: 0.02}

_SEGMENTS = [(Phase.IDLE, 150), (Phase.APPROACH, 45), (Phase.GRASP, 18), (Phase.TRANSPORT, 50),
             (Phase.PLACE, 20), (Phase.IDLE, 90), (Phase.APPROACH, 40), (Phase.GRASP, 15),
             (Phase.TRANSPORT, 45), (Phase.PLACE, 18), (Phase.IDLE, 60)]

IS_BINARY_5 = np.array([False, True, False, False, False])


def _synthetic_phase_data(n_ep: int = 14, seed: int = 1):
    """Episodes shaped like the real ones: five features
    (ee_speed, is_holding, gripper_width, gripper_delta, noise), poor
    continuous separability, and a binary channel that is nearly pure under
    IDLE and APPROACH. That purity is what collapses a Gaussian class std and
    turns the normalizer into an unconditional class prior."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(n_ep):
        f, l = [], []
        for phase, dur in _SEGMENTS:
            for _t in range(dur):
                f.append([_EE_SPEED[phase] + rng.normal(0, 0.045),
                          float(rng.random() < _HOLD_RATE[phase]),
                          _GRIPPER_W[phase] + rng.normal(0, 0.010),
                          rng.normal(0, _DELTA_STD[phase]),
                          rng.normal(0, 1.0)])          # no class information at all
                l.append(phase)
        feats.append(np.array(f))
        labels.append(np.array(l))
    return feats, labels


def _redundant_phase_data(n_ep: int, seed: int, dup: int):
    """One informative but noisy signal, replicated `dup` times. A diagonal
    emission treats the copies as independent evidence and so is overconfident
    by a factor of `dup` -- a controlled stand-in for the real vector's three
    gripper-derived features and three velocity components."""
    rng = np.random.default_rng(seed)
    mean = {Phase.IDLE: 0.0, Phase.APPROACH: 1.0, Phase.GRASP: 1.6,
            Phase.TRANSPORT: 2.6, Phase.PLACE: 1.9}
    feats, labels = [], []
    for _ in range(n_ep):
        f, l = [], []
        for phase, dur in _SEGMENTS:
            for _t in range(dur):
                f.append([mean[phase] + rng.normal(0, 1.1)] * dup)
                l.append(phase)
        feats.append(np.array(f))
        labels.append(np.array(l))
    return feats, labels


def _causal_metrics(params, feats, labels):
    from metrics import classification_metrics
    true, pred = [], []
    for f, l in zip(feats, labels):
        pred.extend(np.argmax(filter_episode(params, f), axis=1).tolist())
        true.extend(np.asarray(l, dtype=int).tolist())
    return classification_metrics(true, pred, Phase.N_CLASSES)





def test_tie_covariance_cancels_the_normalizer_bias_exactly():
    """The diagnostic knob: one shared covariance makes the constant term
    identical for every class, so it cannot bias the posterior at all."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, tie_covariance=True, is_binary=IS_BINARY_5)
    assert np.ptp(p.normalizer_bias()) < 1e-9


def test_binary_feature_under_bernoulli_bounds_the_bias():
    """A pure binary feature must not blow up the constant term. Fit the same
    data twice -- once treating it as binary, once forcing it through a
    Gaussian as v7 did -- and require the Bernoulli fit to have the smaller
    normalizer spread."""
    feats, labels = _synthetic_phase_data()
    bern = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)
    gauss = PhaseHMM.fit(feats, labels, is_binary=np.zeros(5, dtype=bool))
    assert np.ptp(bern.normalizer_bias()) < np.ptp(gauss.normalizer_bias())


def test_relative_std_floor_bounds_normalizer_spread_per_feature():
    """std_floor_rel caps the per-feature spread at log(1/std_floor_rel) by
    construction; v7's absolute 1e-3 floor gave no such guarantee, which is
    how gripper_width_delta (real scale ~1e-3) contributed 1.15 nats."""
    feats, labels = _synthetic_phase_data()
    rel = 0.25
    p = PhaseHMM.fit(feats, labels, std_floor_rel=rel, is_binary=IS_BINARY_5)
    gauss_dims = ~p.binary_mask
    per_feat = -np.log(p.emission_std[:, gauss_dims])
    assert np.all(np.ptp(per_feat, axis=0) <= np.log(1.0 / rel) + 1e-9)


def _class_mean_frame(p, k):
    """The exact class mean, with binary dims rounded so the input stays a
    legal {0, 1} value rather than a probability the model can never see."""
    x = p.emission_mean[k].copy()
    if p.bernoulli_p is not None:
        x = np.where(p.binary_mask, np.round(p.bernoulli_p[k]), x)
    return x


def test_filter_is_never_absorbed_by_a_state():
    """THE IDLE-COLLAPSE BUG, stated precisely. v7's failure was not that it
    confused two similar phases -- it was ABSORPTION: fed the exact APPROACH
    mean it fell into IDLE within ~30 frames, reached belief 1.000, and then
    would not leave even when fed the exact GRASP mean for another 30. A
    filter that cannot be moved by unambiguous evidence is broken regardless
    of what its accuracy says.

    So the assertion is escape, not agreement: after a long IDLE stretch, the
    unmistakable TRANSPORT mean (the one phase every feature separates well)
    must pull the filter out within a bounded number of frames.

    Distinguishing this from ordinary confusion matters: GRASP and PLACE means
    remain hard for this feature set even after the emission is fixed (see
    test_class_mean_confusions_are_bounded), which is a feature-coverage
    limit, not an absorbing state -- a different problem with a different fix.
    """
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)

    frames = [_class_mean_frame(p, Phase.IDLE)] * 300 + [_class_mean_frame(p, Phase.TRANSPORT)] * 120
    pred = np.argmax(filter_episode(p, np.stack(frames)), axis=1)

    assert pred[299] == Phase.IDLE, "fixture should have settled in IDLE before the switch"
    escaped = np.nonzero(pred[300:] == Phase.TRANSPORT)[0]
    assert escaped.size > 0, "filter never left IDLE -- absorbing state, the v7 failure"
    assert escaped[0] < 60, f"took {escaped[0]} frames to leave IDLE on unambiguous evidence"
    assert pred[-1] == Phase.TRANSPORT


def test_class_mean_confusions_are_bounded():
    """Companion to the absorption test, and the honest limit of this feature
    set. Fed each class mean in turn, the fixed model tracks the phases its
    features actually separate (APPROACH, TRANSPORT) and must not have a
    single class swallow everything -- v7 did, predicting one class on nearly
    the whole sequence. GRASP's mean genuinely resembles IDLE's here (its
    is_holding rate rounds to 0 and its EE speed is 0.01 against IDLE's 0.00),
    so it is not asserted: that gap is a missing-feature problem, and the fix
    for it is a duration model or gripper history, not a better emission."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)

    order = [Phase.IDLE, Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT, Phase.PLACE]
    frames, truth = [], []
    for k in order:
        frames.extend([_class_mean_frame(p, k)] * 60)
        truth.extend([k] * 60)
    pred = np.argmax(filter_episode(p, np.stack(frames)), axis=1)
    truth = np.asarray(truth)

    for k in (Phase.APPROACH, Phase.TRANSPORT):
        seg = pred[truth == k]
        top = int(np.bincount(seg, minlength=Phase.N_CLASSES).argmax())
        assert top == k, f"fed the {Phase.NAMES[k]} mean, model said {Phase.NAMES[top]}"

    # Frame-level performance is asserted separately, on realistic (noisy)
    # frames, by test_fixed_emission_beats_the_v7_emission_on_realistic_data.
    # Class means are a harsher input than real frames -- they strip exactly
    # the per-frame variation GRASP and PLACE are actually recognised by -- so
    # they are used here only to detect absorption and gross collapse, never
    # as a performance measure.


def test_nan_feature_contributes_nothing_to_any_class():
    """Missing values must be skipped with their normalizer term, not just
    their Mahalanobis term. Dropping only the latter would reintroduce the
    very per-class constant this rework removes -- and it would fire on
    exactly the frames where a candidate is not visible."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)

    x = p.emission_mean[Phase.GRASP].copy()
    x = np.where(p.binary_mask, np.round(p.bernoulli_p[Phase.GRASP]), x)
    full = emission_loglik(x, p)

    x_missing = x.copy()
    x_missing[4] = np.nan               # the pure-noise dimension
    partial = emission_loglik(x_missing, p)

    # Dropping a dimension that carries no class information must not reorder
    # the classes, and must shift every class by the same amount.
    shift = full - partial
    assert np.ptp(shift) < 0.35, f"skipping a NaN dim shifted classes unevenly: {shift}"
    assert int(np.argmax(partial)) == int(np.argmax(full))


def test_missing_dims_never_change_the_argmax_between_classes():
    """Stronger version of the above with a feature that is missing for the
    whole episode: the posterior must be driven by the remaining features
    only, never by which class happened to have a tighter fit on the absent
    one."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)
    ll = emission_loglik(np.full(5, np.nan), p)
    assert np.ptp(ll) < 1e-12, f"an all-missing frame still had a class preference: {ll}"


def test_emission_temperature_scales_log_likelihood():
    """Temperature must act on the log-likelihood (a true tempering), not on
    the probability, or the transition prior is not actually given more room."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5)
    x = p.emission_mean[Phase.GRASP]
    from dataclasses import replace
    hot, cold = emission_loglik(x, p), emission_loglik(x, replace(p, emission_temp=0.25))
    assert np.allclose(cold, 0.25 * hot)


def test_temperature_recovers_performance_lost_to_redundant_features():
    """WHY THE TEMPERATURE EXISTS. Replicate one noisy signal eight times and a
    diagonal emission counts it as eight independent observations: it becomes
    overconfident, overwhelms the transition prior, and the filter chatters.
    The val-selected temperature must claw back most of what the redundancy
    cost, measured against the same data with a single copy."""
    from dataclasses import replace
    from models.hmm.phase import fit_emission_temperature

    def fit_and_score(dup):
        tr_f, tr_l = _redundant_phase_data(14, 1, dup)
        va_f, va_l = _redundant_phase_data(5, 3, dup)
        te_f, te_l = _redundant_phase_data(6, 2, dup)
        p = PhaseHMM.fit(tr_f, tr_l, is_binary=np.zeros(dup, dtype=bool))
        val = [{"phase_features": f, "phase_labels": l} for f, l in zip(va_f, va_l)]
        temp, _ = fit_emission_temperature(p, val)
        hot = _causal_metrics(replace(p, emission_temp=1.0), te_f, te_l)["macro_f1"]
        tuned = _causal_metrics(replace(p, emission_temp=temp), te_f, te_l)["macro_f1"]
        return temp, hot, tuned

    temp1, hot1, tuned1 = fit_and_score(1)
    temp8, hot8, tuned8 = fit_and_score(8)

    # With no redundancy the sweep must leave the emission alone.
    assert temp1 == 1.0, f"temperature fired on non-redundant features (picked {temp1})"
    # With redundancy it must fire, and recover most of the lost performance.
    assert temp8 < 1.0, "temperature did not fire on eight perfectly correlated copies"
    assert tuned8 > hot8 + 0.1, f"temperature recovered too little: {hot8:.3f} -> {tuned8:.3f}"
    assert tuned8 > 0.9 * tuned1, (
        f"tempered redundant fit ({tuned8:.3f}) should approach the non-redundant "
        f"ceiling ({tuned1:.3f})")


def test_fixed_emission_beats_the_v7_emission_on_realistic_data():
    """END-TO-END. Same data, same transitions, same features -- only the
    emission model differs. The pre-v8 configuration (every feature Gaussian,
    absolute std floor) must lose badly, and specifically must under-predict
    at least one phase, which is the signature that showed up as GRASP 0.15 /
    PLACE 0.06 recall on the real test split."""
    tr_f, tr_l = _synthetic_phase_data(14, seed=1)
    te_f, te_l = _synthetic_phase_data(6, seed=2)

    v7 = PhaseHMM.fit(tr_f, tr_l, is_binary=np.zeros(5, dtype=bool), std_floor_rel=0.0)
    v8 = PhaseHMM.fit(tr_f, tr_l, is_binary=IS_BINARY_5)

    m7, m8 = _causal_metrics(v7, te_f, te_l), _causal_metrics(v8, te_f, te_l)

    assert np.ptp(v7.normalizer_bias()) > 2.0, "fixture no longer reproduces the v7 bias"
    assert np.ptp(v8.normalizer_bias()) < 2.0, f"bias still large: {v8.normalizer_bias()}"
    assert m8["macro_f1"] > m7["macro_f1"] + 0.2, f"{m7['macro_f1']:.3f} -> {m8['macro_f1']:.3f}"
    assert m7["per_class_recall"].min() < 0.5, "fixture no longer starves a phase under v7"
    assert m8["per_class_recall"].min() > 0.6, (
        f"a phase is still starved: {dict(zip(Phase.NAMES.values(), m8['per_class_recall'].round(2)))}")


# --------------------------------------------------------------------------
# Sub-state chains: the duration model must be what the maths says it is.
# --------------------------------------------------------------------------

def _base_prior():
    """v9's fitted prior, renormalised for the same transcription reason as
    _base_transition: the printed four-decimal values sum to 1.0001, and
    expand_chains normalises what it is given."""
    p = np.array([0.9669, 0.0083, 0.0083, 0.0083, 0.0083])
    return p / p.sum()


def _base_transition():
    """The real v9 transition matrix, so these tests exercise the actual
    self-transitions the duration work is aimed at.

    Rows are renormalised because these values are transcribed from a printed
    checkpoint at four decimals and so do not sum to exactly 1 (row 2 sums to
    0.9999). A genuinely fitted matrix is row-stochastic by construction --
    counts divided by their row sum -- and expand_chains reproduces such a
    matrix exactly when the chain lengths are all one, which is what
    test_all_ones_chain_is_exactly_the_plain_model asserts.
    """
    T = np.array([[0.9989, 0.0011, 0, 0, 0],
                  [0.0001, 0.9926, 0.0072, 0.0002, 0],
                  [0, 0.0001, 0.99, 0.0098, 0],
                  [0.0001, 0.0004, 0, 0.9891, 0.0104],
                  [0.012, 0.0055, 0.0002, 0.0001, 0.9822]])
    return T / T.sum(axis=1, keepdims=True)


def _exact_duration_moments(B, idx):
    """(mean, CV) of the time to leave a chain, computed exactly from the
    constructed matrix rather than simulated: E[D] = sum_t P(D>t) and
    E[D^2] = sum_t (2t+1) P(D>t), with P(D>t) read off the in-chain
    sub-matrix. Tests the matrix that will actually run, not the formula it
    was built from."""
    M = B[np.ix_(idx, idx)]
    v = np.zeros(len(idx))
    v[0] = 1.0
    s0 = s1 = 0.0
    t = 0
    while t < 500000:
        p = v.sum()
        if p < 1e-13:
            break
        s0 += p
        s1 += (2 * t + 1) * p
        v = v @ M
        t += 1
    return s0, np.sqrt(max(s1 - s0 ** 2, 0.0)) / s0


def test_expanded_transition_is_a_valid_markov_chain():
    from models.hmm.phase import expand_chains
    T = _base_transition()
    B, pr, owner = expand_chains(T, _base_prior(), np.array([1, 5, 1, 10, 4]))
    assert B.shape == (21, 21)
    assert np.allclose(B.sum(axis=1), 1.0)
    assert (B >= 0).all()
    assert np.isclose(pr.sum(), 1.0)
    assert list(np.bincount(owner)) == [1, 5, 1, 10, 4]


def test_chains_preserve_mean_duration_and_deliver_the_target_cv():
    """The whole point. Chaining N sub-states must leave each phase's MEAN
    duration exactly where the counted transition matrix put it, and pull the
    coefficient of variation down to ~1/sqrt(N) -- from the ~1.0 a geometric
    forces. If the mean moved, the comparison against the plain model would
    be confounded by something other than duration shape."""
    from models.hmm.phase import expand_chains
    T = _base_transition()
    N = np.array([1, 5, 1, 10, 4])
    B, _pr, owner = expand_chains(T, np.full(5, 0.2), N)
    offset = np.concatenate([[0], np.cumsum(N)[:-1]])

    for k in range(5):
        idx = np.arange(offset[k], offset[k] + N[k])
        mean, cv = _exact_duration_moments(B, idx)
        plain_mean = 1.0 / (1.0 - T[k, k])
        assert np.isclose(mean, plain_mean, rtol=1e-6), (
            f"{Phase.NAMES[k]}: mean moved {plain_mean:.1f} -> {mean:.1f}")
        assert np.isclose(cv, 1.0 / np.sqrt(N[k]), atol=0.02), (
            f"{Phase.NAMES[k]}: CV {cv:.3f}, expected ~{1/np.sqrt(N[k]):.3f}")


def test_chains_preserve_where_a_phase_goes_next():
    """Chains change WHEN a phase ends, never WHERE it goes. The exit row of
    each chain must reproduce the base matrix's off-diagonal distribution."""
    from models.hmm.phase import expand_chains
    T = _base_transition()
    N = np.array([1, 5, 1, 10, 4])
    B, _pr, _owner = expand_chains(T, np.full(5, 0.2), N)
    offset = np.concatenate([[0], np.cumsum(N)[:-1]])
    for k in range(5):
        last = offset[k] + N[k] - 1
        others = [l for l in range(5) if l != k]
        got = np.array([B[last, offset[l]] for l in others])
        want = T[k, others]
        assert np.allclose(got / got.sum(), want / want.sum(), atol=1e-12)


def test_all_ones_chain_is_exactly_the_plain_model():
    """The ablation flag must be a true no-op, or `--sub-states 1,1,1,1,1`
    cannot serve as the control condition."""
    from models.hmm.phase import expand_chains
    T = _base_transition()
    prior = _base_prior()
    B, pr, owner = expand_chains(T, prior, np.ones(5, dtype=int))
    assert np.allclose(B, T)
    assert np.allclose(pr, prior)
    assert list(owner) == list(range(5))


def test_step_returns_a_five_phase_posterior_whatever_the_chain_length():
    """Sub-states are an internal detail: every caller keeps seeing the
    five-phase contract, and the posterior still normalises."""
    feats, labels = _synthetic_phase_data()
    p = PhaseHMM.fit(feats, labels, is_binary=IS_BINARY_5, sub_states=np.array([1, 3, 1, 6, 4]))
    hmm = PhaseHMM(p)
    hmm.reset()
    for x in feats[0][:200]:
        post = hmm.step(x)
        assert post.shape == (Phase.N_CLASSES,)
        assert np.isclose(post.sum(), 1.0)
    assert hmm.state_belief.shape == (15,)


def test_chain_sizing_follows_the_observed_durations():
    """Auto-sizing must give regular phases a chain and leave irregular ones
    alone -- chains can only reduce spread, so a phase already at CV >= 1 has
    nothing to gain and must come back with N = 1."""
    from models.hmm.phase import sub_states_from_durations
    rng = np.random.default_rng(3)
    seqs = []
    for _ in range(40):
        seq = []
        for _cycle in range(3):
            seq += [Phase.IDLE] * int(rng.geometric(1 / 200))       # CV ~ 1
            seq += [Phase.APPROACH] * max(2, int(rng.normal(120, 24)))   # CV ~ 0.20
            seq += [Phase.GRASP] * int(rng.geometric(1 / 90))        # CV ~ 1
            seq += [Phase.TRANSPORT] * max(2, int(rng.normal(100, 50)))  # CV ~ 0.50
            seq += [Phase.PLACE] * max(2, int(rng.normal(55, 11)))       # CV ~ 0.20
        seqs.append(np.array(seq))
    n, info = sub_states_from_durations(seqs, Phase.N_CLASSES)

    assert n[Phase.IDLE] == 1, f"idle got {n[Phase.IDLE]} chains for CV {info[Phase.IDLE]['cv']:.2f}"
    assert n[Phase.GRASP] == 1, f"grasp got {n[Phase.GRASP]} chains for CV {info[Phase.GRASP]['cv']:.2f}"
    assert n[Phase.APPROACH] >= 8
    assert n[Phase.PLACE] >= 8
    assert 2 <= n[Phase.TRANSPORT] <= 8
    # A chain can never be longer than the mean duration it has to reproduce.
    for k, d in info.items():
        if np.isfinite(d["mean"]):
            assert n[k] < d["mean"]


def test_chain_sizing_ignores_phases_with_too_little_evidence():
    from models.hmm.phase import sub_states_from_durations
    seqs = [np.array([Phase.IDLE] * 50 + [Phase.GRASP] * 20 + [Phase.IDLE] * 50)]
    n, _info = sub_states_from_durations(seqs, Phase.N_CLASSES)
    assert (n == 1).all(), "sized a duration law from a handful of segments"


def test_chains_reduce_segment_chattering():
    """The behavioural claim: a duration model should stop the belief flapping
    inside a segment. Measured as predicted-segment count against the same
    features and the same emission."""
    from metrics import segments as seg
    feats, labels = _synthetic_phase_data()
    common = dict(is_binary=IS_BINARY_5)
    plain = PhaseHMM.fit(feats, labels, sub_states=np.ones(5, dtype=int), **common)
    chained = PhaseHMM.fit(feats, labels, sub_states=np.array([1, 4, 1, 4, 4]), **common)

    def n_segments(params):
        return sum(len(seg(np.argmax(filter_episode(params, f), axis=1))) for f in feats)

    assert n_segments(chained) <= n_segments(plain)


# --------------------------------------------------------------------------
# End-to-end plumbing: fit -> save -> load must round-trip the new fields.
# --------------------------------------------------------------------------

def _fake_training_dicts(n_ep=8, seed=5, n_cand=3):
    """Per-(episode, side) dicts in the shape train.build_episode_arrays
    produces, with the real feature width so model.load()'s feature-count
    guard is exercised for real."""
    from models.hmm.features import FEATURE_NAMES
    rng = np.random.default_rng(seed)
    feats, labels = _synthetic_phase_data(n_ep, seed)
    out = []
    for f, l in zip(feats, labels):
        T = len(l)
        wide = np.concatenate([f, rng.normal(0, 1, (T, len(FEATURE_NAMES) - f.shape[1]))], axis=1)
        # A realistic dose of missing values in the two distance columns.
        miss = rng.random(T) < 0.15
        wide[miss, 6] = np.nan
        wide[miss, 7] = np.nan
        out.append({
            "phase_features": wide, "phase_labels": l,
            "gaze_xy": rng.random((T, 2)),
            "candidate_px": np.tile(rng.random((n_cand, 2)), (T, 1, 1)),
            "candidate_mask": np.ones((T, n_cand), dtype=bool),
            "gaze_valid": rng.random(T) > 0.05,
            "target_labels": np.where(rng.random(T) < 0.5, NULL_TARGET, rng.integers(0, n_cand, T)),
        })
    return out


def test_fit_save_load_roundtrip(tmp_path=None):
    """The v8 emission fields must survive a checkpoint round trip. A silently
    dropped emission_temp or is_binary would load as the v7 defaults and undo
    the whole fix at scoring time without raising anything."""
    import tempfile
    from dataclasses import replace
    from models.hmm.model import HMMIntentModel

    data = _fake_training_dicts()
    train, val = data[:6], data[6:]

    model = HMMIntentModel()
    history = model.fit(train, val, config={"emission_temp": "auto", "sub_states": [1, 3, 1, 4, 2]})

    assert "emission_temp_table" in history
    assert history["sub_states"] == [1, 3, 1, 4, 2]
    assert "val_phase_macro_f1" in history
    assert np.isfinite(history["normalizer_bias_spread"])

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v8.npz")
        model.save(path)
        reloaded = HMMIntentModel()
        reloaded.load(path)

    a, b = model.phase_params, reloaded.phase_params
    assert np.allclose(a.transition, b.transition)
    assert np.allclose(a.emission_mean, b.emission_mean)
    assert np.allclose(a.emission_std, b.emission_std)
    assert np.allclose(a.bernoulli_p, b.bernoulli_p)
    assert np.array_equal(a.binary_mask, b.binary_mask)
    assert a.emission_temp == b.emission_temp
    assert list(a.chain_lengths) == list(b.chain_lengths)
    assert a.feature_names == b.feature_names

    x = _class_mean_frame(a, Phase.GRASP)
    assert np.allclose(emission_loglik(x, a), emission_loglik(x, b))


def test_load_rejects_a_stale_feature_width():
    """v1-v7 were fit on 8 or 9 features; features.py now emits more. Loading
    one of those against the current extractor would silently misalign every
    emission column, so it must raise instead."""
    import tempfile
    from models.hmm.model import HMMIntentModel

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.npz")
        np.savez(path, transition=np.full((5, 5), 0.2), prior=np.full(5, 0.2),
                 emission_mean=np.zeros((5, 9)), emission_std=np.ones((5, 9)),
                 rho_loose=0.7, rho_tight=0.9, sigma=0.1, null_prior=0.2, sigma_align=np.inf)
        try:
            HMMIntentModel().load(path)
        except ValueError as e:
            assert "phase features" in str(e)
        else:
            raise AssertionError("stale checkpoint loaded without complaint")


def test_gaussian_only_config_reproduces_the_old_emission():
    """The ablation switch must actually switch: with gaussian_only the fit
    must contain no Bernoulli dimensions, so 'is the Bernoulli treatment
    doing the work?' can be answered by rerunning rather than by argument."""
    from models.hmm.model import HMMIntentModel
    data = _fake_training_dicts()
    m = HMMIntentModel()
    m.fit(data[:6], data[6:], config={"gaussian_only": True, "emission_temp": 1.0})
    assert not m.phase_params.binary_mask.any()
    assert m.phase_params.emission_temp == 1.0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n      {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


# --------------------------------------------------------------------------
# Phase-gated held-object exclusion (the 10.8% unreachable-label bug).
# --------------------------------------------------------------------------

class _HeldFrame:
    """Minimal frame for held.py: one object candidate, one bin, holding."""

    def __init__(self, holding=True):
        from tests._stub_contracts import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, CAND_OBJECT, CAND_BIN
        self.candidate_mask = np.ones(2, dtype=bool)
        self.candidate_types = np.array([CAND_OBJECT, CAND_BIN])
        self.candidate_features = np.zeros((2, len(CANDIDATE_FEATURE_NAMES)))
        self.global_features = np.zeros(len(GLOBAL_FEATURE_NAMES))
        self.grasp_confirmed = {"left": holding, "right": holding}


def test_exclusion_does_not_fire_during_grasp_or_approach():
    """THE 10.8% BUG. grasp_confirmed goes true while the fingers are still
    settling, so the old unconditional rule removed the object from the pool
    during GRASP -- the phase whose whole purpose is acting on that object.
    6434 of 9952 unreachable training labels were GRASP frames."""
    from models.hmm.held import target_exclusion_mask
    f = _HeldFrame(holding=True)
    for phase in (Phase.IDLE, Phase.APPROACH, Phase.GRASP):
        assert not target_exclusion_mask(f, "left", phase).any(), (
            f"objects excluded during {Phase.NAMES[phase]} while holding")


def test_exclusion_still_fires_while_carrying():
    """The rule must survive where it was right: once transporting or placing,
    a full gripper genuinely is not reaching for an object."""
    from models.hmm.held import target_exclusion_mask
    f = _HeldFrame(holding=True)
    for phase in (Phase.TRANSPORT, Phase.PLACE):
        m = target_exclusion_mask(f, "left", phase)
        assert m[0] and not m[1], f"expected object excluded, bin kept, in {Phase.NAMES[phase]}"


def test_exclusion_never_fires_when_not_holding():
    from models.hmm.held import target_exclusion_mask
    f = _HeldFrame(holding=False)
    for phase in range(Phase.N_CLASSES):
        assert not target_exclusion_mask(f, "left", phase).any()


def test_phase_none_reproduces_the_old_unconditional_rule():
    """Kept so an old checkpoint can be re-scored under the rule it was fitted
    with. Must be the pre-fix behaviour exactly, not an approximation."""
    from models.hmm.held import target_exclusion_mask
    m = target_exclusion_mask(_HeldFrame(holding=True), "left", None)
    assert m[0] and not m[1]
    assert not target_exclusion_mask(_HeldFrame(holding=False), "left", None).any()

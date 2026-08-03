"""Causal recurrent intent model: a GRU trunk with a phase head and a
permutation-equivariant target head, behind the same IntentModel contract as
models.hmm.

Importing this package does not require PyTorch -- model.py degrades to a
clear ImportError only when a network is actually built, so `eval/score.py`
and the orchestrator keep working on a machine that only ever runs the HMM.

See model.py for the architecture and why each piece is shaped the way it is,
features.py for why the arm features are imported from models.hmm rather than
reimplemented, and train.py for the training decisions carried over from the
HMM work (class-weighted loss, selection on val macro F1, truncated BPTT, and
the shuffled-feature control).
"""

from .model import GRUIntentModel

__all__ = ["GRUIntentModel"]

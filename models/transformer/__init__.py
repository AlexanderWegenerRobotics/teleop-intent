"""Causal windowed-attention intent model, behind the same IntentModel
contract as models.hmm and models.gru.

Importing this package does not require PyTorch -- model.py raises a clear
ImportError only when a network is actually built, so a machine that only runs
the HMM is unaffected.

See model.py for the architecture and what the experiment is testing,
train.py for the controls carried over from the GRU (shuffled features,
window=1 as the memoryless equivalent, class-weighted loss, selection on val
macro F1). Features are imported from models.gru so all three models provably
consume identical numbers.
"""

from .model import TransformerIntentModel

__all__ = ["TransformerIntentModel"]

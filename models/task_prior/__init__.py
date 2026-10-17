"""Task prior: p(next goal | which goals are already filled), learned from labeled episodes.

The perceptual models (HMM, GRU, transformer) stay task-agnostic; this is the one place task
order lives, and it is combined with their output in models/composed.py, never fed into them.
"""

from .state import TaskStateTracker, state_key
from .model import TaskPrior

__all__ = ["TaskStateTracker", "state_key", "TaskPrior"]

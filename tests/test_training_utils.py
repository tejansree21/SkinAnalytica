"""
Regression tests for src/training_utils.py's resolve_resume_epoch(), which
replaced the SA01_EfficientNet_Train.ipynb inline resume logic that once
silently trained zero epochs after a stale checkpoint survived a data-mixing
retrain (see the function's own docstring and docs/MODEL_CARD.md finding #10
for the incident).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training_utils import resolve_resume_epoch


def test_resumes_from_next_epoch_when_checkpoint_is_partial():
    assert resolve_resume_epoch(checkpoint_epoch=9, max_epochs=30) == 10


def test_resumes_from_epoch_zero_equivalent_after_single_completed_epoch():
    assert resolve_resume_epoch(checkpoint_epoch=0, max_epochs=30) == 1


def test_raises_instead_of_silently_training_zero_epochs():
    """The exact incident: a checkpoint already at MAX_EPOCHS-1 (i.e. fully
    complete) must not be allowed to silently resume into an empty range."""
    with pytest.raises(ValueError, match="silently train zero new epochs"):
        resolve_resume_epoch(checkpoint_epoch=29, max_epochs=30)


def test_raises_when_checkpoint_epoch_exceeds_max_epochs():
    """Covers the actual incident's numbers: a stale checkpoint from a run
    with a higher/older MAX_EPOCHS than the current config."""
    with pytest.raises(ValueError, match="silently train zero new epochs"):
        resolve_resume_epoch(checkpoint_epoch=30, max_epochs=30)


def test_error_message_names_both_epoch_numbers():
    with pytest.raises(ValueError, match=r"checkpoint epoch 30.*epoch 31.*MAX_EPOCHS=30"):
        resolve_resume_epoch(checkpoint_epoch=30, max_epochs=30)

"""
SkinAnalytica -- src/training_utils.py
Shared, testable helpers for the SA01*_Train.ipynb notebooks. Exists
because the notebooks themselves aren't unit-testable directly, and one
piece of their logic already caused a real, silent failure once -- see
resolve_resume_epoch()'s docstring.
"""


def resolve_resume_epoch(checkpoint_epoch: int, max_epochs: int) -> int:
    """
    Given a checkpoint's last completed epoch and the configured
    MAX_EPOCHS, return the epoch to resume training from.

    Raises ValueError if resuming would be a silent no-op (start_epoch >=
    max_epochs), instead of letting the training loop's `for epoch in
    range(start_epoch, max_epochs)` silently iterate zero times. This is
    exactly the bug that hit SA01_EfficientNet_Train.ipynb during this
    project's data-mixing retrain: a checkpoint from a PRE-retrain run
    (epoch 30/30, complete under the old ~68K-image pool) was still on
    disk when the training data changed to ~73K images including new
    EMB/MILK10k/SLICE-3D sources. start_epoch became 31, MAX_EPOCHS was
    30, `range(31, 30)` is empty, and the loop printed "Done. Best
    val_auc: 0.9692" using the OLD checkpoint's stale metrics -- without
    training a single new epoch on the new data. Caught by a human
    reading the output and noticing "No checkpoint at ..." was missing;
    this function makes that failure loud instead of silent.
    """
    start_epoch = checkpoint_epoch + 1
    if start_epoch >= max_epochs:
        raise ValueError(
            f"Resuming from checkpoint epoch {checkpoint_epoch} would start "
            f"at epoch {start_epoch}, which is >= MAX_EPOCHS={max_epochs} -- "
            "this checkpoint is already complete for the configured "
            "MAX_EPOCHS, so resuming would silently train zero new epochs. "
            "If the training DATA changed since this checkpoint was saved "
            "(a new dataset was mixed in, samples were re-split, etc.), "
            "delete the checkpoint files before re-running training -- "
            "resuming here would report stale, misleading results. If "
            "MAX_EPOCHS was intentionally lowered and this checkpoint is "
            "genuinely done, that's fine, but the notebook must not "
            "silently 'finish' without making that explicit."
        )
    return start_epoch

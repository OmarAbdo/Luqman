import numpy as np
import pytest

from luqman.splits.walk_forward import WalkForwardSplitter, chronological_holdout


def test_folds_are_chronological_and_disjoint():
    splitter = WalkForwardSplitter(n_splits=5, test_size=63, embargo=1, min_train_size=252)
    n = 1000
    folds = list(splitter.split(n))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_embargo_gap():
    splitter = WalkForwardSplitter(n_splits=3, test_size=50, embargo=5, min_train_size=100)
    for train_idx, test_idx in splitter.split(400):
        assert test_idx.min() - train_idx.max() - 1 == 5


def test_test_windows_tile_the_end():
    splitter = WalkForwardSplitter(n_splits=3, test_size=50, embargo=1, min_train_size=100)
    folds = list(splitter.split(400))
    assert folds[-1][1][-1] == 399
    all_test = np.concatenate([test for _, test in folds])
    assert len(np.unique(all_test)) == 150


def test_too_few_rows_raises():
    splitter = WalkForwardSplitter(n_splits=5, test_size=63, embargo=1, min_train_size=252)
    with pytest.raises(ValueError):
        list(splitter.split(300))


def test_holdout():
    train_idx, test_idx = chronological_holdout(100, test_fraction=0.2, embargo=2)
    assert test_idx.min() == 80
    assert train_idx.max() == 77
    assert test_idx.max() == 99

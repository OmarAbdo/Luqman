from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Expanding-window walk-forward splits over a chronologically ordered series.

    Test windows of `test_size` rows tile the end of the series; each fold trains
    on everything before its test window minus an `embargo` gap (>= target horizon,
    so no training row's forward-looking target overlaps the test window).
    """

    n_splits: int = 5
    test_size: int = 63  # ~ one quarter of trading days
    embargo: int = 1
    min_train_size: int = 252

    def split(self, n: int):
        needed = self.min_train_size + self.embargo + self.n_splits * self.test_size
        if n < needed:
            raise ValueError(
                f"{n} rows is too few for {self.n_splits} folds "
                f"(need >= {needed}: min_train {self.min_train_size} + embargo "
                f"{self.embargo} + {self.n_splits} x test_size {self.test_size})"
            )
        for k in range(self.n_splits):
            test_end = n - (self.n_splits - 1 - k) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.embargo
            yield np.arange(0, train_end), np.arange(test_start, test_end)


def chronological_holdout(n: int, test_fraction: float = 0.2, embargo: int = 1):
    """Single ordered train/test split with an embargo gap."""
    test_start = int(n * (1 - test_fraction))
    return np.arange(0, test_start - embargo), np.arange(test_start, n)

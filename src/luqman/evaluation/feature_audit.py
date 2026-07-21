"""Per-feature information-content audit.

Computes each feature's rank correlation (IC) with forward returns, per
chronological block and over the whole sample, plus a sign-consistency check
that separates a stable relationship from a lucky fit.

This module produced the project's most important negative result: all 24
features of the original set had |IC| below 0.06, an order of magnitude under
what is considered actionable. That explained why three different model
families had failed identically -- there was no signal for any of them to find.
"""

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(feature: pd.Series, target: pd.Series) -> float:
    """Spearman rank correlation between a feature and the forward return
    target — the standard 'IC' measure of a single feature's raw predictive
    content, independent of any model's ability to exploit it."""
    aligned = pd.concat([feature, target], axis=1).dropna()
    if len(aligned) < 3 or aligned.iloc[:, 0].nunique() < 2:
        return float("nan")
    rho, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(rho)


def audit_features(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Per-feature IC computed independently within each of `n_splits`
    chronological blocks, then aggregated. A feature with real, stable signal
    should show an IC of consistent sign and non-trivial magnitude across
    blocks; a feature that is pure noise will show ICs near zero with random
    sign — which is what any model trained on it will also find, no matter
    how sophisticated. This isolates 'does the data contain signal' from
    'can this model architecture find it'.
    """
    n = len(X)
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    rows = []
    for col in X.columns:
        ics = [
            information_coefficient(X[col].iloc[bounds[i]:bounds[i + 1]], y.iloc[bounds[i]:bounds[i + 1]])
            for i in range(n_splits)
        ]
        ics = np.array(ics, dtype=float)
        valid = ics[np.isfinite(ics)]
        rows.append(
            {
                "feature": col,
                "ic_full": information_coefficient(X[col], y),
                "ic_mean_blocks": float(np.mean(valid)) if len(valid) else float("nan"),
                "ic_std_blocks": float(np.std(valid)) if len(valid) else float("nan"),
                "ic_sign_consistency": float(np.mean(np.sign(valid) == np.sign(np.mean(valid))))
                if len(valid) else float("nan"),
            }
        )
    result = pd.DataFrame(rows).set_index("feature")
    return result.reindex(result["ic_full"].abs().sort_values(ascending=False).index)

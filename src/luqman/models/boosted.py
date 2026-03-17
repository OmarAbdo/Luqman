from sklearn.ensemble import HistGradientBoostingRegressor


def make_hist_gbm(seed: int = 42) -> HistGradientBoostingRegressor:
    """Gradient-boosted trees on the tabular feature set. Seeded for reproducibility;
    conservative leaf size and L2 to resist fitting return noise."""
    return HistGradientBoostingRegressor(
        random_state=seed,
        max_iter=400,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
    )

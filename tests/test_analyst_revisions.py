import numpy as np
import pandas as pd

from luqman.features.analyst_revisions import build_analyst_revision_features, net_rating_score

DAYS = pd.bdate_range("2023-01-02", periods=260)


def _grades(months, strong_buy, buy, hold, sell, strong_sell):
    return pd.DataFrame(
        {
            "month": pd.to_datetime(months),
            "strong_buy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strong_sell": strong_sell,
        }
    )


def test_net_rating_score_formula():
    grades = _grades(["2023-01-01"], [10], [5], [5], [0], [0])
    score = net_rating_score(grades)
    # (2*10 + 5 - 0 - 0) / 20 = 25/20 = 1.25
    assert np.isclose(score.iloc[0], 1.25)


def test_net_rating_score_bearish_case():
    grades = _grades(["2023-01-01"], [0], [0], [5], [10], [5])
    score = net_rating_score(grades)
    # (0 + 0 - 10 - 10) / 20 = -1.0
    assert np.isclose(score.iloc[0], -1.0)


def test_net_rating_score_neutral_all_hold():
    grades = _grades(["2023-01-01"], [0], [0], [20], [0], [0])
    score = net_rating_score(grades)
    assert np.isclose(score.iloc[0], 0.0)


def test_revision_reflects_improving_sentiment():
    grades = _grades(
        ["2023-01-01", "2023-02-01", "2023-03-01"],
        [5, 8, 12], [5, 5, 5], [10, 7, 3], [0, 0, 0], [0, 0, 0],
    )
    feat = build_analyst_revision_features(grades, DAYS, availability_lag_days=3)
    valid_revisions = feat["analyst_rating_revision"].dropna()
    assert (valid_revisions > 0).any()  # sentiment improving month over month


def test_availability_lag_respected():
    grades = _grades(["2023-01-01", "2023-02-01"], [20, 0], [0, 0], [0, 0], [0, 20], [0, 0])
    feat = build_analyst_revision_features(grades, DAYS, availability_lag_days=5)
    # Jan 2023 score (fully bullish, 2.0) shouldn't leak before ~Jan 6
    before = feat.loc[:"2023-01-05", "analyst_net_rating"]
    assert (before.dropna() != 2.0).all()
    after = feat.loc["2023-01-09":"2023-01-25", "analyst_net_rating"]
    assert (after.dropna() == 2.0).any()


def test_no_lookahead():
    grades = _grades(
        ["2023-01-01", "2023-02-01", "2023-03-01"],
        [5, 5, 20], [5, 5, 0], [10, 10, 0], [0, 0, 0], [0, 0, 0],
    )
    full = build_analyst_revision_features(grades, DAYS)
    truncated = build_analyst_revision_features(grades.iloc[:2], DAYS)
    cutoff = pd.Timestamp("2023-02-15")
    pd.testing.assert_frame_equal(full.loc[:cutoff], truncated.loc[:cutoff])


def test_empty_grades_all_nan():
    feat = build_analyst_revision_features(
        pd.DataFrame(columns=["month", "strong_buy", "buy", "hold", "sell", "strong_sell"]), DAYS
    )
    assert feat["analyst_net_rating"].isna().all()
    assert feat["analyst_rating_revision"].isna().all()

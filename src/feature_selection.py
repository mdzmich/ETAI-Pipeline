"""Leak-safe feature selection used as an optional pipeline step."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep features selected by at least ``min_votes`` of three fitted models."""

    def __init__(self, min_votes: int = 2, random_state=None):
        if min_votes not in (1, 2, 3):
            raise ValueError("min_votes must be 1, 2, or 3")
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(
            l1_ratio=1.0,
            C=0.01,
            solver="saga",
            max_iter=5000,
            random_state=self.random_state,
        ).fit(X, y)

        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean()).astype(int)
            + (np.abs(lasso.coef_[0]) > 1e-8).astype(int)
        )
        self.support_ = self.votes_ >= self.min_votes
        self.n_features_in_ = self.votes_.size
        if not self.support_.any():
            raise ValueError(
                f"VoteSelector with min_votes={self.min_votes} selected no features"
            )
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

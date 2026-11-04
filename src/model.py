"""Model construction."""
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.feature_selection import VoteSelector
from src.preprocessing import build_preprocessor


def oblique_random_forest(
    n_estimators: int = 100,
    n_components: int = 10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params,
):
    """A forest of oblique trees: each tree sees a random sparse projection before splitting."""
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def adaboost(max_depth: int = 1, **params):
    """AdaBoost over shallow decision trees of chosen depth."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        **params,
    )


def adaboost_logistic_regression(max_depth: int = 1, **params):
    """AdaBoost over a linear base learner, matching the challenge experiment."""
    return AdaBoostClassifier(
        estimator=LogisticRegression(max_iter=2000),
        **params,
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "adaboost_logistic_regression": adaboost_logistic_regression,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(
    preprocessing_config: dict,
    model_config: dict,
    feature_selection_config: dict | None = None,
    memory=None,
) -> Pipeline:
    """Combine preprocessing and model so every learned step is fitted within each fold."""
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if feature_selection_config and feature_selection_config.get("enabled", False):
        method = feature_selection_config.get("method", "vote")
        if method == "vote":
            selector = VoteSelector(
                min_votes=feature_selection_config.get("min_votes", 2),
                random_state=feature_selection_config.get("random_state"),
            )
        elif method == "kbest":
            selector = SelectKBest(
                score_func=f_classif,
                k=feature_selection_config.get("k", 4),
            )
        elif method == "from_model":
            selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=20,
                    random_state=feature_selection_config.get("random_state"),
                    n_jobs=-1,
                ),
                threshold=feature_selection_config.get("threshold", "mean"),
            )
        else:
            raise ValueError(
                f"Unknown feature-selection method: {method}. "
                "Options: ['vote', 'kbest', 'from_model']"
            )
        steps.append((
            "select",
            selector,
        ))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)

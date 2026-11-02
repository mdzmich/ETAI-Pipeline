"""Cross-validation, out-of-fold reports, and a simple group fairness check."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    get_scorer,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list[str]) -> dict:
    """Build sklearn scorers, including custom ``fbeta_<beta>`` metrics."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            try:
                beta = float(name.split("_", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid F-beta metric name: {name!r}") from exc
            if beta <= 0:
                raise ValueError(f"F-beta must be positive in metric {name!r}")
            scorers[name] = make_scorer(
                fbeta_score, beta=beta, zero_division=0
            )
        elif name in {"precision", "recall", "f1"}:
            metric = {
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }[name]
            scorers[name] = make_scorer(metric, zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    if not scorers:
        raise ValueError("At least one scoring metric is required")
    return scorers


def _take_rows(values, indices):
    return values.iloc[indices] if hasattr(values, "iloc") else values[indices]


def _positive_probability(estimator, X) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{type(estimator).__name__} does not provide predict_proba required for OOF metrics"
        )
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError(f"Expected a binary estimator with positive class 1; got {classes!r}")
    return estimator.predict_proba(X)[:, positive[0]]


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """
    Prints -- and returns as text, so it can also be saved to disk -- train accuracy and test accuracy side by side, plus the usual classification report on the test set.
    """
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy

    lines = [
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {gap:+.3f}",
    ]
    lines.append("")
    lines.append("Classification report (test set):")
    lines.append(classification_report(y_test, y_pred))

    text = "\n".join(lines)
    print(text)
    return text


def holdout_report(y_train, y_train_pred, y_validation, y_validation_pred) -> str:
    """Report train/validation performance for one stratified holdout split."""
    train_accuracy = accuracy_score(y_train, y_train_pred)
    validation_accuracy = accuracy_score(y_validation, y_validation_pred)
    lines = [
        "Holdout evaluation (development set)",
        f"Train accuracy:      {train_accuracy:.3f}",
        f"Validation accuracy: {validation_accuracy:.3f}",
        f"Gap (train - validation): {train_accuracy - validation_accuracy:+.3f}",
        "",
        "Classification report (validation set):",
        classification_report(y_validation, y_validation_pred, zero_division=0),
    ]
    text = "\n".join(lines)
    print(text)
    return text


def fairness_report(
    y_true,
    oof_proba,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
    threshold: float = 0.5,
) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = (np.asarray(oof_proba) >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr, dropna=False):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


def cross_validate_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorers: dict,
    selection_metric: str,
    n_jobs: int = 1,
):
    """Return every metric by fold and one out-of-fold probability per row."""
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric {selection_metric!r} is not in the configured scorers"
        )
    splits = list(cv.split(X, y))
    validation_counts = np.zeros(len(X), dtype=int)
    for _, validation_indices in splits:
        validation_counts[validation_indices] += 1
    if not np.all(validation_counts == 1):
        raise ValueError("Out-of-fold probabilities require CV folds that validate each row once")

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=splits,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    fold_scores = pd.DataFrame({
        "fold": range(1, len(splits) + 1),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X), dtype=float)
    for estimator, (_, validation_indices) in zip(scores["estimator"], splits):
        validation_X = _take_rows(X, validation_indices)
        oof_proba[validation_indices] = _positive_probability(estimator, validation_X)
    return fold_scores, oof_proba


def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list[str],
    selection_metric: str,
    title: str = "Cross-validation",
) -> str:
    """Format per-fold results and mean/standard deviation for each metric."""
    lines = [
        f"{title} ({len(fold_scores)} folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    lines.append(
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )
    lines.append(
        f"\n{selection_metric} train-validation gap mean = {fold_scores['gap'].mean():+.3f}"
    )
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(
    y_true, oof_proba, threshold: float = 0.5
) -> str:
    """Format a confusion matrix and classification report from OOF probabilities."""
    y_pred = (np.asarray(oof_proba) >= threshold).astype(int)
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=["true 0", "true 1"],
        columns=["predicted 0", "predicted 1"],
    )
    text = (
        f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
        f"{matrix.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )
    print(text)
    return text


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """Corrected paired t-test for scores measured on the same CV folds."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.ndim != 1 or scores_b.ndim != 1 or len(scores_a) < 2:
        raise ValueError("At least two paired fold scores are required")
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired score arrays must have the same length")
    if test_train_ratio < 0:
        raise ValueError("test_train_ratio cannot be negative")

    differences = scores_a - scores_b
    standard_error = np.sqrt(
        (1 / len(differences) + test_train_ratio) * differences.var(ddof=1)
    )
    statistic = float(differences.mean() / standard_error) if standard_error else 0.0
    p_value = float(2 * stats.t.sf(abs(statistic), df=len(differences) - 1))
    return {
        "mean_diff": float(differences.mean()),
        "wins": f"{int((differences > 0).sum())}/{len(differences)}",
        "t": statistic,
        "p_value": p_value,
    }

"""Cross-validation reports and a simple group fairness check."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate


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


def fairness_report(y_test, y_pred, extras_test: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """Return per-fold train/validation scores and out-of-fold predictions."""
    splits = list(cv.split(X, y))
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        n_jobs=n_jobs,
    )
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for estimator, (_, validation_indices) in zip(scores["estimator"], splits):
        validation_X = X.iloc[validation_indices] if hasattr(X, "iloc") else X[validation_indices]
        y_oof[validation_indices] = estimator.predict(validation_X)
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Format per-fold results and mean/standard deviation for each score."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    for column in ["train", "validation", "gap"]:
        sign = "+" if column == "gap" else ""
        lines.append(
            f"{column.capitalize():<11s} mean = {fold_scores[column].mean():{sign}.3f}   "
            f"std = {fold_scores[column].std(ddof=1):.3f}"
        )
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, y_pred) -> str:
    """Format classification metrics for out-of-fold predictions."""
    text = (
        "Classification report (out-of-fold predictions, development set):\n"
        + classification_report(y_true, y_pred, zero_division=0)
    )
    print(text)
    return text

"""Leak-safe hyperparameter tuning and nested cross-validation."""
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorer,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Tune a pipeline with seeded Optuna trials scored by cross-validation."""
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(
                    f"Unknown search-space type for {name}: {spec['type']}. "
                    "Options: int, float, categorical"
                )

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            error_score="raise",
        )
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scorers: dict,
    selection_metric: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Estimate tuning performance using outer folds untouched by the inner search."""
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric {selection_metric!r} is not in the configured scorers"
        )
    rows = []
    y_oof_proba = np.empty(len(X), dtype=float)

    for fold, (train_indices, validation_indices) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_indices] if hasattr(X, "iloc") else X[train_indices]
        y_train = y.iloc[train_indices] if hasattr(y, "iloc") else y[train_indices]
        X_validation = X.iloc[validation_indices] if hasattr(X, "iloc") else X[validation_indices]
        y_validation = y.iloc[validation_indices] if hasattr(y, "iloc") else y[validation_indices]
        best_pipeline, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            random_state,
            n_jobs,
        )
        best_pipeline.fit(X_train, y_train)

        train_score = scorers[selection_metric](best_pipeline, X_train, y_train)
        validation_scores = {
            name: scorer(best_pipeline, X_validation, y_validation)
            for name, scorer in scorers.items()
        }
        classes = np.asarray(best_pipeline.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise ValueError(
                f"Expected a binary estimator with positive class 1; got {classes!r}"
            )
        y_oof_proba[validation_indices] = best_pipeline.predict_proba(
            X_validation
        )[:, positive[0]]
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": validation_scores[selection_metric],
            "gap": train_score - validation_scores[selection_metric],
            "inner_best": study.best_value,
            **validation_scores,
            **study.best_params,
        })

    return pd.DataFrame(rows), y_oof_proba


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    selection_metric: str,
    top: int = 5,
) -> str:
    """Report the final search and compare inner and honest outer scores."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(
        columns=lambda column: column.replace("params_", "").replace("user_attrs_", "")
    ).rename(columns={"number": "trial", "value": f"mean {selection_metric}"})
    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()
    lines = [
        f"Tuning on the whole development set "
        f"({len(study.trials)} Optuna trials, metric: {selection_metric})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (maximum over trials; optimistic)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {selection_metric}", ascending=False)
        .head(top)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  tuning score inside outer folds (inner_best) mean = {inner:.3f}",
        f"  honest score on outer folds (validation) mean       = {outer:.3f}",
        f"  optimism = {inner - outer:+.3f}   -> report the honest score",
    ]
    text = "\n".join(lines)
    print(text)
    return text
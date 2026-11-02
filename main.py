"""
Entry point for the cross-validated predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> clean -> split -> cross-validate pipeline
    -> refit on development data -> save results
"""
from datetime import datetime

import yaml
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import load_data
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    corrected_paired_test,
    cv_report,
    fairness_report,
    holdout_report,
    oof_classification_report,
)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean, config["diagnostics"].get("id_column")
    )
    df_clean = add_engineered_features(df_clean)
    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        config["preprocessing"].get("mnar_indicator_sources", []),
    )

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    holdout_config = config["holdout"]
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_dev,
        y_dev,
        test_size=holdout_config["size"],
        random_state=holdout_config["random_state"],
        stratify=y_dev,
    )
    cv_config = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config.get("shuffle", True),
        random_state=(cv_config.get("random_state") if cv_config.get("shuffle", True) else None),
    )
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    if selection_metric not in scorers:
        raise ValueError(
            f"cv.selection_metric {selection_metric!r} must be listed in cv.metrics"
        )

    model_configs = dict(config.get("model_comparison", {}))
    selected_model_name = config["model"]["type"]
    model_configs[selected_model_name] = config["model"]
    comparison_rows = []
    model_results = {}

    for model_name, model_config in model_configs.items():
        feature_config = config.get("feature_selection", {})
        model_feature_config = (
            feature_config
            if model_name == selected_model_name and feature_config.get("enabled", False)
            else None
        )
        holdout_model = build_pipeline(
            config["preprocessing"], model_config, model_feature_config
        )
        holdout_model.fit(X_train, y_train)
        train_predictions = holdout_model.predict(X_train)
        validation_predictions = holdout_model.predict(X_validation)
        fold_scores, y_oof = cross_validate_pipeline(
            build_pipeline(
                config["preprocessing"], model_config, model_feature_config
            ),
            X_dev,
            y_dev,
            cv,
            scorers=scorers,
            selection_metric=selection_metric,
            n_jobs=cv_config.get("n_jobs", 1),
        )
        model_results[model_name] = {
            "fold_scores": fold_scores,
            "oof_proba": y_oof,
            "train_predictions": train_predictions,
            "validation_predictions": validation_predictions,
        }
        comparison_rows.append({
            "model": model_name,
            "holdout train": accuracy_score(y_train, train_predictions),
            "holdout validation": accuracy_score(y_validation, validation_predictions),
            f"CV {selection_metric} mean": fold_scores["validation"].mean(),
            f"CV {selection_metric} std": fold_scores["validation"].std(ddof=1),
            "CV train-validation gap": fold_scores["gap"].mean(),
        })

    comparison_table = pd.DataFrame(comparison_rows).set_index("model")
    comparison_text = comparison_table.to_string(float_format=lambda value: f"{value:.3f}")
    print("Holdout and cross-validation comparison (same splits for every model):")
    print(comparison_text)

    report = "Model comparison (same holdout split and CV folds):\n\n" + comparison_text

    selected_pipeline = build_pipeline(
        config["preprocessing"],
        config["model"],
        config.get("feature_selection"),
    )
    selected_results = model_results[selected_model_name]
    untuned_scores = selected_results["fold_scores"]
    oof_proba = selected_results["oof_proba"]
    floor_scores, _ = cross_validate_pipeline(
        build_pipeline(config["preprocessing"], {"type": "dummy", "params": {}}),
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs=cv_config.get("n_jobs", 1),
    )
    floor = floor_scores["validation"].mean()
    untuned = untuned_scores["validation"].mean()
    report += (
        f"\n\nDefault hyperparameters: {selection_metric} = {untuned:.3f}"
        f"\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}\n"
    )
    tuning_config = config.get("tuning", {})
    tuning_text = ""
    study = None
    final_scores = untuned_scores
    if tuning_config.get("enabled", False):
        search_spaces = tuning_config.get("search_spaces") or {}
        if selected_model_name not in search_spaces:
            raise ValueError(
                f"Tuning is enabled, but no search space is configured for "
                f"{selected_model_name!r}. "
                f"Available model types: {list(search_spaces)}"
            )

        search_space = dict(search_spaces[selected_model_name])
        feature_config = config.get("feature_selection", {})
        if feature_config.get("enabled", False):
            search_space["select__min_votes"] = {
                "type": "categorical",
                "choices": feature_config.get("min_votes_candidates", [1, 2, 3]),
            }
        n_trials = tuning_config.get("n_trials_by_model", {}).get(
            selected_model_name, tuning_config["n_trials"]
        )
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )
        n_jobs = cv_config.get("n_jobs", 1)
        final_scores, oof_proba = nested_cross_validate(
            selected_pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        selected_results["fold_scores"] = final_scores
        selected_results["oof_proba"] = oof_proba
        print("\nNested cross-validation for the selected model:")
        tuning_cv_text = cv_report(
            final_scores, metrics, selection_metric, "Nested cross-validation"
        )

        selected_pipeline, study = tune_pipeline(
            selected_pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        tuning_text = tuning_report(study, final_scores, selection_metric)
        report += "\n\nNested cross-validation for the selected model\n" + tuning_cv_text
        report += "\n\n" + tuning_text

    feature_config = config.get("feature_selection", {})
    if feature_config.get("enabled", False):
        baseline_pipeline = build_pipeline(
            config["preprocessing"],
            config["model"],
            {**feature_config, "enabled": False},
        )
        if tuning_config.get("enabled", False):
            baseline_search_space = {
                name: spec
                for name, spec in search_space.items()
                if name != "select__min_votes"
            }
            baseline_scores, _ = nested_cross_validate(
                baseline_pipeline,
                X_dev,
                y_dev,
                cv,
                inner_cv,
                scorers,
                selection_metric,
                baseline_search_space,
                n_trials,
                tuning_seed,
                n_jobs=cv_config.get("n_jobs", 1),
            )
        else:
            baseline_scores, _ = cross_validate_pipeline(
                baseline_pipeline,
                X_dev,
                y_dev,
                cv,
                scorers,
                selection_metric,
                n_jobs=cv_config.get("n_jobs", 1),
            )
        paired = corrected_paired_test(
            final_scores["validation"],
            baseline_scores["validation"],
            1 / (cv_config["n_splits"] - 1),
        )
        selection_comparison = (
            "Feature selection comparison (positive difference = selection wins)\n"
            f"  mean difference: {paired['mean_diff']:+.3f}\n"
            f"  wins: {paired['wins']}\n"
            f"  corrected paired t: {paired['t']:.3f}\n"
            f"  p-value: {paired['p_value']:.3f}"
        )
        print("\n" + selection_comparison)
        report += "\n\n" + selection_comparison

    for model_name, results in model_results.items():
        print(f"\nDetailed reports: {model_name}")
        report += f"\n\n\n{model_name} - without cross-validation\n"
        report += holdout_report(
            y_train,
            results["train_predictions"],
            y_validation,
            results["validation_predictions"],
        )
        report += f"\n\n{model_name} - with cross-validation\n"
        report += cv_report(
            results["fold_scores"], metrics, selection_metric
        )
        report += "\n\n" + oof_classification_report(
            y_dev, results["oof_proba"]
        )
        report += "\n" + fairness_report(
            y_dev,
            results["oof_proba"],
            extras_dev,
            sensitive_attr=config["data"]["sensitive_attr"],
        )

    final_model = selected_pipeline.fit(X_dev, y_dev)
    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")
    print(f"Locked test set reserved: {len(X_test)} rows.")

    scoreboard_row = {
        "run": run_id,
        "data": config["data"]["path"],
        "model": selected_model_name,
        "tuned": bool(tuning_config.get("enabled", False)),
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params", {})),
        "tuned_params": str(study.best_params) if study is not None else "",
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{index}": score
            for index, score in enumerate(final_scores["validation"], start=1)
        },
    }
    output_config = config.get("output", {})
    runs = save_scores(output_config.get("scoreboard", "results/runs.csv"), scoreboard_row)
    same_runs = runs[
        (runs["data"] == scoreboard_row["data"])
        & (runs["metric"] == selection_metric)
    ]
    board = (
        same_runs.sort_values("mean", ascending=False)[
            ["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nScoreboard (same data and selection metric, best first):\n{board}")
    report += f"\n\nScoreboard:\n{board}\n"

    results_dir = output_config.get("results_dir", "results")
    path = save_run(results_dir, config, report, run_id=run_id)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()

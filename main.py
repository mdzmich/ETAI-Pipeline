"""
Entry point for the cross-validated predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> clean -> split -> cross-validate pipeline
    -> refit on development data -> save results
"""
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    holdout_report,
    oof_classification_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_pipeline(model_config: dict, preprocessing_config: dict) -> Pipeline:
    """Combine preprocessing and model so both are fitted inside every split."""
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean, config["diagnostics"].get("id_column")
    )
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

    model_configs = dict(config.get("model_comparison", {}))
    selected_model_name = config["model"]["type"]
    model_configs[selected_model_name] = config["model"]
    comparison_rows = []
    model_results = {}
    scoring = cv_config.get("scoring", "accuracy")

    for model_name, model_config in model_configs.items():
        holdout_model = make_pipeline(model_config, config["preprocessing"])
        holdout_model.fit(X_train, y_train)
        train_predictions = holdout_model.predict(X_train)
        validation_predictions = holdout_model.predict(X_validation)
        fold_scores, y_oof = cross_validate_pipeline(
            make_pipeline(model_config, config["preprocessing"]),
            X_dev,
            y_dev,
            cv,
            scoring=scoring,
            n_jobs=cv_config.get("n_jobs", 1),
        )
        model_results[model_name] = {
            "fold_scores": fold_scores,
            "y_oof": y_oof,
            "train_predictions": train_predictions,
            "validation_predictions": validation_predictions,
        }
        comparison_rows.append({
            "model": model_name,
            "holdout train": accuracy_score(y_train, train_predictions),
            "holdout validation": accuracy_score(y_validation, validation_predictions),
            f"CV {scoring} mean": fold_scores["validation"].mean(),
            f"CV {scoring} std": fold_scores["validation"].std(ddof=1),
            "CV train-validation gap": fold_scores["gap"].mean(),
        })

    comparison_table = pd.DataFrame(comparison_rows).set_index("model")
    comparison_text = comparison_table.to_string(float_format=lambda value: f"{value:.3f}")
    print("Holdout and cross-validation comparison (same splits for every model):")
    print(comparison_text)

    report = "Model comparison (same holdout split and CV folds):\n\n" + comparison_text
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
        report += cv_report(results["fold_scores"], scoring)
        report += "\n\n" + oof_classification_report(y_dev, results["y_oof"])
        report += "\n" + fairness_report(
            y_dev,
            results["y_oof"],
            extras_dev,
            sensitive_attr=config["data"]["sensitive_attr"],
        )

    final_model = make_pipeline(config["model"], config["preprocessing"]).fit(X_dev, y_dev)
    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")
    print(f"Locked test set reserved: {len(X_test)} rows.")

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()

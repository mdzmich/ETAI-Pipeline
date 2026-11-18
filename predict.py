"""Predict rows from a label-free CSV using the pipeline saved by main.py."""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    split_features_target,
)


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Return one prediction for every input row."""
    clean = clean_dataset(df, config["diagnostics"])
    clean = add_engineered_features(clean)

    preprocessing = config["preprocessing"]
    required = set(preprocessing["numeric_features"]) | set(preprocessing["categorical_features"])
    required = {column for column in required if not column.endswith("_was_missing")}
    for engineered in ("priors_per_adult_year", "offenses_per_adult_year"):
        if engineered in required:
            required.update(
                {"age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"}
            )
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required model columns: " + ", ".join(missing)
        )

    X, _, _ = split_features_target(
        clean,
        config["data"],
        preprocessing["mnar_indicator_sources"],
    )
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"{type(pipeline).__name__} does not provide predict_proba")
    positive_indices = (pipeline.classes_ == 1).nonzero()[0]
    if len(positive_indices) != 1:
        raise ValueError(
            f"Expected a binary target with positive class 1; got classes {pipeline.classes_!r}"
        )
    probabilities = pipeline.predict_proba(X)[:, positive_indices[0]]
    id_column = config["diagnostics"]["id_column"]
    if id_column not in df.columns:
        raise ValueError(f"Input CSV is missing required id column: {id_column}")

    return pd.DataFrame({
        id_column: df[id_column].to_numpy(),
        "probability": probabilities,
        "prediction": (probabilities >= 0.5).astype(int),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a CSV with the model saved by main.py."
    )
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument(
        "output",
        nargs="?",
        default="predictions.csv",
        help="where to write the predictions",
    )
    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")

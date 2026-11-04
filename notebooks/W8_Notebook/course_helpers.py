"""
Course helpers: last week's pipeline in one file, so the notebook runs on its own.

Everything here is the course pipeline exactly as it was at the end of last week (same
functions, same config). This week's new code is written in the notebook, not here.
`load_course_data()` loads, cleans and splits the data the same way `main.py` does.
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from category_encoders import CountEncoder
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
DATA_DIR = Path(__file__).parent / "data"

# config.yaml, as it was at the end of last week
CONFIG = yaml.safe_load("""
data:
  path: "data/compas_two_year_recidivism.csv"
  target: "two_year_recid"
  sensitive_attr: "race"                            # never a feature; used only for the fairness report
  drop_columns: ["id", "decile_score", "score_text"]  # kept in the data, never used as features

# Locked test set: never used to fit, tune or choose anything. Do not change once set.
test_set:
  size: 0.2
  random_state: 42

cv:
  n_splits: 5
  shuffle: true
  random_state: 42
  # Reported for every run. fbeta_<beta> weights recall beta times as much as precision.
  metrics: ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "fbeta_2", "roc_auc", "average_precision"]
  selection_metric: "roc_auc"                       # one of metrics: what tuning maximises and runs are ranked by
  n_jobs: 1                                         # -1 = all CPU cores

diagnostics:
  id_column: "id"
  numeric_text_columns: ["priors_count", "prior_offenses"]
  # "nan" catches real NaNs turned into text by the category cleanup.
  placeholder_tokens: ["-", "?", "n/a", "N/A", "na", "NA", "", "nan"]
  validity_rules:
    age: {min: 18, max: 100}
    decile_score: {min: 1, max: 10}
    juv_fel_count: {min: 0}
    priors_count: {min: 0, max: 60}
  canonical_categories:
    sex:
      male: "Male"
      female: "Female"
    race:
      african-american: "African-American"
      african american: "African-American"
      caucasian: "Caucasian"
      hispanic: "Hispanic"
      other: "Other"
      asian: "Asian"
      native american: "Native American"
    c_charge_degree:
      f: "F"
      felony: "F"
      m: "M"
      misdemeanor: "M"
    score_text:
      low: "Low"
      medium: "Medium"
      high: "High"
  redundant_columns: ["prior_offenses", "age_in_months", "juvenile_total"]

preprocessing:
  encoder: "target"                                 # onehot | ordinal | count | target
  scaler: "robust"                                  # none | standard | minmax | robust
  numeric_features: ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
  categorical_features: ["sex", "age_cat", "c_charge_degree"]
  mnar_indicator_sources: ["priors_count", "c_charge_degree"]   # get a <col>_was_missing flag
  random_state: 42                                  # target encoder's internal folds
  imputation:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"

model:
  type: "naive_bayes"  # dummy | logistic_regression | naive_bayes | decision_tree | random_forest | oblique_random_forest | adaboost | gradient_boosting | hist_gradient_boosting
  params:
    {}
  # Other models' params, to swap in. class_weight "balanced" weights each class by 1 / its frequency
  # (GradientBoostingClassifier and GaussianNB have no class_weight).
  # logistic_regression:    {max_iter: 2000, class_weight: balanced}
  # naive_bayes:            {}
  # decision_tree:          {random_state: 42, class_weight: balanced}
  # random_forest:          {n_estimators: 200, random_state: 42, n_jobs: -1, class_weight: balanced}
  # oblique_random_forest:  {n_estimators: 200, n_components: 10, random_state: 42, n_jobs: -1}
  # adaboost:               {max_depth: 1, n_estimators: 200, learning_rate: 1.0, random_state: 42}
  # hist_gradient_boosting: {random_state: 42, class_weight: balanced}

tuning:
  enabled: true
  n_trials: 30
  n_splits: 5                                       # folds that score each candidate
  random_state: 42                                  # seeds the sampler and those folds
  # Keys are Pipeline parameter names. Types: int / float (low, high, log) or categorical (choices).
  # Ensembles take minutes per run: use a small n_trials while testing.
  search_spaces:
    decision_tree:
      model__max_depth: {type: int, low: 1, high: 20}
      model__min_samples_leaf: {type: int, low: 1, high: 200, log: true}
      model__criterion: {type: categorical, choices: ["gini", "entropy"]}
    random_forest:
      model__max_depth: {type: int, low: 2, high: 30}
      model__min_samples_leaf: {type: int, low: 1, high: 200, log: true}
      model__max_features: {type: categorical, choices: ["sqrt", 0.5, 1.0]}
    oblique_random_forest:
      model__estimator__tree__max_depth: {type: int, low: 2, high: 30}
      model__estimator__tree__min_samples_leaf: {type: int, low: 1, high: 200, log: true}
      model__estimator__project__n_components: {type: int, low: 3, high: 10}   # <= number of features
    adaboost:
      model__estimator__max_depth: {type: int, low: 1, high: 8}
      model__n_estimators: {type: int, low: 25, high: 400, log: true}
      model__learning_rate: {type: float, low: 0.01, high: 1.0, log: true}
    gradient_boosting:
      model__n_estimators: {type: int, low: 25, high: 500, log: true}
      model__learning_rate: {type: float, low: 0.01, high: 0.5, log: true}
      model__max_depth: {type: int, low: 1, high: 6}
      model__min_samples_leaf: {type: int, low: 1, high: 200, log: true}
      model__subsample: {type: float, low: 0.5, high: 1.0}
    hist_gradient_boosting:
      model__max_iter: {type: int, low: 25, high: 500, log: true}
      model__learning_rate: {type: float, low: 0.01, high: 0.5, log: true}
      model__max_leaf_nodes: {type: int, low: 2, high: 63, log: true}
      model__min_samples_leaf: {type: int, low: 5, high: 200, log: true}
      model__l2_regularization: {type: float, low: 0.000001, high: 10.0, log: true}
    naive_bayes:
      model__var_smoothing: {type: float, low: 0.000000000001, high: 0.1, log: true}

output:
  results_dir: "results"
  scoreboard: "results/runs.csv"                    # one row per run; delete it when the data, cleaning or folds change
""")


# ---- src/data.py, src/preprocessing.py, src/model.py, src/evaluate.py, src/tuning.py ----

def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(path)


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN, in place. Returns the number of violations per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the cleaning rules in `diagnostics_config`: placeholder tokens and domain-rule
    violations become NaN, category spellings are canonicalized and redundant columns are
    dropped. Row-preserving and target-agnostic, so it runs unchanged on data to predict.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    Training data only, before `split_dev_test`: drops exact duplicate rows and repeated ids
    (keeping the first). Never call it on data to predict: every row needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each listed column, before it is imputed."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `extras` holds the columns kept out of the features for auditing
    (the sensitive attribute and COMPAS's own score). On label-free data `y` is None.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    return df[feature_cols], y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# sklearn's TargetEncoder cross-fits: a training row's encoding never uses its own label.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: imputation and
    scaling for numeric columns, imputation and encoding for categorical ones, and the
    missingness flags passed through. Every encoder tolerates unseen categories.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split into a development set, used for every decision, and a locked test set,
    used only for the final assessment. Returns X_dev, X_test, y_dev, y_test, extras_dev,
    extras_test, row-aligned.
    """
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)


def adaboost(max_depth: int = 1, **params):
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps); `params` go to
    AdaBoostClassifier. The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample, on `n_components`
    random sparse linear combinations of the features (a new set per tree), so its splits are
    hyperplanes. `tree_params` go to every tree (`max_features` defaults to "sqrt"). Tuned as
    `model__estimator__tree__<param>` and `model__estimator__project__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
}


def build_model(model_config: dict):
    """Builds the model named by `model_config["type"]` with `model_config["params"]`."""
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing and model as one
    Pipeline, so every learned step is re-fit on the training part of each fold. Step names
    address hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
    Any new learned step (e.g. feature selection) belongs here, between "prep" and "model".
    """
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probability of the positive class for every row, from the
        fold model that did not train on it. `cv` must be a partition (each row validated
        exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5) -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
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
    return "\n".join(lines)


def tune_pipeline(pipeline, X, y, cv, scorer, search_space: dict, n_trials: int,
                  random_state: int, n_jobs: int = 1):
    """
    Runs an Optuna study (seeded TPE sampler) on (X, y). Each trial draws one value per entry
    of `search_space` (`type`: int / float / categorical; keys are Pipeline parameter names,
    e.g. `model__max_depth`), sets them on a fresh copy of `pipeline`, and returns its mean
    CV score over `cv` with `scorer` (the selection metric).

    Returns (best_pipeline, study): an unfitted copy of `pipeline` with the best trial's
    hyperparameters, and the study with every trial (mean score and fold std).
    """
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type for {name}: {spec['type']}. "
                                 "Options: int, float, categorical")
        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(pipeline, X, y, outer_cv, inner_cv, scorers: dict, selection_metric: str,
                          search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1):
    """
    Honest estimate of the tuning procedure. For every outer fold: tune on the outer-training
    rows (`tune_pipeline` with `inner_cv`, on `selection_metric`), refit the best pipeline on
    them, and score it on the outer-validation rows, which took no part in the tuning.

    Returns (fold_scores, oof_proba), as `cross_validate_pipeline` does, with two extra kinds
    of columns: `inner_best` (the tuning score inside that fold) and the hyperparameters chosen.
    """
    rows = []
    oof_proba = np.empty(len(X))
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(pipeline, X_train, y_train, inner_cv, scorers[selection_metric],
                                             search_space, n_trials, random_state, n_jobs)
        best_pipeline.fit(X_train, y_train)

        train_score = scorers[selection_metric](best_pipeline, X_train, y_train)
        val_scores = {name: scorer(best_pipeline, X_val, y_val) for name, scorer in scorers.items()}
        oof_proba[val_idx] = best_pipeline.predict_proba(X_val)[:, 1]
        rows.append({"fold": fold, "train": train_score, "validation": val_scores[selection_metric],
                     "gap": train_score - val_scores[selection_metric], **val_scores,
                     "inner_best": study.best_value, **study.best_params})
    return pd.DataFrame(rows), oof_proba


def tuning_report(study, nested_scores: pd.DataFrame, selection_metric: str, top: int = 5) -> str:
    """
    Best hyperparameters and top trials of the final study, plus the optimism check (mean
    tuning score vs mean honest score over the outer folds), as text.
    """
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": f"mean {selection_metric}"})
    inner, outer = nested_scores["inner_best"].mean(), nested_scores["validation"].mean()
    lines = [
        f"Tuning on the whole development set ({len(study.trials)} Optuna trials, metric: {selection_metric})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (the maximum over all trials -- optimistic)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {selection_metric}", ascending=False).head(top)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  tuning score inside the outer folds (inner_best) mean = {inner:.3f}",
        f"  honest score on the outer folds (validation)     mean = {outer:.3f}",
        f"  optimism = {inner - outer:+.3f}   -> report the honest one",
    ]
    return "\n".join(lines)


def load_course_data(config: dict = CONFIG):
    """
    Loads and cleans the course data and locks the test set away, exactly as `main.py` does.
    Returns X_dev, y_dev, extras_dev (race and COMPAS's own score, never features) and the
    stratified CV folds from `config["cv"]`.
    """
    df = load_data(DATA_DIR / Path(config["data"]["path"]).name)
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"])
    cv = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=config["cv"]["shuffle"],
                         random_state=config["cv"]["random_state"])
    return X_dev, y_dev, extras_dev, cv

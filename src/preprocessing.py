"""Data cleaning, feature splitting, and leak-safe preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies a dict of {column: {"min": ..., "max": ...}} domain rules (either bound is
    optional) and converts violations to NaN **in place** on 'df'. An "impossible but
    not missing" value (an age of -3, a COMPAS decile score of 15) counts as missing
    once this runs -- '.isna()' alone would never have caught it.

    Rule columns are converted to numeric; unparseable values become missing.
    Returns a small report counting out-of-range numeric values per column.
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
        df[column] = numeric.mask(violations)
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Normalize configured categories without converting missing values to text."""
    out = df.copy()
    tokens = {str(token).strip().lower() for token in placeholder_tokens}
    for column, mapping in columns_and_maps.items():
        if column not in out.columns:
            continue
        cleaned = out[column].astype("string").str.strip()
        lowered = cleaned.str.lower()
        normalized_mapping = {str(key).strip().lower(): value for key, value in mapping.items()}
        normalized = (
            lowered.map(normalized_mapping).fillna(cleaned)
            .astype("string").mask(lowered.isin(tokens))
        )
        out[column] = normalized.astype(object).where(normalized.notna(), np.nan)
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies category cleanup, domain-rule/placeholder -> NaN conversion, and removal
    of redundant columns. Row-preserving and target-agnostic; duplicate removal is a
    separate training-only step.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """Remove duplicate training records and repeated IDs, keeping the first row."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Add indicator features before missing values are imputed."""
    out = df.copy()
    for column in mnar_indicator_sources:
        if column in out.columns:
            out[f"{column}_was_missing"] = out[column].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """Return model features, optional target, and metadata reserved for auditing."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None
    extras_columns = [column for column in [sensitive_attr, "score_text"] if column in df.columns]
    extras = df[extras_columns].copy() if extras_columns else None

    excluded = set(drop_columns) | {target, sensitive_attr}
    feature_columns = [column for column in df.columns if column not in excluded]
    return df[feature_columns], y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Create stratified development and locked test sets, keeping metadata aligned."""
    return train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )


_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _make_encoder(name: str, random_state: int):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "target":
        return TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        )
    raise ValueError(f"Unknown encoder: {name}. Options: ['onehot', 'ordinal', 'target']")


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Build preprocessing whose learned statistics are fitted within each CV fold."""
    scaler_name = preprocessing_config["scaler"]
    encoder_name = preprocessing_config["encoder"]
    if scaler_name not in _SCALERS:
        raise ValueError(f"Unknown scaler: {scaler_name}. Options: {list(_SCALERS)}")

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    imputation = preprocessing_config.get("imputation", {})
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", _make_encoder(encoder_name, preprocessing_config.get("random_state", 42))),
    ])

    indicator_columns = [
        f"{column}_was_missing"
        for column in preprocessing_config.get("mnar_indicator_sources", [])
    ]
    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_columns),
    ])

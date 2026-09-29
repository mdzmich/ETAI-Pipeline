"""
Configurable data cleaning and baseline preprocessing.

After cleaning, preprocessing retains these baseline limitations:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        out[column] = (
            lowered.map(normalized_mapping).fillna(cleaned)
            .astype("string").mask(lowered.isin(tokens))
        )
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies this week's diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, de-duplication, and redundant-column removal. Target-agnostic -- safe
    to call on label-free inference data, since none of this depends on a target column.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        # Missing IDs do not establish that two different records are duplicates.
        out = out.loc[out[id_column].isna() | ~out[id_column].duplicated(keep="first")]

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    """Split cleaned data into model features, target, and fairness metadata."""
    df = df.dropna()
    y = df[target]
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        column for column in drop_columns if column in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, extras_test

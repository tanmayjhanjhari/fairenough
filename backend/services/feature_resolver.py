"""
FairEnough ? Domain-Agnostic Feature Name Resolver

Resolves feature name differences between trained machine learning models
(which often preserve case-sensitive, PascalCase, or original feature names
such as 'CheckingStatus', 'Sex', 'LoanDuration') and FairEnough's internally
normalized dataset columns ('checkingstatus', 'sex', 'loanduration').

Guarantees:
1. Domain-agnostic: No hardcoded column names or assumptions about dataset fields.
2. Exact Reconstruction: Assembles a feature matrix X whose columns and order
   match model.feature_names_in_ exactly.
3. Ambiguity Detection: If multiple dataset columns match a single model feature,
   or if multiple model features map to the same dataset column, raises a descriptive
   ValueError rather than guessing.
4. Missing Feature Detection: If required model features cannot be mapped, reports
   an informative error detailing what was expected vs available.
5. Backward-compatibility: If model features already match dataset columns,
   maps them directly without alteration.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def normalize_column_name(col: str) -> str:
    """The standard normalization used in FairEnough DataPreprocessor."""
    new_col = str(col).strip().strip(chr(65279)).lower()
    new_col = re.sub(r'[\s\-]+', '_', new_col)
    new_col = re.sub(r'[^a-z0-9_]', '', new_col)
    return new_col


def canonical_alphanumeric(col: str) -> str:
    """Strict alphanumeric lowercase string for robust case/separator-agnostic matching."""
    return re.sub(r'[^a-z0-9]', '', str(col).strip().strip(chr(65279)).lower())


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            curr[j + 1] = prev[j] if c1 == c2 else 1 + min(prev[j], prev[j + 1], curr[j])
        prev = curr
    return prev[len(s2)]


def resolve_model_features(
    model: Any,
    df: pd.DataFrame,
    target_col: str | None = None,
    column_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Resolve model's expected features (from feature_names_in_) to DataFrame columns.

    Args:
        model: Trained scikit-learn compatible estimator.
        df: Preprocessed DataFrame (columns normalized).
        target_col: Optional target column name to exclude from feature mapping.
        column_mapping: Optional dict of {original_col_name: normalized_col_name}.

    Returns:
        tuple of (resolved_mapping, missing_features)
        where resolved_mapping is {model_feature_name: df_column_name}
        and missing_features is a list of model feature names that could not be resolved.

    Raises:
        ValueError if an ambiguous mapping is detected.
    """
    raw_features = getattr(model, "feature_names_in_", None)
    if raw_features is None:
        return {}, []

    expected_features = [str(f).strip().strip(chr(65279)) for f in raw_features]
    df_columns = [str(c).strip().strip(chr(65279)) for c in df.columns]
    target_cols_to_exclude = set()
    if target_col:
        target_cols_to_exclude.add(target_col)
        target_cols_to_exclude.add(target_col.lower())
        target_cols_to_exclude.add(normalize_column_name(target_col))
        if column_mapping:
            if target_col in column_mapping:
                target_cols_to_exclude.add(column_mapping[target_col])
            for orig, norm in column_mapping.items():
                if norm == target_col or orig == target_col:
                    target_cols_to_exclude.add(orig)
                    target_cols_to_exclude.add(norm)

    available_cols = [
        c for c in df_columns
        if c not in target_cols_to_exclude and not c.startswith("__")
    ]

    # Pre-index columns for fast lookup
    exact_set = set(available_cols)
    lower_map: dict[str, list[str]] = {}
    canon_map: dict[str, list[str]] = {}
    norm_map: dict[str, list[str]] = {}

    for c in available_cols:
        l = c.lower()
        lower_map.setdefault(l, []).append(c)

        cn = canonical_alphanumeric(c)
        canon_map.setdefault(cn, []).append(c)

        nm = normalize_column_name(c)
        norm_map.setdefault(nm, []).append(c)

    canon_colmap: dict[str, str] = {}
    if column_mapping:
        for orig, norm in column_mapping.items():
            canon_orig = canonical_alphanumeric(orig)
            if canon_orig not in canon_colmap:
                canon_colmap[canon_orig] = norm

    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    for f_exp in expected_features:
        # Step 1: Direct exact match
        if f_exp in exact_set:
            resolved[f_exp] = f_exp
            continue

        # Step 2: Upload column mapping lookup (original CSV column -> normalized column)
        if column_mapping and f_exp in column_mapping:
            mapped = column_mapping[f_exp]
            if mapped in exact_set:
                resolved[f_exp] = mapped
                continue

        canon_f = canonical_alphanumeric(f_exp)
        if canon_colmap and canon_f in canon_colmap:
            mapped = canon_colmap[canon_f]
            if mapped in exact_set:
                resolved[f_exp] = mapped
                continue

        # Step 3: Match via standard FairEnough normalization
        norm_f = normalize_column_name(f_exp)
        if norm_f in exact_set:
            resolved[f_exp] = norm_f
            continue

        # Step 4: Case-insensitive match
        matches_lower = lower_map.get(f_exp.lower(), [])
        if len(matches_lower) == 1:
            resolved[f_exp] = matches_lower[0]
            continue
        elif len(matches_lower) > 1:
            raise ValueError(
                f"Ambiguous feature mapping for model feature '{f_exp}': "
                f"matches multiple dataset columns case-insensitively: {matches_lower}."
            )

        # Step 5: Canonical alphanumeric match (ignoring underscores/spaces/hyphens)
        canon_f = canonical_alphanumeric(f_exp)
        matches_canon = canon_map.get(canon_f, [])
        if len(matches_canon) == 1:
            resolved[f_exp] = matches_canon[0]
            continue
        elif len(matches_canon) > 1:
            raise ValueError(
                f"Ambiguous feature mapping for model feature '{f_exp}': "
                f"matches multiple dataset columns alphanumerically: {matches_canon}."
            )

        # Step 6: Unambiguous prefix/stem or edit-distance <= 1 match for minor variations
        if len(canon_f) >= 4:
            fuzzy_matches = []
            for c in available_cols:
                cn = canonical_alphanumeric(c)
                if len(cn) >= 4:
                    if (canon_f.startswith(cn) or cn.startswith(canon_f)) and abs(len(canon_f) - len(cn)) <= 2:
                        fuzzy_matches.append(c)
                    elif _edit_distance(canon_f, cn) <= 1:
                        fuzzy_matches.append(c)
            fuzzy_matches = list(dict.fromkeys(fuzzy_matches))
            if len(fuzzy_matches) == 1:
                resolved[f_exp] = fuzzy_matches[0]
                continue
            elif len(fuzzy_matches) > 1:
                raise ValueError(f"Ambiguous feature mapping for model feature '{f_exp}': matches multiple dataset columns via fuzzy match: {fuzzy_matches}.")

        unresolved.append(f_exp)

    # Ambiguity check: ensure no two different model features mapped to the exact same dataset column
    col_to_features: dict[str, list[str]] = {}
    for f_exp, c_df in resolved.items():
        col_to_features.setdefault(c_df, []).append(f_exp)

    for c_df, feats in col_to_features.items():
        if len(feats) > 1:
            raise ValueError(
                f"Ambiguous feature collision: model features {feats} all map to the "
                f"same dataset column '{c_df}'. Cannot resolve features unambiguously."
            )

    return resolved, unresolved


def build_model_feature_matrix(
    model: Any,
    df: pd.DataFrame,
    target_col: str | None = None,
    sensitive_attr: str | None = None,
    column_mapping: dict[str, str] | None = None,
    dropped_cols: dict[str, Any] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix X whose columns and ordering match what the model expects.

    - If model has feature_names_in_, maps names to df columns and outputs a DataFrame
      with columns == model.feature_names_in_.
    - If model has no feature_names_in_, falls back to all numeric features excluding
      target and internal columns.
    - Encodes object/category columns to numeric representation so standard classifiers
      can execute without type errors.

    Raises:
        ValueError if features are missing, ambiguous, or count mismatches occur.
    """
    raw_features = getattr(model, "feature_names_in_", None)

    if raw_features is not None:
        expected_features = [str(f).strip().strip(chr(65279)) for f in raw_features]
        resolved_map, missing = resolve_model_features(
            model=model,
            df=df,
            target_col=target_col,
            column_mapping=column_mapping,
        )

        recoverable_dropped: dict[str, Any] = {}
        unrecoverable_missing: list[str] = []

        if missing:
            if dropped_cols:
                if isinstance(dropped_cols, dict):
                    dropped_lookup = {canonical_alphanumeric(k): v for k, v in dropped_cols.items()}
                else:
                    dropped_lookup = {canonical_alphanumeric(k): 0 for k in dropped_cols}
                for f_exp in missing:
                    cn = canonical_alphanumeric(f_exp)
                    if cn in dropped_lookup:
                        recoverable_dropped[f_exp] = dropped_lookup[cn]
                    elif column_mapping:
                        orig_cn = {canonical_alphanumeric(k): canonical_alphanumeric(v) for k, v in column_mapping.items()}
                        if cn in orig_cn and orig_cn[cn] in dropped_lookup:
                            recoverable_dropped[f_exp] = dropped_lookup[orig_cn[cn]]
                        else:
                            unrecoverable_missing.append(f_exp)
                    else:
                        unrecoverable_missing.append(f_exp)
            else:
                unrecoverable_missing = missing

            if unrecoverable_missing:
                raise ValueError(
                    f'Model expects feature(s) {unrecoverable_missing} that are not present in the uploaded dataset. '
                    f'Expected features: {expected_features}. Available dataset columns: {list(df.columns)}.'
                )

        # Reconstruct DataFrame with exact model feature names in exact expected order
        X = pd.DataFrame(index=df.index)
        for f_exp in expected_features:
            if f_exp in resolved_map:
                c_df = resolved_map[f_exp]
                X[f_exp] = df[c_df].copy()
            elif f_exp in recoverable_dropped:
                X[f_exp] = recoverable_dropped[f_exp]

    else:
        # Fallback for models without feature_names_in_
        internal_cols = {
            target_col,
            "__y__",
            "__predictions__",
            "__sens_binned__",
            "__weight__",
            "__target__",
            "__sens__",
            "__truth__",
            "__s__",
            "__y_tmp__",
            "__sens_raw__",
        }
        cols_to_drop = [c for c in internal_cols if c and c in df.columns]
        X = df.select_dtypes(include="number").drop(columns=cols_to_drop, errors="ignore").copy()

        expected_n = getattr(model, "n_features_in_", None)
        if expected_n is not None and X.shape[1] > expected_n:
            # Common case: model was trained on non-sensitive features only
            if sensitive_attr and sensitive_attr in X.columns and (X.shape[1] - 1) == expected_n:
                X = X.drop(columns=[sensitive_attr])

        if expected_n is not None and X.shape[1] != expected_n:
            raise ValueError(
                f"Model expects {expected_n} numeric feature(s) (n_features_in_), "
                f"but found {X.shape[1]} numeric column(s) in the dataset."
            )

    # Encode any object/category/bool columns to prevent string-to-float errors
    for c in X.columns:
        if X[c].dtype == "object" or X[c].dtype.name in ("category", "bool"):
            try:
                X[c] = X[c].astype(float)
            except (ValueError, TypeError):
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    X = X.fillna(0)
    return X

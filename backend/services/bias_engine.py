"""
FairEnough – Bias Engine

Computes dataset-level and model-level fairness metrics.

METRIC MODES
============
dataset_level (no y_pred provided):
  - SPD  (Statistical Parity Difference)  — REAL measurement
  - DI   (Disparate Impact)               — REAL measurement
  - EOD  (Equal Opportunity Difference)   — NOT AVAILABLE (None)
  - AOD  (Average Odds Difference)        — NOT AVAILABLE (None)

model_level (y_pred provided via __predictions__ column):
  - SPD  — REAL measurement
  - DI   — REAL measurement
  - EOD  — REAL measurement
  - AOD  — REAL measurement

IMPORTANT: EOD and AOD are NEVER fabricated from an internal simulation
model when no external predictions exist.  null/None means "not available",
not "perfectly fair".

CONTINUOUS ATTRIBUTE BINNING
=============================
Numeric sensitive attributes with more than CARDINALITY_BIN_THRESHOLD
unique values are automatically partitioned into two groups using a
data-driven median split:
  attr <= median  → "{attr} <= {median}"
  attr > median   → "{attr} > {median}"

This is completely domain-agnostic and avoids high-cardinality fragmentation
where groups have too few samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BiasEngine:
    """
    Compute fairness metrics for a dataset.

    Privileged group = the group with the highest positive outcome rate.
    This is pragmatic and avoids requiring the caller to know which group
    is historically advantaged.
    """

    BOOTSTRAP_N: int = 200
    BOOTSTRAP_SEED: int = 42

    # Continuous attribute binning config
    # If a numeric attribute has more than this many unique values it is
    # automatically binned into two groups.
    CARDINALITY_BIN_THRESHOLD: int = 10
    CONTINUOUS_BIN_THRESHOLD: float | None = None
    CONTINUOUS_BIN_LABELS: tuple[str, str] = ("Low (<= median)", "High (> median)")

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_attrs: list[str],
        use_predictions: bool = False,
    ) -> dict[str, Any]:
        """
        Run full bias analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset.  If ``use_predictions`` is True the frame must contain a
            column named ``__predictions__`` with model-generated labels.
        target_col : str
            Ground-truth label column.
        sensitive_attrs : list[str]
            Protected-attribute columns to analyse.
        use_predictions : bool
            When True, metrics are computed against ``__predictions__`` instead
            of ``target_col``; this enables EOD and AOD.

        Returns
        -------
        dict
            ``metrics_per_attr``, ``audit_score``, ``overall_severity``,
            ``grade``, ``grade_label``, ``metrics_mode``
        """
        label_col = "__predictions__" if use_predictions else target_col
        metrics_mode = "model_level" if use_predictions else "dataset_level"

        metrics_per_attr: dict[str, Any] = {}

        for attr in sensitive_attrs:
            if attr not in df.columns:
                metrics_per_attr[attr] = {
                    "error": f"Column '{attr}' not found in dataset."
                }
                continue

            # Drop rows where the attribute or label is null
            sub = df.dropna(subset=list(dict.fromkeys([attr, label_col])))
            if sub.empty:
                metrics_per_attr[attr] = {
                    "error": "No valid rows after dropping nulls."
                }
                continue

            metrics_per_attr[attr] = self._compute_attr_metrics(
                sub, attr, target_col, label_col, use_predictions
            )

        # ── Audit Score ─────────────────────────────────────────────────────
        audit_score_rounded = self._compute_audit_score(metrics_per_attr)

        # Grade and overall severity MUST come from audit_score only
        grade, overall_severity, grade_label, _ = self._derive_grade_and_severity(audit_score_rounded)

        return {
            "metrics_per_attr": metrics_per_attr,
            "audit_score": audit_score_rounded,
            "grade": grade,
            "overall_severity": overall_severity,
            "grade_label": grade_label,
            "metrics_mode": metrics_mode,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _bin_continuous_attr(
        self,
        series: pd.Series,
        attr: str,
    ) -> tuple[pd.Series, str]:
        """
        Bin a continuous numeric attribute into two labelled groups using data-driven median split.
        Completely domain-agnostic (never assumes age or fixed domain cutoffs).
        Returns the binned Series and a description string.
        """
        valid = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if valid.empty or valid.nunique() < 2:
            return series.astype(str), f"'{attr}' has insufficient variation to partition into groups."

        split_val = float(valid.median())
        high_mask = valid > split_val

        # If median equals max value, split at < median instead so both bins are non-empty
        if high_mask.sum() == 0 and (valid < split_val).sum() > 0:
            low_op, high_op = "<", ">="
        else:
            low_op, high_op = "<=", ">"

        split_str = f"{split_val:g}"
        low_label = f"{attr} {low_op} {split_str}"
        high_label = f"{attr} {high_op} {split_str}"

        def _assign_bin(v):
            if pd.isna(v):
                return low_label
            try:
                fv = float(v)
                if high_op == ">=":
                    return high_label if fv >= split_val else low_label
                else:
                    return high_label if fv > split_val else low_label
            except (ValueError, TypeError):
                return low_label

        binned = series.apply(_assign_bin)
        desc = (
            f"'{attr}' has {valid.nunique()} unique numeric values. "
            f"Automatically partitioned by median ({split_str}): "
            f"'{low_label}' / '{high_label}'."
        )
        return binned, desc

    def _compute_attr_metrics(
        self,
        sub: pd.DataFrame,
        attr: str,
        target_col: str,
        label_col: str,
        use_predictions: bool = False,
    ) -> dict[str, Any]:
        """Compute all metrics for a single sensitive attribute."""
        feature_cols = [c for c in sub.columns if c not in [target_col, attr, label_col]]
        cols_to_select = list(dict.fromkeys([target_col, attr, label_col] + feature_cols))
        df_work = sub[cols_to_select].copy()
        df_work = df_work.dropna(subset=[target_col, attr])

        warnings_list: list[str] = []
        binning_applied: bool = False
        binning_note: str | None = None

        # ── Step 1: Binarize target ──────────────────────────────────────────
        y = df_work[label_col]
        if set(y.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            y_bin = y.astype(int)
        elif y.nunique() == 2:
            vals = sorted(y.unique())
            y_bin = y.map({vals[0]: 0, vals[1]: 1})
        elif pd.api.types.is_numeric_dtype(y):
            median = y.median()
            y_bin = (y > median).astype(int)
        else:
            y_bin = (y == y.mode()[0]).astype(int)
        df_work['__target__'] = y_bin

        # ── Step 2: Handle sensitive attribute grouping ──────────────────────
        # If numeric with high cardinality → bin into two meaningful groups.
        raw_n_unique = df_work[attr].nunique()
        if (
            pd.api.types.is_numeric_dtype(df_work[attr])
            and raw_n_unique > self.CARDINALITY_BIN_THRESHOLD
        ):
            binned_series, binning_note = self._bin_continuous_attr(df_work[attr], attr)
            df_work['__sens_raw__'] = binned_series
            binning_applied = True
            warnings_list.append(binning_note)
        else:
            df_work['__sens_raw__'] = df_work[attr].astype(str)

        # ── Step 3: Encode sensitive groups as integers ──────────────────────
        le = LabelEncoder()
        df_work['__sens__'] = le.fit_transform(df_work['__sens_raw__'])
        group_names = {i: name for i, name in enumerate(le.classes_)}

        # ── Step 4: Compute group positive rates ────────────────────────────
        groups = df_work['__sens__'].unique()
        group_stats: dict[str, Any] = {}
        small_sample_groups: list[str] = []

        for g in groups:
            mask = df_work['__sens__'] == g
            group_name = group_names[g]
            count = int(mask.sum())
            if count <= 1:
                warnings_list.append(
                    f"Group '{group_name}' has only {count} member(s) — excluded from metrics."
                )
                continue
            pos_count = int(df_work.loc[mask, '__target__'].sum())
            pos_rate = float(df_work.loc[mask, '__target__'].mean())
            group_stats[str(group_name)] = {
                "count": count,
                "positive_count": pos_count,
                "positive_rate": round(pos_rate, 4),
                "pct_of_total": round(count / len(df_work) * 100, 1),
            }
            # Small-sample warning
            if count < 30:
                small_sample_groups.append(group_name)
                warnings_list.append(
                    f"Group '{group_name}' has only {count} records. "
                    f"Statistical estimates for this group may be unstable. "
                    f"Observed disparity should not be interpreted as definitive "
                    f"evidence of systematic discrimination."
                )

        if len(group_stats) < 2:
            return {
                "error": f"'{attr}' has fewer than 2 valid groups after filtering.",
                "group_stats": group_stats,
                "binning_applied": binning_applied,
                "binning_note": binning_note,
            }

        # ── Step 5: Identify privileged / unprivileged groups ───────────────
        # Privileged = highest positive rate (most advantaged outcome).
        # For a binned attribute, group status is determined strictly
        # by data distribution, not hardcoded — intentionally data-driven.
        priv_name   = max(group_stats, key=lambda g: group_stats[g]["positive_rate"])
        unpriv_name = min(group_stats, key=lambda g: group_stats[g]["positive_rate"])
        priv_rate   = group_stats[priv_name]["positive_rate"]
        unpriv_rate = group_stats[unpriv_name]["positive_rate"]

        # ── Step 6: Dataset-level metrics (always available) ─────────────────
        # SPD = positive_rate(unprivileged) - positive_rate(privileged)
        # Convention: negative = unprivileged group receives fewer positives.
        SPD = round(unpriv_rate - priv_rate, 4)

        # DI  = positive_rate(unprivileged) / positive_rate(privileged)
        if priv_rate > 0:
            DI = round(unpriv_rate / priv_rate, 4)
        elif unpriv_rate == 0:
            DI = 1.0  # both groups have 0 positives — no disparity
        else:
            DI = None  # denominator is 0 but numerator is not — undefined
            warnings_list.append(
                "Disparate Impact is undefined: the privileged group has 0 positive outcomes. "
                "Check that your target column is correctly encoded."
            )

        # ── Step 7: Model-level metrics (only when predictions exist) ────────
        # EOD and AOD REQUIRE y_pred. Without external predictions these
        # metrics are UNAVAILABLE — represented as None, never fabricated.
        EOD: float | None = None
        AOD: float | None = None

        if use_predictions and label_col != target_col:
            # External model predictions available — compute vs ground truth
            y_true = df_work[target_col]
            if set(y_true.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                y_true_bin = y_true.astype(int)
            elif y_true.nunique() == 2:
                vals = sorted(y_true.unique())
                y_true_bin = y_true.map({vals[0]: 0, vals[1]: 1})
            elif pd.api.types.is_numeric_dtype(y_true):
                median = y_true.median()
                y_true_bin = (y_true > median).astype(int)
            else:
                y_true_bin = (y_true == y_true.mode()[0]).astype(int)
            df_work['__truth__'] = y_true_bin

            priv_encoded   = next(k for k, v in group_names.items() if str(v) == priv_name)
            unpriv_encoded = next(k for k, v in group_names.items() if str(v) == unpriv_name)
            eod, aod = self._equal_opportunity_encoded(
                df_work, '__truth__', '__target__', '__sens__',
                priv_encoded, unpriv_encoded
            )
            EOD = round(eod, 4) if eod is not None else None
            AOD = round(aod, 4) if aod is not None else None
        # else: EOD = None, AOD = None — correct, no fabrication

        # ── Step 8: Extreme-value sanity check ───────────────────────────────
        spd_abs = abs(SPD)
        if spd_abs > 0.99 and (DI is not None and DI < 0.01):
            warnings_list.append(
                "Metrics look extreme. Check that the target column is correctly "
                "binary and the sensitive attribute has meaningful variation."
            )

        # ── Step 9: Bootstrapped CI for SPD ──────────────────────────────────
        ci, statistically_significant = self._bootstrap_spd_ci(
            df_work, '__sens__', '__target__',
            next(k for k, v in group_names.items() if str(v) == priv_name),
            next(k for k, v in group_names.items() if str(v) == unpriv_name),
        )

        return {
            "privileged_group":   str(priv_name),
            "unprivileged_group": str(unpriv_name),
            "group_stats":  group_stats,
            "spd":  SPD,
            "di":   DI,
            "eod":  EOD,   # None when no predictions
            "aod":  AOD,   # None when no predictions
            # Upper-case aliases for backwards compatibility
            "SPD":  SPD,
            "DI":   DI,
            "EOD":  EOD,
            "AOD":  AOD,
            "severity":        self._severity(SPD),
            "legal_flag":      (DI is not None and DI < 0.8),
            "bootstrapped_ci": ci,
            "statistically_significant": statistically_significant,
            "warnings":         warnings_list,
            "binning_applied":  binning_applied,
            "binning_note":     binning_note,
            "raw_cardinality":  raw_n_unique,
            "small_sample_groups": small_sample_groups,
            "eod_available":    EOD is not None,
            "aod_available":    AOD is not None,
            "metrics_mode":     "model_level" if use_predictions else "dataset_level",
        }

    def _equal_opportunity_encoded(
        self,
        sub: pd.DataFrame,
        truth_col: str,
        target_col: str,
        sens_col: str,
        privileged: Any,
        unprivileged: Any,
    ) -> tuple[float | None, float | None]:
        def tpr_fpr(mask: pd.Series) -> tuple[float, float]:
            grp    = sub[mask]
            actual = grp[truth_col]
            pred   = grp[target_col]
            tp = int(((pred == 1) & (actual == 1)).sum())
            fn = int(((pred == 0) & (actual == 1)).sum())
            fp = int(((pred == 1) & (actual == 0)).sum())
            tn = int(((pred == 0) & (actual == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return tpr, fpr

        priv_mask   = sub[sens_col] == privileged
        unpriv_mask = sub[sens_col] == unprivileged

        tpr_priv,   fpr_priv   = tpr_fpr(priv_mask)
        tpr_unpriv, fpr_unpriv = tpr_fpr(unpriv_mask)

        # EOD = TPR(unprivileged) - TPR(privileged)  [negative = disadvantaged]
        eod = tpr_unpriv - tpr_priv
        # AOD = 0.5 * [(TPR_u - TPR_p) + (FPR_u - FPR_p)]
        aod = ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv)) / 2.0

        return float(eod), float(aod)

    # ── Bootstrapped CI ──────────────────────────────────────────────────────

    def _bootstrap_spd_ci(
        self,
        sub: pd.DataFrame,
        attr: str,
        label_col: str,
        privileged: Any,
        unprivileged: Any,
    ) -> tuple[dict[str, float], bool]:
        """
        Compute 95% bootstrapped confidence interval for SPD.

        Returns (ci_dict, statistically_significant).
        """
        rng = np.random.default_rng(self.BOOTSTRAP_SEED)
        n   = len(sub)
        spd_samples: list[float] = []

        for _ in range(self.BOOTSTRAP_N):
            sample    = sub.iloc[rng.integers(0, n, size=n)]
            priv_rate   = float(sample.loc[sample[attr] == privileged,   label_col].mean())
            unpriv_rate = float(sample.loc[sample[attr] == unprivileged, label_col].mean())
            if np.isnan(priv_rate) or np.isnan(unpriv_rate):
                continue
            # Match sign convention: unprivileged - privileged
            spd_samples.append(unpriv_rate - priv_rate)

        if len(spd_samples) < 10:
            return {"low_95": None, "high_95": None}, False

        low_95  = float(np.percentile(spd_samples, 2.5))
        high_95 = float(np.percentile(spd_samples, 97.5))
        # Statistically significant if CI does NOT cross zero
        significant = not (low_95 <= 0 <= high_95)

        return (
            {"low_95": round(low_95, 4), "high_95": round(high_95, 4)},
            significant,
        )

    # ── Severity & Grade ─────────────────────────────────────────────────────

    @staticmethod
    def _derive_grade_and_severity(audit_score: float):
        if audit_score >= 85:
            return "A", "low",    "Fair",           "#22C55E"
        elif audit_score >= 70:
            return "B", "low",    "Minor Issues",   "#84CC16"
        elif audit_score >= 50:
            return "C", "medium", "Moderate Bias",  "#F59E0B"
        else:
            return "F", "high",   "High Bias",      "#EF4444"

    @staticmethod
    def _get_overall_severity(audit_score: float) -> str:
        return BiasEngine._derive_grade_and_severity(audit_score)[1]

    @staticmethod
    def _get_grade(audit_score: float) -> str:
        return BiasEngine._derive_grade_and_severity(audit_score)[0]

    @staticmethod
    def _severity(spd: float) -> str:
        """Per-attribute severity based on SPD thresholds."""
        abs_spd = abs(spd)
        if abs_spd < 0.1:
            return "low"
        if abs_spd < 0.2:
            return "medium"
        return "high"

    @staticmethod
    def _grade(score: float) -> str:
        """Legacy method kept for backwards compat."""
        return BiasEngine._get_grade(score)

    def _compute_audit_score(self, metrics_per_attr: dict) -> float:
        """
        Compute overall audit score (0-100, higher is fairer).

        IMPORTANT: EOD and AOD are only included in the penalty when they
        are genuinely available (not None).  Unavailable metrics do NOT
        contribute a zero penalty (which would incorrectly boost the score)
        and do NOT contribute a maximum penalty (which would incorrectly
        penalise datasets without predictions).
        """
        if not metrics_per_attr:
            return 100.0
        penalties = []
        for attr, m in metrics_per_attr.items():
            if "error" in m:
                continue
            spd = abs(m.get("spd", m.get("SPD", 0)) or 0)
            di  = m.get("di", m.get("DI", 1.0))
            eod = m.get("eod", m.get("EOD"))  # will be None if dataset-level

            # SPD penalty: 0.1→15pts, 0.2→35pts, 0.3→55pts, 0.5→80pts
            spd_penalty = min(80, spd * 160)

            # DI penalty: only when below 0.8 legal threshold
            if di is None:
                di_penalty = 0  # DI unavailable — no penalty contribution
            else:
                di_penalty = min(45, max(0, (0.8 - di) * 75)) if di < 0.8 else 0

            # EOD penalty: ONLY when EOD is a real measurement (not None)
            # Never treat None as 0.
            if eod is not None and not (isinstance(eod, float) and np.isnan(eod)):
                eod_penalty = min(20, abs(eod) * 60)
            else:
                eod_penalty = 0  # Not available — excluded from scoring

            penalties.append(spd_penalty + di_penalty + eod_penalty)

        if not penalties:
            return 100.0

        total_penalty = sum(penalties) / max(len(penalties), 1)
        score = max(0.0, min(100.0, 100.0 - total_penalty))
        return float(round(score, 1))
